using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A small clipped-surrogate PPO implementation for the companion's on-device policy.
///
/// One environment step is one turn. Transitions are (observation, chosen action, log-prob, value,
/// reward, done). The trainer buffers turns, computes GAE advantages, and runs a few epochs of
/// minibatch updates against the shared <see cref="NeuralNetwork"/>.
///
/// This deliberately has no external dependency (no ML-Agents, no Python) so the shipped build can
/// keep training the companion on the player's machine. The enemy stays frozen and never uses this.
/// </summary>
public class PPOTrainer
{
    public class Transition
    {
        public float[] obs;
        public float[] nextObs;
        public int action;
        public float logProb;   // log-prob at collection time (the PPO ratio denominator)
        public float value;     // value estimate at collection time
        public float reward;
        public bool done;

        // --- Hybrid policy ---
        //
        // The raw PRE-SQUASH samples, not the gameplay values. The Gaussian log-prob is a density over
        // the raw variable, so storing the squashed value would mean inverting the squash on every read,
        // and the clamping in that round trip would move the action the stored log-prob describes - which
        // makes the importance ratio quietly wrong rather than obviously wrong.
        public float[] rawZ;

        // Which fighter produced this transition, and which match.
        //
        // Needed because 2v1 puts TWO fighters on one shared brain, so the buffer interleaves two
        // independent trajectories. GAE walks the buffer as if it were a single sequence, which would
        // bootstrap each fighter's value toward the OTHER fighter's next state - a wrong value on both
        // sides, and therefore noise in every advantage. Evidence this was real: with one fighter per
        // brain the critic reached explained variance around -0.05, while the two-fighter shared brain
        // sat at -1.6 and worsening.
        public int agentId;
        public int episode;
    }

    [Serializable]
    public class Hyper
    {
        public float learningRate = 3e-4f;
        public float gamma = 0.99f;
        public float lambda = 0.95f;
        public float clipEpsilon = 0.2f;
        public float valueCoef = 0.5f;
        public float entropyCoef = 0.01f;
        public int epochs = 4;
        public int minibatch = 32;
        public int bufferSize = 512;

        /// <summary>
        /// Global-norm clip on the accumulated gradient. 0 disables it.
        ///
        /// Added after a training run produced NaN weights, which killed every policy in the process:
        /// one pathological minibatch produced a step large enough to overflow, and from there the
        /// network could never recover. Clipping bounds the step no matter how bad the batch is.
        /// </summary>
        public float maxGradNorm = 0.5f;
    }

    private readonly NeuralNetwork net;
    private readonly Hyper hyper;
    private readonly List<Transition> buffer = new List<Transition>();
    private readonly System.Random rng;
    private int[] shuffle;

    public PPOTrainer(NeuralNetwork net, Hyper hyper, System.Random rng)
    {
        this.net = net;
        this.hyper = hyper;
        this.rng = rng;
        shuffle = new int[hyper.bufferSize];
    }

    public int Buffered => buffer.Count;
    public int UpdateCount { get; private set; }

    // --- Update diagnostics --------------------------------------------------
    //
    // Recorded because "the policy stopped improving" has several very different causes and the
    // console cannot tell them apart. These are the numbers that distinguish them, and they are the
    // reason a tuning change can be judged rather than guessed at.

    /// <summary>Mean loss on the value head over the last update.</summary>
    public float ValueLoss { get; private set; }

    /// <summary>Mean surrogate policy loss over the last update.</summary>
    public float PolicyLoss { get; private set; }

    /// <summary>
    /// Mean |advantage| over the last update. Before normalisation this is the raw advantage scale;
    /// it is recorded because a value that is wildly large or tiny is a strong hint the reward scale
    /// and the value target disagree.
    /// </summary>
    public float AdvantageMagnitude { get; private set; }

    /// <summary>
    /// Fraction of samples whose importance ratio fell outside the clip range.
    ///
    /// Near 0 means the policy is barely moving per update (learning rate too low, or the advantage
    /// signal is noise). Near 1 means it is moving so far that clipping is doing the work and the
    /// gradient is mostly discarded. A healthy run sits loosely around 0.05 to 0.2.
    /// </summary>
    public float ClipFraction { get; private set; }

    /// <summary>
    /// Mean KL divergence between the collection-time and update-time distributions.
    ///
    /// The single best "is this update too aggressive" signal. It should stay well under ~0.02; if it
    /// climbs much higher the policy is taking steps far outside the region the samples described.
    /// </summary>
    public float ApproxKl { get; private set; }

    /// <summary>
    /// 1 - Var(returns) / Var(returns predicted by the critic).
    ///
    /// The critic's quality score. 1 is perfect, 0 is no better than predicting the mean, and
    /// NEGATIVE means the critic is actively worse than a constant predictor - in which case every
    /// advantage estimate is noise and no amount of reward tuning will help until it is fixed.
    /// </summary>
    public float ExplainedVariance { get; private set; }

    /// <summary>Total parameter-update steps applied, for spotting a stalled learner.</summary>
    public int MinibatchCount { get; private set; }

    /// <summary>
    /// Gradient steps skipped because the accumulated gradient was not finite.
    ///
    /// This counter exists because the failure it prevents is invisible: a single NaN gradient
    /// permanently poisons the weights, and the policy then keeps sampling happily while always making
    /// the same choice. A non-zero value means the run was saved from that, and is worth investigating
    /// rather than ignoring - it means something upstream produced a NaN.
    /// </summary>
    public int NonFiniteSteps => net.NonFiniteGradientSteps;

    /// <summary>POLICY gradient steps rescaled by the norm clip. High values mean unstable rewards.</summary>
    public int ClippedSteps => net.ClippedGradSteps;

    /// <summary>CRITIC steps rescaled. Compared against ClippedSteps: a policy that clips constantly while
    /// the critic rarely does is the signature of a critic that cannot keep up.</summary>
    public int ClippedCriticSteps => net.ClippedCriticGradSteps;

    /// <summary>
    /// Largest weight magnitude in the network. A leading indicator: this run reached the hundreds just
    /// before the weights went NaN, so a rising value is a run about to die while it is still fixable.
    /// </summary>
    public float MaxAbsWeight => net.MaxAbsWeight;

    /// <summary>
    /// Record one transition.
    ///
    /// Non-finite inputs are REJECTED rather than stored. This is the last line of defence before NaN
    /// can reach the gradient: a single NaN reward or value poisons every advantage computed from it,
    /// and because the rollout buffer is a sliding window that NaN then survives for hundreds of
    /// updates, long after the turn that caused it. Rejecting the transition loses one sample;
    /// accepting it loses the network.
    ///
    /// RejectedTransitionCount makes the rejections visible, because a run quietly dropping samples is
    /// otherwise indistinguishable from a run with nothing to learn.
    /// </summary>
    public void Add(float[] obs, float[] nextObs, int action, float logProb, float value, float reward,
                    bool done, float[] rawZ = null, int agentId = 0, int episode = 0)
    {
        if (!IsFinite(reward) || !IsFinite(value) || !IsFinite(logProb)
            || obs == null || nextObs == null || !AllFinite(obs) || !AllFinite(nextObs)
            || (rawZ != null && !AllFinite(rawZ)))
        {
            RejectedTransitionCount++;
            return;
        }

        buffer.Add(new Transition
        {
            obs = obs, nextObs = nextObs, action = action, rawZ = rawZ,
            logProb = logProb, value = value, reward = reward, done = done,
            agentId = agentId, episode = episode,
        });
        if (buffer.Count > hyper.bufferSize) buffer.RemoveAt(0);
    }

    /// <summary>Transitions discarded for carrying a non-finite value. Should stay at 0.</summary>
    public int RejectedTransitionCount { get; private set; }

    private static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

    private static bool AllFinite(float[] v)
    {
        for (int i = 0; i < v.Length; i++) if (!IsFinite(v[i])) return false;
        return true;
    }

    /// <summary>Apply a late reward adjustment (e.g. the player's vote) to the most recent turn.</summary>
    public void AdjustLastReward(float delta)
    {
        if (buffer.Count > 0) buffer[buffer.Count - 1].reward += delta;
    }

    public bool ReadyToUpdate() => buffer.Count >= hyper.minibatch;

    /// <summary>
    /// Epoch counter, stamped onto every transition.
    ///
    /// Bumped when a match ends so GAE does not chain across the reset: the final states of one match
    /// are not the predecessors of the opening states of the next.
    /// </summary>
    public int CurrentEpisode { get; private set; }

    public void BeginEpisode() => CurrentEpisode++;

    /// <summary>
    /// True when the network itself has been poisoned by non-finite weights.
    ///
    /// Once this is true the policy is dead: it keeps sampling, so nothing looks wrong, but the
    /// distribution is all-NaN and every action resolves to the same fallback. Callers should reload
    /// the last good weights rather than keep training.
    /// </summary>
    public bool NetworkIsPoisoned => !net.WeightsAreFinite();

    /// <summary>Run the PPO update over the current buffer. Consumes nothing; buffer persists until trimmed.</summary>
    public void Update()
    {
        int n = buffer.Count;
        if (n < 2) return;

        UpdateCount++;

        // --- GAE advantages ---
        var advantage = new float[n];
        var returns = new float[n];
        float gae = 0f;
        for (int t = n - 1; t >= 0; t--)
        {
            var tr = buffer[t];
            float nextValue = 0f;
            if (!tr.done) nextValue = net.Forward(tr.nextObs);
            float delta = tr.reward + hyper.gamma * nextValue - tr.value;

            // The continuation term may only carry over from the transition that actually FOLLOWS this
            // one in the same fighter's own trajectory. With a shared 2v1 brain the buffer interleaves
            // two fighters, so without this guard each fighter's advantage is discounted toward the
            // other's next state - corrupting both, silently.
            bool continues = !tr.done && t < n - 1
                             && buffer[t + 1].agentId == tr.agentId
                             && buffer[t + 1].episode == tr.episode;
            gae = delta + (continues ? hyper.gamma * hyper.lambda * gae : 0f);
            advantage[t] = gae;
            returns[t] = gae + tr.value;
        }

        // Diagnostics are computed on the RAW advantages and returns, before normalisation. Computing
        // them afterwards would be meaningless: normalised advantages are zero-mean unit-variance by
        // construction, so they could not distinguish "the critic learned nothing" from "the critic
        // learned fine".
        {
            float advMag = 0f;
            for (int t = 0; t < n; t++) advMag += Mathf.Abs(advantage[t]);
            AdvantageMagnitude = advMag / n;

            // Explained variance: 1 - Var(returns - predicted) / Var(returns).
            float meanRet = 0f;
            for (int t = 0; t < n; t++) meanRet += returns[t];
            meanRet /= n;
            float varRet = 0f, varErr = 0f;
            for (int t = 0; t < n; t++)
            {
                float dRet = returns[t] - meanRet;
                varRet += dRet * dRet;
                float dErr = returns[t] - buffer[t].value;
                varErr += dErr * dErr;
            }
            varRet /= n;
            varErr /= n;
            ExplainedVariance = varRet > 1e-8f ? 1f - varErr / varRet : 0f;
        }

        // Normalise advantages for a stable step size.
        float mean = 0f;
        for (int t = 0; t < n; t++) mean += advantage[t];
        mean /= n;
        float varSum = 0f;
        for (int t = 0; t < n; t++) { float d = advantage[t] - mean; varSum += d * d; }
        float std = Mathf.Sqrt(varSum / n) + 1e-8f;
        for (int t = 0; t < n; t++) advantage[t] = (advantage[t] - mean) / std;

        // --- Clipped PPO epochs ---
        float klSum = 0f, sampleTotal = 0f, clippedCount = 0f;
        float valueLossSum = 0f, policyLossSum = 0f;
        int minibatchCount = 0;

        // Scalar-head scratch, allocated once per update. Sized from the network, so a move-only network
        // gets zero-length buffers and the scalar path costs nothing.
        var heads = HeadsFor(net.ScalarCount);
        var dMuBuf = new float[net.ScalarCount];
        var dLogSdBuf = new float[net.ScalarCount];
        for (int epoch = 0; epoch < hyper.epochs; epoch++)
        {
            Shuffle(n);
            for (int start = 0; start < n; start += hyper.minibatch)
            {
                int end = Mathf.Min(start + hyper.minibatch, n);
                int count = end - start;
                if (count == 0) continue;
                net.ClearGradients();
                float batchValueLoss = 0f, batchPolicyLoss = 0f;

                for (int i = start; i < end; i++)
                {
                    var tr = buffer[shuffle[i]];
                    // One forward pass feeds both the value head and the log-prob. Calling Forward
                    // twice (once here, once inside LogProb) would double the cost and, worse, read
                    // logits from a second pass whose result is thrown away.
                    float value = net.Forward(tr.obs);
                    // One forward pass feeds the value head, the categorical log-prob AND the scalar
                    // heads. Re-calling Forward per term would read means left over from another pass.
                    float newLogProb = JointLogProbAfterForward(tr.action, tr.rawZ, heads);
                    float ratio = Mathf.Exp(newLogProb - tr.logProb);

                    // Textbook clipping mask: when the clipped branch is the active (smaller) term,
                    // the ratio is pinned and contributes no policy gradient.
                    float A = advantage[i];
                    bool useClipped = (A > 0f && ratio > 1f + hyper.clipEpsilon)
                                    || (A < 0f && ratio < 1f - hyper.clipEpsilon);

                    // Diagnostics accumulate over the FIRST epoch only. Later epochs deliberately move
                    // the policy far from where the samples were collected, so averaging across them
                    // would report a KL the update never actually started from.
                    if (epoch == 0)
                    {
                        sampleTotal++;
                        if (useClipped) clippedCount++;
                        // Schulman's low-variance KL estimator: ratio is exp(newLogProb - oldLogProb).
                        klSum += ratio - 1f - (newLogProb - tr.logProb);
                    }

                    float dLogProb = useClipped ? 0f : A * ratio;
                    float dValue = hyper.valueCoef * (value - returns[i]);

                    // Scalar heads get the SAME coefficient as the categorical head, because the ratio
                    // above is over the JOINT log-prob: one number scales every term. Clipping applies to
                    // the joint too, so a clipped sample contributes nothing to any head.
                    float dScalarMu = useClipped ? 0f : A * ratio;
                    if (heads != null)
                    {
                        ScalarCoefficients(tr.rawZ, heads, dMuBuf, dLogSdBuf);
                        for (int s = 0; s < dMuBuf.Length; s++)
                        {
                            dMuBuf[s] *= dScalarMu;
                            dLogSdBuf[s] *= dScalarMu;
                        }
                    }
                    else { Array.Clear(dMuBuf, 0, dMuBuf.Length); Array.Clear(dLogSdBuf, 0, dLogSdBuf.Length); }

                    // Losses recorded from the FIRST epoch only, matching the KL and clip-fraction
                    // scope. By later epochs the policy has already moved well away from the collected
                    // samples, so those numbers describe a region the update did not start from.
                    if (epoch == 0)
                    {
                        batchValueLoss += (value - returns[i]) * (value - returns[i]);
                        batchPolicyLoss += -A * ratio;
                    }

                    net.Backprop(tr.obs, tr.action, dLogProb, dValue, hyper.entropyCoef,
                                 dMuBuf, dLogSdBuf);
                }
                if (epoch == 0 && count > 0)
                {
                    valueLossSum += batchValueLoss / count;
                    policyLossSum += batchPolicyLoss / count;
                    minibatchCount++;
                }
                net.ApplyGradients(hyper.learningRate / count, hyper.maxGradNorm);
            }
        }

        ApproxKl = sampleTotal > 0f ? klSum / sampleTotal : 0f;
        ClipFraction = sampleTotal > 0f ? clippedCount / sampleTotal : 0f;
        MinibatchCount = minibatchCount;
        ValueLoss = minibatchCount > 0 ? valueLossSum / minibatchCount : 0f;
        PolicyLoss = minibatchCount > 0 ? policyLossSum / minibatchCount : 0f;
    }

    // Range descriptors for the scalar heads, shared by index with NeuralPolicy. Kept here so the
    // trainer can evaluate the same density the policy sampled from, without a policy reference.
    private static ContinuousHead[] HeadsFor(int count)
    {
        if (count <= 0) return null;
        var all = ActionScalars.CreateAll();
        if (count > all.Length) return all;
        var trimmed = new ContinuousHead[count];
        Array.Copy(all, trimmed, count);
        return trimmed;
    }

    // Joint log-prob, given a forward pass that has ALREADY been done for obs. Splitting it this way
    // matters: the minibatch loop needs the value and the log-prob from the SAME pass, and re-forwarding
    // would silently read logits and means from a different input.
    private float JointLogProbAfterForward(int action, float[] rawZ, ContinuousHead[] heads)
    {
        float lp = NeuralNetwork.LogSoftmax(net.Logits, action);
        if (heads == null || rawZ == null) return lp;
        var means = net.ScalarMeans;
        var logSds = net.ScalarLogSd;
        for (int s = 0; s < heads.Length && s < rawZ.Length; s++)
        {
            heads[s].mu = means[s];
            heads[s].logSd = logSds[s];
            lp += heads[s].LogProbRaw(rawZ[s]);
        }
        return lp;
    }

    // d(logProb)/d(mu_s) and d(logProb)/d(logSd_s) for the stored samples, given the CURRENT parameters.
    private void ScalarCoefficients(float[] rawZ, ContinuousHead[] heads, float[] dMu, float[] dLogSd)
    {
        if (heads == null || rawZ == null) return;
        var means = net.ScalarMeans;
        var logSds = net.ScalarLogSd;
        for (int s = 0; s < heads.Length && s < rawZ.Length; s++)
        {
            heads[s].mu = means[s];
            heads[s].logSd = logSds[s];
            dMu[s] = heads[s].DLogProbDMu(rawZ[s]);
            dLogSd[s] = heads[s].DLogProbDLogSd(rawZ[s]);
        }
    }

    private void Shuffle(int n)
    {
        if (shuffle.Length < n)
            shuffle = new int[Mathf.NextPowerOfTwo(Mathf.Max(n, 16))];
        for (int i = 0; i < n; i++) shuffle[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            int tmp = shuffle[i]; shuffle[i] = shuffle[j]; shuffle[j] = tmp;
        }
    }
}