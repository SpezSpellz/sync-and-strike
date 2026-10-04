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

    /// <summary>Lambda used to build the CRITIC's regression target, separate from the policy's.
    ///
    /// Decoupled-GAE, from arXiv:2503.01491. With lambda &lt; 1 the value target is built from the
    /// critic's own current prediction, which is semi-gradient descent and is self-reinforcing: a critic
    /// that starts near zero builds a target near zero and keeps predicting near zero. Setting this to
    /// 1.0 makes the critic regress onto accumulated rewards instead - plain, stable gradient descent.
    ///
    /// Measured on this game before the fix: the critic's output std sat at 0.01 against a return std
    /// of 1.42, a ratio of about 1:100. The critic was not under-capacity; it was being trained on a
    /// target it had itself collapsed.
    ///
    /// The policy keeps <see cref="lambda"/>. The paper proves (Eq. 8) that using a critic fitted with a
    /// different lambda introduces no additional bias in the policy gradient, so the policy retains the
    /// variance reduction it needs.</summary>
    public float valueLambda = 1f;
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

    /// <summary>Variance of the critic's predictions over the last batch. Near zero means the head has
    /// collapsed to a constant and the critic is not participating in learning at all.</summary>
    public float ValueVariance { get; private set; }
    public float ValueStd { get; private set; }
    public float ReturnStd { get; private set; }

    /// <summary>Correlation between predicted value and return over the last batch. Healthy PPO runs
    /// sit well above 0.5 once the critic has warmed up.</summary>
    public float Correlation { get; private set; }

    /// <summary>Fraction of transitions that received multi-step GAE credit rather than being treated
    /// as a chain start. Near 0 means the recursion is severed almost everywhere; near 1 means the
    /// trajectories are intact. A healthy 2v1 shared brain sits high, because the buffer interleaves
    /// fighters but each fighter's own run is contiguous after sorting.</summary>
    public float ChainedFraction { get; private set; }

    /// <summary>ValueStd / ReturnStd. See the note at the computation site for how to read it.</summary>
    public float RatioValueToReturn { get; private set; }

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

        // Order in which transitions were collected. 2v1 shares one brain across Player and Companion,
        // and TurnManager resolves them in list order, so the buffer arrives INTERLEAVED: P, C, P, C...
        // GAE is a recursion along a trajectory, so walking this array in collection order would
        // discount each fighter's advantage toward the OTHER fighter's next state.
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;

        // Stable sort by (agentId, episode) so each fighter's own transitions form one contiguous run
        // in collection order. Sorting is what makes the recursion valid - the previous attempt instead
        // refused to continue the chain whenever the next index belonged to a different fighter, which
        // in a strictly alternating buffer is almost every index. That quietly reduced GAE to one-step
        // TD, which made explained variance look repaired (-1.6 -> 0.00) while the critic was in fact
        // being trained to predict single-turn rewards that barely vary. The metric improved because
        // the question got easier, not because the critic improved.
        // Array.Sort is NOT stable, so an explicit sequence stamp is required. Without it two transitions from
        // the same (agent, episode) pair can be reordered relative to each other, which reverses the
        // direction GAE walks them in and turns the recursion into a sum of unrelated terms. That is a
        // silent failure: no warning, no NaN, just advantages that mean nothing.
        //
        // The comparators must agree: the boundary tests below compare agentId and episode, so a
        // mismatch between how transitions are SORTED and how GROUPS are detected would split one group
        // in two places and re-introduce the bug the sort exists to remove.
        var seq = new int[n];
        for (int i = 0; i < n; i++) seq[i] = i;
        Array.Sort(order, (a, b) =>
        {
            var ta = buffer[a]; var tb = buffer[b];
            int c = ta.agentId.CompareTo(tb.agentId);
            if (c != 0) return c;
            c = ta.episode.CompareTo(tb.episode);
            if (c != 0) return c;
            return seq[a].CompareTo(seq[b]);
        });

        // Bootstrap value for the last transition of each group, evaluated once per group.
        var bootValue = new float[n];
        for (int k = 0; k < n; k++)
        {
            int idx = order[k];
            bool lastOfGroup = k == n - 1
                               || buffer[order[k + 1]].agentId != buffer[idx].agentId
                               || buffer[order[k + 1]].episode != buffer[idx].episode;
            if (lastOfGroup && !buffer[idx].done) bootValue[k] = net.Forward(buffer[idx].nextObs);
        }

        // Walk each group backwards, maintaining a separate GAE accumulator per group.
        var groupGae = new Dictionary<long, float>();
        int chained = 0;
        for (int k = n - 1; k >= 0; k--)
        {
            int idx = order[k];
            var tr = buffer[idx];
            float nextValue = tr.done ? 0f : bootValue[k];
            float delta = tr.reward + hyper.gamma * nextValue - tr.value;

            long key = ((long)tr.agentId << 32) ^ (uint)tr.episode;
            bool firstOfGroupBackwards = k == n - 1
                                        || buffer[order[k + 1]].agentId != tr.agentId
                                        || buffer[order[k + 1]].episode != tr.episode;
            float prev = 0f;
            if (!firstOfGroupBackwards) groupGae.TryGetValue(key, out prev);

            // Policy advantages keep the biased, variance-reduced lambda; the critic's regression
            // target uses valueLambda (1.0), which makes it regress onto accumulated rewards rather
            // than onto its own current output.
            float gaePolicy = delta + (firstOfGroupBackwards ? 0f : hyper.gamma * hyper.lambda * prev);
            float gaeValue = delta + (firstOfGroupBackwards ? 0f : hyper.gamma * hyper.valueLambda * prev);
            groupGae[key] = gaePolicy;
            if (!firstOfGroupBackwards) chained++;

            advantage[idx] = gaePolicy;
            // returns must be the CRITIC's target, so they follow valueLambda. Using the policy's
            // lambda here would re-import exactly the self-referential target Decoupled-GAE removes.
            returns[idx] = gaeValue + tr.value;
        }

        // Fraction of transitions that actually received multi-step credit. Near 0 means the recursion
        // is being severed almost everywhere - which is what a mis-ordered or over-eager boundary guard
        // looks like, and is invisible in explained variance because a one-step target is trivially
        // predictable. This is the check whose absence let the previous bug masquerade as a fix.
        ChainedFraction = n > 1 ? (float)chained / (n - 1) : 0f;

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

            // Variance of the critic's own predictions, and the correlation with the returns.
            //
            // Explained variance alone cannot distinguish its two failure modes. A critic that predicts
            // a constant and one that predicts the right SHAPE with the wrong offset both score ~0, but
            // they need completely different fixes - the first is capacity or dead units, the second is
            // a bias/scale problem that a centring term would solve. This run sat at EV ~ 0 through three
            // separate fixes (wider trunk, isolated trunk, independent clipping), so the next step is to
            // measure which of the two it is rather than guess a fourth.
            //
            // RatioValueToReturn is the discriminator:
            //   ~0.0  -> the head has collapsed to (nearly) a constant: no output variance at all.
            //   ~1.0  -> matching spread but uncorrelated: capacity or feature problem.
            //   >1.0  -> over-dispersed predictions, which the buffer's stale values would produce.
            float meanVal = 0f;
            for (int t = 0; t < n; t++) meanVal += buffer[t].value;
            meanVal /= n;
            float varPred = 0f, cov = 0f;
            for (int t = 0; t < n; t++)
            {
                float dPred = buffer[t].value - meanVal;
                float dRet2 = returns[t] - meanRet;
                varPred += dPred * dPred;
                cov += dPred * dRet2;
            }
            varPred /= n;
            cov /= n;
            ValueVariance = varPred;
            ReturnStd = Mathf.Sqrt(varRet);
            ValueStd = Mathf.Sqrt(varPred);
            Correlation = varRet > 1e-8f && varPred > 1e-8f ? cov / Mathf.Sqrt(varRet * varPred) : 0f;
            RatioValueToReturn = varRet > 1e-8f ? varPred / varRet : 0f;
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