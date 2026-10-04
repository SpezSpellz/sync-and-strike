using System;
using UnityEngine;

/// <summary>
/// A tiny MLP for the on-device PPO policy: input -> ReLU -> ReLU -> (categorical policy head,
/// N scalar Gaussian heads, scalar value head). Backprop is implemented by hand because we only
/// need a handful of gradients and want zero external dependencies so it can run in the shipped
/// build without ML-Agents or a Python runtime.
///
/// The policy is HYBRID: a categorical over (move x facing) plus one squashed Gaussian per scalar
/// (jump power/angle, DI power/angle). The joint log-prob is the sum of the categorical term and the
/// Gaussian terms, and that sum is what PPO's importance ratio compares. See ContinuousHead for why
/// those terms need a Jacobian correction - omitting it biases every continuous update silently, with
/// no NaN and nothing in the log.
///
/// The enemy never uses this. It is only for the companion, which continues to train on the
/// player's machine; the offline ML-Agents policy for the enemy is a separate, frozen artifact.
/// </summary>
[Serializable]
public class NeuralNetwork
{
    [Serializable]
    public class Weights
    {
        public int input, hidden, actions;

        /// <summary>Width of the critic's own trunk. Separate from hidden because the two heads want
        /// different capacity; 0 in files written before the widths diverged.</summary>
        public int valueHidden;
        public float[] w1, b1, w2, b2, wp, bp, wv, bv;

        /// <summary>Continuous-head output weights and biases, plus the state-independent log-sd per
        /// head. Null in files written before the continuous heads existed, which is why Import
        /// validates rather than assuming they are present.</summary>
        public float[] wmu, bmu, logSd;

        // The critic trunk, kept separate from the policy's. See the field comment.
        public float[] vw1, vb1, vw2, vb2;
    }

    public int InputSize { get; }
    public int HiddenSize { get; }
    public int ActionCount { get; }

    /// <summary>Width of the critic trunk, independent of <see cref="HiddenSize"/>.
    ///
    /// The critic has to integrate a reward stream over a ~100-turn effective horizon
    /// (gamma 0.99) and resolve small differences in expected outcome, while the policy mostly needs
    /// to rank discrete actions. Sizing them the same forced the critic to be the smaller of the two,
    /// and explained variance sat at ~0 - the critic merely matching a constant. Widening only the
    /// critic is cheap: it does not run on the inference path for the shipped enemy, where the frozen
    /// policy never asks for a value.</summary>
    public int ValueHiddenSize { get; }

    /// <summary>How many scalar (Gaussian) outputs this network has. Zero for the old
    /// move-only shape, which keeps Export/Import able to read files written before this existed.</summary>
    public int ScalarCount { get; }

    private float[] w1, b1, w2, b2, wp, bp, wv, bv;
    private float[] gw1, gb1, gw2, gb2, gwp, gbp, gwv, gbv;

    // The critic runs its OWN trunk (vw1/vb1 -> vw2/vb2 -> value) instead of reading the policy trunk.
    //
    // Sharing one trunk for both objectives was the likely cause of persistently NEGATIVE explained
    // variance. The policy gradient, plus an entropy bonus that exists solely to serve the policy,
    // reshapes the shared features on every update, and the critic - which needs features suited to
    // predicting a RETURN rather than discriminating actions - never gets to keep them. A critic worse
    // than predicting a constant makes every advantage estimate noise, which caps how well the policy
    // can learn however the reward is tuned.
    //
    // The cost is parameters and one extra forward pass. Both are negligible at this size, and the
    // isolation is what stops the two objectives fighting.
    private float[] vw1, vb1, vw2, vb2;
    private float[] gvw1, gvb1, gvw2, gvb2;
    private float[] vh1, vh2;

    // Continuous heads: a linear read-out of the trunk per scalar, plus a global log-sd each.
    private float[] wmu, bmu, logSd;
    private float[] gwmu, gbmu, glogSd;
    private float[] means;

    // Cached forward activations for the current Forward() call.
    private float[] h1, h2, logits;
    private float value;
    private float[] probs;

    // --- Adam optimiser state ---
    //
    // First and second moment estimates, one array per parameter tensor, plus a single step counter
    // shared by all of them (Adam's bias correction is global, not per tensor).
    //
    // These are deliberately NOT part of the exported Weights. They are optimiser state describing the
    // path taken to the current weights, not the weights themselves: shipping a frozen enemy wants the
    // policy, and resuming a run recreates its own moments from scratch. Persisting them would also
    // mean every saved file carries two extra copies of the network.
    private float[] mw1, vw1a, mb1, vb1a, mw2, vw2a, mb2, vb2a;
    private float[] mwp, vwpa, mbp, vbpa, mwv, vwva, mbv, vbva;
    private float[] mvw1, vvw1a, mvb1, vvb1a, mvw2, vvw2a, mvb2, vvb2a;
    private float[] mwmu, vwmu, mbmu, vbmu, mlogSd, vlogSd;
    private int adamStep;

    /// <summary>Adam steps taken. Exposed so telemetry can tell an untrained network from a stalled one.</summary>
    public int OptimizerSteps => adamStep;

    /// <summary>Discard the Adam moments, e.g. after importing weights that were trained by a different
    /// optimiser or a different network. The moments describe the old trajectory, so keeping them would
    /// push the first few steps in a direction that has nothing to do with the weights now loaded.</summary>
    public void ResetOptimizerState()
    {
        Array.Clear(mw1, 0, mw1.Length); Array.Clear(vw1a, 0, vw1a.Length);
        Array.Clear(mb1, 0, mb1.Length); Array.Clear(vb1a, 0, vb1a.Length);
        Array.Clear(mw2, 0, mw2.Length); Array.Clear(vw2a, 0, vw2a.Length);
        Array.Clear(mb2, 0, mb2.Length); Array.Clear(vb2a, 0, vb2a.Length);
        Array.Clear(mwp, 0, mwp.Length); Array.Clear(vwpa, 0, vwpa.Length);
        Array.Clear(mbp, 0, mbp.Length); Array.Clear(vbpa, 0, vbpa.Length);
        Array.Clear(mwv, 0, mwv.Length); Array.Clear(vwva, 0, vwva.Length);
        Array.Clear(mbv, 0, mbv.Length); Array.Clear(vbva, 0, vbva.Length);
        Array.Clear(mvw1, 0, mvw1.Length); Array.Clear(vvw1a, 0, vvw1a.Length);
        Array.Clear(mvb1, 0, mvb1.Length); Array.Clear(vvb1a, 0, vvb1a.Length);
        Array.Clear(mvw2, 0, mvw2.Length); Array.Clear(vvw2a, 0, vvw2a.Length);
        Array.Clear(mvb2, 0, mvb2.Length); Array.Clear(vvb2a, 0, vvb2a.Length);
        if (mwmu != null)
        {
            Array.Clear(mwmu, 0, mwmu.Length); Array.Clear(vwmu, 0, vwmu.Length);
            Array.Clear(mbmu, 0, mbmu.Length); Array.Clear(vbmu, 0, vbmu.Length);
            Array.Clear(mlogSd, 0, mlogSd.Length); Array.Clear(vlogSd, 0, vlogSd.Length);
        }
        adamStep = 0;
    }

    /// <param name="valueHidden">Width of the critic trunk. 0 (the default) means "same as
    /// <paramref name="hidden"/>", which is what every pre-existing call site meant.</param>
    public NeuralNetwork(int input, int hidden, int actions, int seed = 1234, int scalars = 0,
                         int valueHidden = 0)
    {
        InputSize = input; HiddenSize = hidden; ActionCount = actions; ScalarCount = scalars;
        ValueHiddenSize = valueHidden > 0 ? valueHidden : hidden;
        int hh = hidden * hidden;
        int vh = ValueHiddenSize;
        int vhh = vh * vh;
        w1 = new float[hidden * input]; b1 = new float[hidden];
        w2 = new float[hh]; b2 = new float[hidden];
        wp = new float[actions * hidden]; bp = new float[actions];
        wv = new float[vh]; bv = new float[1];
        gw1 = new float[w1.Length]; gb1 = new float[hidden];
        gw2 = new float[w2.Length]; gb2 = new float[hidden];
        gwp = new float[wp.Length]; gbp = new float[actions];
        gwv = new float[vh]; gbv = new float[1];
        h1 = new float[hidden]; h2 = new float[hidden];

        // Critic trunk: independent weights AND an independent width.
        vw1 = new float[vh * input]; vb1 = new float[vh];
        vw2 = new float[vhh]; vb2 = new float[vh];
        gvw1 = new float[vw1.Length]; gvb1 = new float[vh];
        gvw2 = new float[vw2.Length]; gvb2 = new float[vh];
        vh1 = new float[vh]; vh2 = new float[vh];
        logits = new float[actions]; probs = new float[actions];

        if (scalars > 0)
        {
            wmu = new float[scalars * hidden]; bmu = new float[scalars];
            logSd = new float[scalars];
            gwmu = new float[wmu.Length]; gbmu = new float[scalars];
            glogSd = new float[scalars];
            means = new float[scalars];
            for (int i = 0; i < scalars; i++) logSd[i] = ContinuousHead.DefaultLogSd;
        }

        var rng = new System.Random(seed);
        float s1 = (float)Math.Sqrt(2.0 / input);
        float s2 = (float)Math.Sqrt(2.0 / hidden);
        float s3 = 0.01f; // near-zero policy head so the initial policy is ~uniform
        for (int i = 0; i < w1.Length; i++) w1[i] = (float)(rng.NextDouble() * 2 - 1) * s1;
        for (int i = 0; i < w2.Length; i++) w2[i] = (float)(rng.NextDouble() * 2 - 1) * s2;
        for (int i = 0; i < wp.Length; i++) wp[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
        // Critic trunk uses normal hidden-layer init, not the near-zero policy init: the value head has
        // to produce a spread of predictions immediately, or every advantage looks identical and the
        // first few thousand updates are wasted learning a constant.
        //
        // sv2 is derived from the CRITIC's own fan-in. Reusing the policy's s2 would under-scale the
        // activations by sqrt(vh/hidden) once the widths differ - small here, but it is exactly the kind
        // of silent scale error that reads as "the critic just doesn't learn".
        float sv2 = (float)Math.Sqrt(2.0 / vh);
        for (int i = 0; i < vw1.Length; i++) vw1[i] = (float)(rng.NextDouble() * 2 - 1) * s1;
        for (int i = 0; i < vw2.Length; i++) vw2[i] = (float)(rng.NextDouble() * 2 - 1) * sv2;
        for (int i = 0; i < wv.Length; i++) wv[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
        // Scalar heads also start near zero, so the initial mu sits at 0 - the MIDPOINT of every
        // squashed range, because sigmoid(0) = 0.5. A mid-range jump or DI is a legal, harmless
        // default, which is what we want before warm start has cloned anything.
        for (int i = 0; i < (wmu?.Length ?? 0); i++) wmu[i] = (float)(rng.NextDouble() * 2 - 1) * s3;

        // Adam moments: allocated once, zero-initialised. Not filled with noise on purpose - at step 0
        // both moments are zero, bias correction divides by (1 - beta1^t) and (1 - beta2^t), and the
        // first update comes out at exactly lr * g regardless of what m and v started as.
        mw1 = new float[w1.Length]; vw1a = new float[w1.Length];
        mb1 = new float[b1.Length]; vb1a = new float[b1.Length];
        mw2 = new float[w2.Length]; vw2a = new float[w2.Length];
        mb2 = new float[b2.Length]; vb2a = new float[b2.Length];
        mwp = new float[wp.Length]; vwpa = new float[wp.Length];
        mbp = new float[bp.Length]; vbpa = new float[bp.Length];
        mwv = new float[wv.Length]; vwva = new float[wv.Length];
        mbv = new float[1]; vbva = new float[1];
        mvw1 = new float[vw1.Length]; vvw1a = new float[vw1.Length];
        mvb1 = new float[vb1.Length]; vvb1a = new float[vb1.Length];
        mvw2 = new float[vw2.Length]; vvw2a = new float[vw2.Length];
        mvb2 = new float[vb2.Length]; vvb2a = new float[vb2.Length];
        if (scalars > 0)
        {
            mwmu = new float[wmu.Length]; vwmu = new float[wmu.Length];
            mbmu = new float[bmu.Length]; vbmu = new float[bmu.Length];
            mlogSd = new float[logSd.Length]; vlogSd = new float[logSd.Length];
        }
    }

    public float Forward(float[] x)
    {
        for (int i = 0; i < HiddenSize; i++)
        {
            float s = b1[i];
            int off = i * InputSize;
            for (int j = 0; j < InputSize; j++) s += w1[off + j] * x[j];
            h1[i] = s > 0f ? s : 0f;
        }
        for (int i = 0; i < HiddenSize; i++)
        {
            float s = b2[i];
            int off = i * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) s += w2[off + j] * h1[j];
            h2[i] = s > 0f ? s : 0f;
        }
        float maxLogit = float.NegativeInfinity;
        for (int a = 0; a < ActionCount; a++)
        {
            float s = bp[a];
            int off = a * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) s += wp[off + j] * h2[j];
            logits[a] = s;
            if (s > maxLogit) maxLogit = s;
        }
        float sum = 0f;
        for (int a = 0; a < ActionCount; a++)
        {
            probs[a] = Mathf.Exp(logits[a] - maxLogit);
            sum += probs[a];
        }
        float inv = 1f / Mathf.Max(sum, 1e-8f);
        for (int a = 0; a < ActionCount; a++) probs[a] *= inv;

        // Scalar (Gaussian) read-outs, off the same trunk in ONE pass. Callers must not re-forward to
        // get these: they would read means left over from a different input.
        for (int s = 0; s < ScalarCount; s++)
        {
            float m = bmu[s];
            int off = s * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) m += wmu[off + j] * h2[j];
            means[s] = m;
        }

        // Critic: its own forward pass over its own trunk. Deliberately NOT reading h2 - see the field
        // comment. Callers must not assume this is the same computation as before; every gradient check
        // is re-run because of it.
        for (int i = 0; i < ValueHiddenSize; i++)
        {
            float s = vb1[i];
            int off = i * InputSize;
            for (int j = 0; j < InputSize; j++) s += vw1[off + j] * x[j];
            vh1[i] = s > 0f ? s : 0f;
        }
        for (int i = 0; i < ValueHiddenSize; i++)
        {
            float s = vb2[i];
            int off = i * ValueHiddenSize;
            for (int j = 0; j < ValueHiddenSize; j++) s += vw2[off + j] * vh1[j];
            vh2[i] = s > 0f ? s : 0f;
        }
        float v = bv[0];
        for (int j = 0; j < ValueHiddenSize; j++) v += wv[j] * vh2[j];
        value = v;
        return value;
    }

    public float[] Probabilities => probs;
    public float[] Logits => logits;
    public float Value => value;

    /// <summary>Raw pre-squash means for the scalar heads, from the last Forward().</summary>
    public float[] ScalarMeans => means;

    /// <summary>State-independent log standard deviations, one per scalar head.</summary>
    public float[] ScalarLogSd => logSd;

    public static float LogSoftmax(float[] logits, int a)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) sum += Mathf.Exp(logits[i] - max);
        return (logits[a] - max) - Mathf.Log(Mathf.Max(sum, 1e-8f));
    }

    public static int FirstLegal(int[] legalMask)
    {
        if (legalMask != null)
            for (int a = 0; a < legalMask.Length; a++) if (legalMask[a] != 0) return a;
        return 0;
    }

    public void ClearGradients()
    {
        Array.Clear(gw1, 0, gw1.Length); Array.Clear(gb1, 0, gb1.Length);
        Array.Clear(gw2, 0, gw2.Length); Array.Clear(gb2, 0, gb2.Length);
        Array.Clear(gwp, 0, gwp.Length); Array.Clear(gbp, 0, gbp.Length);
        Array.Clear(gwv, 0, gwv.Length); gbv[0] = 0f;
        Array.Clear(gvw1, 0, gvw1.Length); Array.Clear(gvb1, 0, gvb1.Length);
        Array.Clear(gvw2, 0, gvw2.Length); Array.Clear(gvb2, 0, gvb2.Length);
        if (ScalarCount > 0)
        {
            Array.Clear(gwmu, 0, gwmu.Length); Array.Clear(gbmu, 0, gbmu.Length);
            Array.Clear(glogSd, 0, glogSd.Length);
        }
    }

    /// <summary>
    /// Backprop for the scalar (Gaussian) heads.
    ///
    /// <paramref name="dScalarMu"/> is the coefficient on d(logProb)/d(mu) for the sampled raw value,
    /// and <paramref name="dScalarLogSd"/> the coefficient on d(logProb)/d(logSd). Both are per-head.
    ///
    /// The means' gradient flows back into the SHARED trunk, which is why this must be called before the
    /// trunk backward pass accumulates its total - see <see cref="Backprop"/>, which sums all three
    /// sources (policy, value, scalar) into one dh2. Computing the scalar contribution separately and
    /// forgetting to include it in dh2 is the same class of bug as the detached critic.
    /// </summary>
    private void BackpropScalars(float[] dScalarMu, float[] dScalarLogSd)
    {
        if (ScalarCount == 0 || dScalarMu == null) return;
        for (int s = 0; s < ScalarCount; s++)
        {
            float d = dScalarMu[s];
            if (d != 0f)
            {
                gbmu[s] += d;
                int off = s * HiddenSize;
                for (int j = 0; j < HiddenSize; j++) gwmu[off + j] += d * h2[j];
            }
            if (dScalarLogSd != null && dScalarLogSd[s] != 0f) glogSd[s] += dScalarLogSd[s];
        }
    }

    /// <summary>
    /// Accumulate the scalar heads' contribution into the trunk gradient, in place.
    ///
    /// Separate from <see cref="BackpropScalars"/> so the trunk's total dh2 can be assembled from the
    /// policy head, the value head and this, in one place, and so it is obvious that all three are
    /// required.
    /// </summary>
    private void ScalarTrunkGradient(float[] dScalarMu, float[] dh2)
    {
        if (ScalarCount == 0 || dScalarMu == null) return;
        for (int s = 0; s < ScalarCount; s++)
        {
            float d = dScalarMu[s];
            if (d == 0f) continue;
            int off = s * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) dh2[j] += d * wmu[off + j];
        }
    }

    /// <summary>
    /// Backprop for one PPO sample. <paramref name="dLogProb"/> is the scalar coefficient on the
    /// gradient of log pi(chosen) (advantage * clipped ratio), <paramref name="dValue"/> the
    /// coefficient on the value error, and <paramref name="entropyCoef"/> adds an entropy bonus.
    ///
    /// <paramref name="dScalarMu"/> and <paramref name="dScalarLogSd"/> carry the equivalent
    /// coefficients for the scalar heads, as produced by the Gaussian log-prob. They may be null when the
    /// network has no scalar heads.
    ///
    /// Must be called after <see cref="Forward"/> with the same <paramref name="x"/>.
    /// </summary>
    public void Backprop(float[] x, int action, float dLogProb, float dValue, float entropyCoef,
                         float[] dScalarMu = null, float[] dScalarLogSd = null)
    {
        // Scalar heads first, so their weight gradients land before the trunk pass below.
        BackpropScalars(dScalarMu, dScalarLogSd);

        // Categorical policy gradient: dLogits[a] = dLogProb * (onehot(a) - softmax).
        var dlogits = new float[ActionCount];
        for (int a = 0; a < ActionCount; a++) dlogits[a] = dLogProb * ((a == action ? 1f : 0f) - probs[a]);

        if (entropyCoef != 0f)
        {
            float H = 0f;
            for (int a = 0; a < ActionCount; a++) H -= probs[a] * Mathf.Log(Mathf.Max(probs[a], 1e-8f));
            for (int a = 0; a < ActionCount; a++)
                dlogits[a] += entropyCoef * probs[a] * (Mathf.Log(Mathf.Max(probs[a], 1e-8f)) + H);
        }

        for (int a = 0; a < ActionCount; a++)
        {
            float d = dlogits[a];
            if (d == 0f) continue;
            gbp[a] += d;
            int off = a * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) gwp[off + j] += d * h2[j];
        }

        // Policy heads into the POLICY trunk. The value term is deliberately absent: the critic has its
        // own trunk below, and letting the value gradient into the policy trunk is what made the two
        // objectives fight.
        var dh2 = new float[HiddenSize];
        for (int j = 0; j < HiddenSize; j++)
        {
            float g = 0f;
            for (int a = 0; a < ActionCount; a++) g += dlogits[a] * wp[a * HiddenSize + j];
            dh2[j] = g;
        }
        ScalarTrunkGradient(dScalarMu, dh2);

        // Critic: a completely separate backward pass over its own trunk, seeded only by dValue.
        // Nothing here touches dh2, gw1, gw2 or the policy hidden layers.
        if (dValue != 0f)
        {
            for (int j = 0; j < ValueHiddenSize; j++) gwv[j] += dValue * vh2[j];
            gbv[0] += dValue;

            var dvh2 = new float[ValueHiddenSize];
            for (int j = 0; j < ValueHiddenSize; j++) dvh2[j] = dValue * wv[j];

            var dvh2pre = new float[ValueHiddenSize];
            for (int i = 0; i < ValueHiddenSize; i++)
            {
                float g = dvh2[i] * (vh2[i] > 0f ? 1f : 0f);
                dvh2pre[i] = g;
                if (g == 0f) continue;
                gvb2[i] += g;
                int off2 = i * ValueHiddenSize;
                for (int j = 0; j < ValueHiddenSize; j++) gvw2[off2 + j] += g * vh1[j];
            }

            for (int j = 0; j < ValueHiddenSize; j++)
            {
                float g = 0f;
                for (int i = 0; i < ValueHiddenSize; i++) g += dvh2pre[i] * vw2[i * ValueHiddenSize + j];
                g = g * (vh1[j] > 0f ? 1f : 0f);
                if (g == 0f) continue;
                gvb1[j] += g;
                int off1 = j * InputSize;
                for (int i = 0; i < InputSize; i++) gvw1[off1 + i] += g * x[i];
            }
        }

        var dh2pre = new float[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
        {
            float dh2i = dh2[i] * (h2[i] > 0f ? 1f : 0f);
            dh2pre[i] = dh2i;
            if (dh2i == 0f) continue;
            gb2[i] += dh2i;
            int off2 = i * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) gw2[off2 + j] += dh2i * h1[j];
        }

        var dh1pre = new float[HiddenSize];
        for (int j = 0; j < HiddenSize; j++)
        {
            float g = 0f;
            for (int i = 0; i < HiddenSize; i++) g += dh2pre[i] * w2[i * HiddenSize + j];
            dh1pre[j] = g * (h1[j] > 0f ? 1f : 0f);
        }
        for (int i = 0; i < HiddenSize; i++)
        {
            float g = dh1pre[i];
            if (g == 0f) continue;
            gb1[i] += g;
            int off1 = i * InputSize;
            for (int j = 0; j < InputSize; j++) gw1[off1 + j] += g * x[j];
        }
    }

    /// <summary>
    /// Apply one gradient step, with two safety nets that this trainer had been missing.
    ///
    /// NaN is unrecoverable here: once any weight becomes NaN, Forward returns NaN logits, the
    /// sampled distribution is all-NaN, every subsequent gradient is NaN, and the policy is dead for
    /// the rest of the run while still LOOKING alive (it keeps sampling, it just always picks the
    /// same action). That is exactly the failure observed in training - entropy 0, top-1 share 1.0,
    /// and advantage magnitude NaN - and it is unrecoverable without reloading weights from disk.
    ///
    /// The two nets:
    ///   * Global-norm gradient clipping, so one pathological minibatch cannot produce a step large
    ///     enough to overflow a float or blow the network apart.
    ///   * A non-finite check on the incoming gradients, which skips the step entirely. Skipping is
    ///     strictly better than applying: a NaN step poisons the weights permanently, whereas a
    ///     skipped step just loses one update.
    /// </summary>
    public void ApplyGradients(float lr, float maxGradNorm = 0f, int samples = 1, bool useAdam = false)
    {
        if (!GradientsAreFinite())
        {
            NonFiniteGradientSteps++;
            // Clear the accumulator so the bad batch cannot contaminate the next one either.
            ClearGradients();
            return;
        }

        // Normalise the accumulated SUM to a MEAN before clipping, so the norm clip applies to the mean
        // gradient and the step size is independent of minibatch size.
        //
        // This is the bug that kept the policy frozen. The caller passed lr/count while the gradient
        // arrays hold the SUM over the minibatch, so the clip (norm <= 0.5 on the sum) was divided by
        // the count a second time. Largest possible per-step change was 0.5 * lr / count ~= 4.7e-6 at
        // lr=3e-4, count=32. Over a whole run the categorical head moved so little that KL sat at ~1e-6,
        // clip fraction at 0, entropy stayed at the uniform ln(16)~2.77 and max|weight| did not change
        // to four decimals. The critic, whose error is far larger, still moved enough to train - which
        // is why value loss fell while the policy learned nothing.
        if (samples > 1)
        {
            float inv = 1f / samples;
            for (int i = 0; i < gw1.Length; i++) gw1[i] *= inv;
            for (int i = 0; i < gb1.Length; i++) gb1[i] *= inv;
            for (int i = 0; i < gw2.Length; i++) gw2[i] *= inv;
            for (int i = 0; i < gb2.Length; i++) gb2[i] *= inv;
            for (int i = 0; i < gwp.Length; i++) gwp[i] *= inv;
            for (int i = 0; i < gbp.Length; i++) gbp[i] *= inv;
            for (int i = 0; i < gwv.Length; i++) gwv[i] *= inv;
            gbv[0] *= inv;
            for (int i = 0; i < gvw1.Length; i++) gvw1[i] *= inv;
            for (int i = 0; i < gvb1.Length; i++) gvb1[i] *= inv;
            for (int i = 0; i < gvw2.Length; i++) gvw2[i] *= inv;
            for (int i = 0; i < gvb2.Length; i++) gvb2[i] *= inv;
            for (int i = 0; i < (gwmu?.Length ?? 0); i++) gwmu[i] *= inv;
            for (int i = 0; i < (gbmu?.Length ?? 0); i++) gbmu[i] *= inv;
            for (int i = 0; i < (glogSd?.Length ?? 0); i++) glogSd[i] *= inv;
        }

        if (maxGradNorm > 0f)
        {
            // Policy and critic are clipped INDEPENDENTLY.
            //
            // A single global norm over both objectives silently starves whichever has the smaller
            // gradient. The policy gradient - which also carries the entropy bonus - is much larger than
            // the value gradient, so the shared scale factor was tiny and the critic moved by a
            // vanishing fraction of learningRate every step. Measured consequence over one run: value
            // loss grew 3.8 -> 3376 while ~95% of steps clipped, explained variance degraded past -4,
            // and the advantage scale drifted to |A| ~ 50. The critic now gets a budget of its own, so
            // its step size is governed by its own error rather than by the policy's.
            float policyNorm = 0f;
            for (int i = 0; i < gw1.Length; i++) policyNorm += gw1[i] * gw1[i];
            for (int i = 0; i < gw2.Length; i++) policyNorm += gw2[i] * gw2[i];
            for (int i = 0; i < gwp.Length; i++) policyNorm += gwp[i] * gwp[i];
            // The scalar heads count toward the POLICY norm. Excluding them would let them take
            // arbitrarily large steps while the rest of the network stayed clipped.
            for (int i = 0; i < (gwmu?.Length ?? 0); i++) policyNorm += gwmu[i] * gwmu[i];
            for (int i = 0; i < (glogSd?.Length ?? 0); i++) policyNorm += glogSd[i] * glogSd[i];

            float criticNorm = 0f;
            for (int i = 0; i < gwv.Length; i++) criticNorm += gwv[i] * gwv[i];
            for (int i = 0; i < gvw1.Length; i++) criticNorm += gvw1[i] * gvw1[i];
            for (int i = 0; i < gvw2.Length; i++) criticNorm += gvw2[i] * gvw2[i];

            policyNorm = Mathf.Sqrt(policyNorm);
            criticNorm = Mathf.Sqrt(criticNorm);

            if (policyNorm > maxGradNorm)
            {
                float scale = maxGradNorm / (policyNorm + 1e-6f);
                for (int i = 0; i < gw1.Length; i++) gw1[i] *= scale;
                for (int i = 0; i < gb1.Length; i++) gb1[i] *= scale;
                for (int i = 0; i < gw2.Length; i++) gw2[i] *= scale;
                for (int i = 0; i < gb2.Length; i++) gb2[i] *= scale;
                for (int i = 0; i < gwp.Length; i++) gwp[i] *= scale;
                for (int i = 0; i < gbp.Length; i++) gbp[i] *= scale;
                for (int i = 0; i < (gwmu?.Length ?? 0); i++) gwmu[i] *= scale;
                for (int i = 0; i < (gbmu?.Length ?? 0); i++) gbmu[i] *= scale;
                for (int i = 0; i < (glogSd?.Length ?? 0); i++) glogSd[i] *= scale;
                ClippedGradSteps++;
            }

            if (criticNorm > maxGradNorm)
            {
                float scale = maxGradNorm / (criticNorm + 1e-6f);
                for (int i = 0; i < gwv.Length; i++) gwv[i] *= scale;
                gbv[0] *= scale;
                for (int i = 0; i < gvw1.Length; i++) gvw1[i] *= scale;
                for (int i = 0; i < gvb1.Length; i++) gvb1[i] *= scale;
                for (int i = 0; i < gvw2.Length; i++) gvw2[i] *= scale;
                for (int i = 0; i < gvb2.Length; i++) gvb2[i] *= scale;
                ClippedCriticGradSteps++;
            }
        }

        // Adam path. Adam rescales each parameter by its own gradient magnitude, which is the point: the
        // critic's error is orders of magnitude larger than the policy gradient, and a single shared
        // learning rate cannot serve both. Stable-Baselines3 and ML-Agents both default to Adam for PPO.
        if (useAdam)
        {
            AdamStep(lr, 0.9f, 0.999f, 1e-8f);
            RecomputeMaxWeight();
            return;
        }

        for (int i = 0; i < w1.Length; i++) w1[i] -= lr * gw1[i];
        for (int i = 0; i < b1.Length; i++) b1[i] -= lr * gb1[i];
        for (int i = 0; i < w2.Length; i++) w2[i] -= lr * gw2[i];
        for (int i = 0; i < b2.Length; i++) b2[i] -= lr * gb2[i];
        for (int i = 0; i < wp.Length; i++) wp[i] -= lr * gwp[i];
        for (int i = 0; i < bp.Length; i++) bp[i] -= lr * gbp[i];
        for (int i = 0; i < wv.Length; i++) wv[i] -= lr * gwv[i];
        bv[0] -= lr * gbv[0];
        for (int i = 0; i < vw1.Length; i++) vw1[i] -= lr * gvw1[i];
        for (int i = 0; i < vb1.Length; i++) vb1[i] -= lr * gvb1[i];
        for (int i = 0; i < vw2.Length; i++) vw2[i] -= lr * gvw2[i];
        for (int i = 0; i < vb2.Length; i++) vb2[i] -= lr * gvb2[i];
        for (int i = 0; i < (wmu?.Length ?? 0); i++) wmu[i] -= lr * gwmu[i];
        for (int i = 0; i < (bmu?.Length ?? 0); i++) bmu[i] -= lr * gbmu[i];
        for (int i = 0; i < (logSd?.Length ?? 0); i++)
        {
            logSd[i] -= lr * glogSd[i];
            // Bounded here as well as in ContinuousHead, because a gradient step can carry log-sd past
            // the bound and an out-of-range sd either spikes the density (ratio explodes) or goes flat
            // (nothing learnable).
            logSd[i] = Mathf.Clamp(logSd[i], ContinuousHead.MinLogSd, ContinuousHead.MaxLogSd);
        }
        RecomputeMaxWeight();
    }

    /// <summary>True when every accumulated gradient is a finite number.</summary>
    private bool GradientsAreFinite()
    {
        for (int i = 0; i < gw1.Length; i++) if (!IsFinite(gw1[i])) return false;
        for (int i = 0; i < gb1.Length; i++) if (!IsFinite(gb1[i])) return false;
        for (int i = 0; i < gw2.Length; i++) if (!IsFinite(gw2[i])) return false;
        for (int i = 0; i < gb2.Length; i++) if (!IsFinite(gb2[i])) return false;
        for (int i = 0; i < gwp.Length; i++) if (!IsFinite(gwp[i])) return false;
        for (int i = 0; i < gbp.Length; i++) if (!IsFinite(gbp[i])) return false;
        for (int i = 0; i < gwv.Length; i++) if (!IsFinite(gwv[i])) return false;
        for (int i = 0; i < gvw1.Length; i++) if (!IsFinite(gvw1[i])) return false;
        for (int i = 0; i < gvw2.Length; i++) if (!IsFinite(gvw2[i])) return false;
        for (int i = 0; i < gvb1.Length; i++) if (!IsFinite(gvb1[i])) return false;
        for (int i = 0; i < gvb2.Length; i++) if (!IsFinite(gvb2[i])) return false;
        for (int i = 0; i < (gwmu?.Length ?? 0); i++) if (!IsFinite(gwmu[i])) return false;
        for (int i = 0; i < (glogSd?.Length ?? 0); i++) if (!IsFinite(glogSd[i])) return false;
        return IsFinite(gbv[0]);
    }

    private static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

    /// <summary>
    /// True when every weight is finite.
    ///
    /// A poisoned network is otherwise indistinguishable from a working one at runtime, so this is
    /// checked by the learner and reported loudly. Recovery is to reload the last saved weights.
    /// </summary>
    public bool WeightsAreFinite()
    {
        for (int i = 0; i < w1.Length; i++) if (!IsFinite(w1[i])) return false;
        for (int i = 0; i < w2.Length; i++) if (!IsFinite(w2[i])) return false;
        for (int i = 0; i < wp.Length; i++) if (!IsFinite(wp[i])) return false;
        for (int i = 0; i < wv.Length; i++) if (!IsFinite(wv[i])) return false;
        for (int i = 0; i < vw1.Length; i++) if (!IsFinite(vw1[i])) return false;
        for (int i = 0; i < vw2.Length; i++) if (!IsFinite(vw2[i])) return false;
        for (int i = 0; i < (wmu?.Length ?? 0); i++) if (!IsFinite(wmu[i])) return false;
        for (int i = 0; i < (logSd?.Length ?? 0); i++) if (!IsFinite(logSd[i])) return false;
        return IsFinite(bv[0]) && IsFinite(bp[0]);
    }

    /// <summary>How many gradient steps were skipped because the gradients were not finite.</summary>
    public int NonFiniteGradientSteps { get; private set; }

    /// <summary>
    /// Largest magnitude seen in any weight or bias at the last update.
    ///
    /// A leading indicator of the NaN failure rather than a report of it. Once weights reach the
    /// hundreds the logits are enormous, the softmax saturates to one-hot, and a single update can
    /// overflow - which is exactly the sequence that killed the first training run. Watching this climb
    /// towards ~100 warns that a run is about to die while it can still be fixed.
    /// </summary>
    public float MaxAbsWeight { get; private set; }

    /// <summary>One Adam step over every parameter tensor.</summary>
    /// <param name="beta1">Exponential decay for the first moment. 0.9 is the standard.</param>
    /// <param name="beta2">Exponential decay for the second moment. 0.999 is the standard.</param>
    /// <param name="eps">Added inside sqrt(v) to keep the very first step finite, when v is still zero.</param>
    ///
    /// Bias correction is what makes the first step come out at roughly lr*g instead of zero. Without it
    /// the moments start at 0, so both estimates start at 0, and the update is silently a fraction of
    /// the learning rate for the first few dozen steps - the same class of "the policy is not moving"
    /// bug this project has already hit twice. Correction uses the GLOBAL step count across all
    /// tensors, which is what the original paper specifies.
    ///
    /// logSd is deliberately NOT treated like a weight: it is a scale parameter that is clamped to a
    /// legal range, and letting Adam's moments accumulate across a clamp boundary would push it back
    /// out every step. It is stepped with the same rule but re-clamped immediately after.
    private void AdamStep(float lr, float beta1, float beta2, float eps)
    {
        adamStep++;
        float bc1 = 1f - Mathf.Pow(beta1, adamStep);
        float bc2 = 1f - Mathf.Pow(beta2, adamStep);
        float invBc1 = bc1 > 0f ? 1f / bc1 : 1f;
        float invBc2 = bc2 > 0f ? 1f / bc2 : 1f;

        w1 = AdamUpdate(w1, gw1, mw1, vw1a, beta1, beta2, eps, lr, invBc1, invBc2);
        b1 = AdamUpdate(b1, gb1, mb1, vb1a, beta1, beta2, eps, lr, invBc1, invBc2);
        w2 = AdamUpdate(w2, gw2, mw2, vw2a, beta1, beta2, eps, lr, invBc1, invBc2);
        b2 = AdamUpdate(b2, gb2, mb2, vb2a, beta1, beta2, eps, lr, invBc1, invBc2);
        wp = AdamUpdate(wp, gwp, mwp, vwpa, beta1, beta2, eps, lr, invBc1, invBc2);
        bp = AdamUpdate(bp, gbp, mbp, vbpa, beta1, beta2, eps, lr, invBc1, invBc2);
        wv = AdamUpdate(wv, gwv, mwv, vwva, beta1, beta2, eps, lr, invBc1, invBc2);
        bv = AdamUpdate(bv, gbv, mbv, vbva, beta1, beta2, eps, lr, invBc1, invBc2);
        vw1 = AdamUpdate(vw1, gvw1, mvw1, vvw1a, beta1, beta2, eps, lr, invBc1, invBc2);
        vb1 = AdamUpdate(vb1, gvb1, mvb1, vvb1a, beta1, beta2, eps, lr, invBc1, invBc2);
        vw2 = AdamUpdate(vw2, gvw2, mvw2, vvw2a, beta1, beta2, eps, lr, invBc1, invBc2);
        vb2 = AdamUpdate(vb2, gvb2, mvb2, vvb2a, beta1, beta2, eps, lr, invBc1, invBc2);
        if (wmu != null)
        {
            wmu = AdamUpdate(wmu, gwmu, mwmu, vwmu, beta1, beta2, eps, lr, invBc1, invBc2);
            bmu = AdamUpdate(bmu, gbmu, mbmu, vbmu, beta1, beta2, eps, lr, invBc1, invBc2);
            logSd = AdamUpdate(logSd, glogSd, mlogSd, vlogSd, beta1, beta2, eps, lr, invBc1, invBc2);
            for (int i = 0; i < logSd.Length; i++)
                logSd[i] = Mathf.Clamp(logSd[i], ContinuousHead.MinLogSd, ContinuousHead.MaxLogSd);
        }
    }

    /// <summary>Adam on a single tensor. Returns the same array it was given (mutated in place), so the
    /// call sites read as assignments without allocating per step.</summary>
    /// <param name="invBc1">1/(1 - beta1^t), the first-moment bias correction.</param>
    /// <param name="invBc2">1/(1 - beta2^t), the second-moment bias correction.</param>
    ///
    /// This is the textbook update written out in full,
    /// <c>p -= lr * (m*invBc1) / (sqrt(v*invBc2) + eps)</c>, rather than the algebraically equivalent
    /// "absorb sqrt(1-beta2^t) into the step size" form that most reference code uses.
    ///
    /// The absorbed form is shorter and is why this is worth being explicit about: it is very easy to
    /// absorb that sqrt into the numerator as well as the denominator, which leaves an extra
    /// <c>sqrt(invBc2)</c> of ~30x on every step - a learning rate that is wrong by a factor of thirty,
    /// with no error, no NaN and a plausible-looking loss curve. That is exactly what happened the first
    /// time this was written, and it was caught only because the update is checked against an
    /// independent reference implementation rather than merely being inspected.
    private static float[] AdamUpdate(float[] p, float[] g, float[] m, float[] v,
                                      float beta1, float beta2, float eps, float lr,
                                      float invBc1, float invBc2)
    {
        if (p == null || g == null) return p;
        for (int i = 0; i < p.Length; i++)
        {
            float gi = g[i];
            m[i] = beta1 * m[i] + (1f - beta1) * gi;
            v[i] = beta2 * v[i] + (1f - beta2) * gi * gi;
            float mHat = m[i] * invBc1;
            float vHat = v[i] * invBc2;
            p[i] -= lr * mHat / (Mathf.Sqrt(vHat) + eps);
        }
        return p;
    }

    /// <summary>Recompute <see cref="MaxAbsWeight"/>. Cheap; called once per update, and by the weight
    /// loader so it can reject a file that parses but carries no trained weights at all.</summary>
    public void RecomputeMaxWeight()
    {
        float m = 0f;
        for (int i = 0; i < w1.Length; i++) { float a = Mathf.Abs(w1[i]); if (a > m) m = a; }
        for (int i = 0; i < b1.Length; i++) { float a = Mathf.Abs(b1[i]); if (a > m) m = a; }
        for (int i = 0; i < w2.Length; i++) { float a = Mathf.Abs(w2[i]); if (a > m) m = a; }
        for (int i = 0; i < b2.Length; i++) { float a = Mathf.Abs(b2[i]); if (a > m) m = a; }
        for (int i = 0; i < wp.Length; i++) { float a = Mathf.Abs(wp[i]); if (a > m) m = a; }
        for (int i = 0; i < bp.Length; i++) { float a = Mathf.Abs(bp[i]); if (a > m) m = a; }
        for (int i = 0; i < wv.Length; i++) { float a = Mathf.Abs(wv[i]); if (a > m) m = a; }
        for (int i = 0; i < vw1.Length; i++) { float a = Mathf.Abs(vw1[i]); if (a > m) m = a; }
        for (int i = 0; i < vw2.Length; i++) { float a = Mathf.Abs(vw2[i]); if (a > m) m = a; }
        for (int i = 0; i < (wmu?.Length ?? 0); i++) { float a = Mathf.Abs(wmu[i]); if (a > m) m = a; }
        m = Mathf.Max(m, Mathf.Abs(bv[0]));
        MaxAbsWeight = m;
    }

    /// <summary>How many POLICY gradient steps were rescaled by the norm clip.</summary>
    public int ClippedGradSteps { get; private set; }

    /// <summary>How many CRITIC gradient steps were rescaled. Tracked separately because a policy that
    /// clips constantly while the critic rarely does is the signature of a starved critic.</summary>
    public int ClippedCriticGradSteps { get; private set; }

    public Weights Export()
    {
        return new Weights
        {
            input = InputSize, hidden = HiddenSize, actions = ActionCount, valueHidden = ValueHiddenSize,
            w1 = (float[])w1.Clone(), b1 = (float[])b1.Clone(),
            w2 = (float[])w2.Clone(), b2 = (float[])b2.Clone(),
            wp = (float[])wp.Clone(), bp = (float[])bp.Clone(),
            wv = (float[])wv.Clone(), bv = (float[])bv.Clone(),
            wmu = wmu != null ? (float[])wmu.Clone() : null,
            bmu = bmu != null ? (float[])bmu.Clone() : null,
            logSd = logSd != null ? (float[])logSd.Clone() : null,
            vw1 = (float[])vw1.Clone(), vb1 = (float[])vb1.Clone(),
            vw2 = (float[])vw2.Clone(), vb2 = (float[])vb2.Clone(),
        };
    }

    /// <summary>
    /// Import weights, refusing anything whose shape does not match.
    ///
    /// Validated rather than trusted because the shapes have changed over time (the scalar heads were
    /// added later), and a partially-applied import would leave a network with, say, new policy weights
    /// and stale log-sd values - which produces a run that trains and never improves, with nothing in the
    /// log to say why. Returning false lets the caller fall back to a fresh network and say so.
    /// </summary>
    public bool Import(Weights w)
    {
        if (w == null) return false;
        if (w.input != InputSize || w.hidden != HiddenSize || w.actions != ActionCount) return false;
        if (w.w1 == null || w.b1 == null || w.w2 == null || w.b2 == null
            || w.wp == null || w.bp == null || w.wv == null || w.bv == null) return false;
        if (w.w1.Length != w1.Length || w.w2.Length != w2.Length || w.wp.Length != wp.Length
            || w.wv.Length != wv.Length) return false;

        // A scalar-head network cannot load a file that has no scalar arrays: it would run with
        // log-sd and means at their initial values while claiming the policy's weights were restored.
        if (ScalarCount > 0)
        {
            if (w.wmu == null || w.bmu == null || w.logSd == null) return false;
            if (w.wmu.Length != wmu.Length || w.logSd.Length != logSd.Length) return false;
        }
        // The critic trunk is not optional: a file without it would load policy weights fine and leave
        // the critic at its random initialisation while claiming a successful resume.
        if (w.vw1 == null || w.vb1 == null || w.vw2 == null || w.vb2 == null) return false;
        if (w.vw1.Length != vw1.Length || w.vw2.Length != vw2.Length) return false;
        // A file whose recorded critic width disagrees with ours is refused even when the array lengths
        // happen to match. The length check alone cannot see a future width change that preserves the
        // total parameter count, and loading it would silently pair weights with the wrong fan-in.
        if (w.valueHidden != 0 && w.valueHidden != ValueHiddenSize) return false;

        Array.Copy(w.w1, w1, w1.Length); Array.Copy(w.b1, b1, b1.Length);
        Array.Copy(w.w2, w2, w2.Length); Array.Copy(w.b2, b2, b2.Length);
        Array.Copy(w.wp, wp, wp.Length); Array.Copy(w.bp, bp, bp.Length);
        Array.Copy(w.wv, wv, wv.Length); Array.Copy(w.bv, bv, 1);
        if (ScalarCount > 0)
        {
            Array.Copy(w.wmu, wmu, wmu.Length);
            Array.Copy(w.bmu, bmu, bmu.Length);
            Array.Copy(w.logSd, logSd, logSd.Length);
        }
        Array.Copy(w.vw1, vw1, vw1.Length); Array.Copy(w.vb1, vb1, vb1.Length);
        Array.Copy(w.vw2, vw2, vw2.Length); Array.Copy(w.vb2, vb2, vb2.Length);
        return true;
    }
}
