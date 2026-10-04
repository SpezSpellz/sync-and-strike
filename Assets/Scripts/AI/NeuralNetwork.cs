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

    public NeuralNetwork(int input, int hidden, int actions, int seed = 1234, int scalars = 0)
    {
        InputSize = input; HiddenSize = hidden; ActionCount = actions; ScalarCount = scalars;
        int hh = hidden * hidden;
        w1 = new float[hidden * input]; b1 = new float[hidden];
        w2 = new float[hh]; b2 = new float[hidden];
        wp = new float[actions * hidden]; bp = new float[actions];
        wv = new float[hidden]; bv = new float[1];
        gw1 = new float[w1.Length]; gb1 = new float[hidden];
        gw2 = new float[w2.Length]; gb2 = new float[hidden];
        gwp = new float[wp.Length]; gbp = new float[actions];
        gwv = new float[hidden]; gbv = new float[1];
        h1 = new float[hidden]; h2 = new float[hidden];

        // Critic trunk: same shape as the policy trunk, independent weights.
        vw1 = new float[hidden * input]; vb1 = new float[hidden];
        vw2 = new float[hh]; vb2 = new float[hidden];
        gvw1 = new float[vw1.Length]; gvb1 = new float[hidden];
        gvw2 = new float[vw2.Length]; gvb2 = new float[hidden];
        vh1 = new float[hidden]; vh2 = new float[hidden];
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
        for (int i = 0; i < wv.Length; i++) wv[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
        // Critic trunk uses normal hidden-layer init, not the near-zero policy init: the value head has
        // to produce a spread of predictions immediately, or every advantage looks identical and the
        // first few thousand updates are wasted learning a constant.
        for (int i = 0; i < vw1.Length; i++) vw1[i] = (float)(rng.NextDouble() * 2 - 1) * s1;
        for (int i = 0; i < vw2.Length; i++) vw2[i] = (float)(rng.NextDouble() * 2 - 1) * s2;
        for (int i = 0; i < wv.Length; i++) wv[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
        // Scalar heads also start near zero, so the initial mu sits at 0 - the MIDPOINT of every
        // squashed range, because sigmoid(0) = 0.5. A mid-range jump or DI is a legal, harmless
        // default, which is what we want before warm start has cloned anything.
        for (int i = 0; i < (wmu?.Length ?? 0); i++) wmu[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
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
        for (int i = 0; i < HiddenSize; i++)
        {
            float s = vb1[i];
            int off = i * InputSize;
            for (int j = 0; j < InputSize; j++) s += vw1[off + j] * x[j];
            vh1[i] = s > 0f ? s : 0f;
        }
        for (int i = 0; i < HiddenSize; i++)
        {
            float s = vb2[i];
            int off = i * HiddenSize;
            for (int j = 0; j < HiddenSize; j++) s += vw2[off + j] * vh1[j];
            vh2[i] = s > 0f ? s : 0f;
        }
        float v = bv[0];
        for (int j = 0; j < HiddenSize; j++) v += wv[j] * vh2[j];
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
            for (int j = 0; j < HiddenSize; j++) gwv[j] += dValue * vh2[j];
            gbv[0] += dValue;

            var dvh2 = new float[HiddenSize];
            for (int j = 0; j < HiddenSize; j++) dvh2[j] = dValue * wv[j];

            var dvh2pre = new float[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                float g = dvh2[i] * (vh2[i] > 0f ? 1f : 0f);
                dvh2pre[i] = g;
                if (g == 0f) continue;
                gvb2[i] += g;
                int off2 = i * HiddenSize;
                for (int j = 0; j < HiddenSize; j++) gvw2[off2 + j] += g * vh1[j];
            }

            for (int j = 0; j < HiddenSize; j++)
            {
                float g = 0f;
                for (int i = 0; i < HiddenSize; i++) g += dvh2pre[i] * vw2[i * HiddenSize + j];
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
    public void ApplyGradients(float lr, float maxGradNorm = 0f)
    {
        if (!GradientsAreFinite())
        {
            NonFiniteGradientSteps++;
            // Clear the accumulator so the bad batch cannot contaminate the next one either.
            ClearGradients();
            return;
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
        TrackMaxWeight();
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

    /// <summary>Recompute <see cref="MaxAbsWeight"/>. Cheap; called once per update.</summary>
    private void TrackMaxWeight()
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
            input = InputSize, hidden = HiddenSize, actions = ActionCount,
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
