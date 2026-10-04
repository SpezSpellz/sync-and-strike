using System;
using UnityEngine;

/// <summary>
/// A tiny MLP for the on-device PPO policy: input -> ReLU -> ReLU -> (categorical policy head,
/// scalar value head). Backprop is implemented by hand because we only need two gradients
/// (selected-action log-prob and value) and want zero external dependencies so it can run in the
/// shipped build without ML-Agents or a Python runtime.
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
    }

    public int InputSize { get; }
    public int HiddenSize { get; }
    public int ActionCount { get; }

    private float[] w1, b1, w2, b2, wp, bp, wv, bv;
    private float[] gw1, gb1, gw2, gb2, gwp, gbp, gwv, gbv;

    // Cached forward activations for the current Forward() call.
    private float[] h1, h2, logits;
    private float value;
    private float[] probs;

    public NeuralNetwork(int input, int hidden, int actions, int seed = 1234)
    {
        InputSize = input; HiddenSize = hidden; ActionCount = actions;
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
        logits = new float[actions]; probs = new float[actions];

        var rng = new System.Random(seed);
        float s1 = (float)Math.Sqrt(2.0 / input);
        float s2 = (float)Math.Sqrt(2.0 / hidden);
        float s3 = 0.01f; // near-zero policy head so the initial policy is ~uniform
        for (int i = 0; i < w1.Length; i++) w1[i] = (float)(rng.NextDouble() * 2 - 1) * s1;
        for (int i = 0; i < w2.Length; i++) w2[i] = (float)(rng.NextDouble() * 2 - 1) * s2;
        for (int i = 0; i < wp.Length; i++) wp[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
        for (int i = 0; i < wv.Length; i++) wv[i] = (float)(rng.NextDouble() * 2 - 1) * s3;
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

        float v = bv[0];
        for (int j = 0; j < HiddenSize; j++) v += wv[j] * h2[j];
        value = v;
        return value;
    }

    public float[] Probabilities => probs;
    public float[] Logits => logits;
    public float Value => value;

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
    }

    /// <summary>
    /// Backprop for one PPO sample. <paramref name="dLogProb"/> is the scalar coefficient on the
    /// gradient of log pi(chosen) (advantage * clipped ratio), <paramref name="dValue"/> the
    /// coefficient on the value error, and <paramref name="entropyCoef"/> adds an entropy bonus.
    /// Must be called after <see cref="Forward"/> with the same <paramref name="x"/>.
    /// </summary>
    public void Backprop(float[] x, int action, float dLogProb, float dValue, float entropyCoef)
    {
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

        // Value head, and both gradients accumulated back into the shared trunk.
        // dh2 MUST include the value term (dValue * wv[j]). Omitting it leaves the value head
        // detached from the trunk: the value loss trains wv/bv but the hidden layers receive
        // nothing from it, which silently breaks the critic.
        var dh2 = new float[HiddenSize];
        for (int j = 0; j < HiddenSize; j++)
        {
            float g = dValue * wv[j];
            for (int a = 0; a < ActionCount; a++) g += dlogits[a] * wp[a * HiddenSize + j];
            dh2[j] = g;
            gwv[j] += dValue * h2[j];
        }
        gbv[0] += dValue;

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
            float norm = 0f;
            for (int i = 0; i < gw1.Length; i++) norm += gw1[i] * gw1[i];
            for (int i = 0; i < gw2.Length; i++) norm += gw2[i] * gw2[i];
            for (int i = 0; i < gwp.Length; i++) norm += gwp[i] * gwp[i];
            for (int i = 0; i < gwv.Length; i++) norm += gwv[i] * gwv[i];
            norm = Mathf.Sqrt(norm);
            if (norm > maxGradNorm)
            {
                float scale = maxGradNorm / (norm + 1e-6f);
                for (int i = 0; i < gw1.Length; i++) gw1[i] *= scale;
                for (int i = 0; i < gb1.Length; i++) gb1[i] *= scale;
                for (int i = 0; i < gw2.Length; i++) gw2[i] *= scale;
                for (int i = 0; i < gb2.Length; i++) gb2[i] *= scale;
                for (int i = 0; i < gwp.Length; i++) gwp[i] *= scale;
                for (int i = 0; i < gbp.Length; i++) gbp[i] *= scale;
                for (int i = 0; i < gwv.Length; i++) gwv[i] *= scale;
                gbv[0] *= scale;
                ClippedGradSteps++;
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
        m = Mathf.Max(m, Mathf.Abs(bv[0]));
        MaxAbsWeight = m;
    }

    /// <summary>How many gradient steps were rescaled by the global-norm clip.</summary>
    public int ClippedGradSteps { get; private set; }

    public Weights Export()
    {
        return new Weights
        {
            input = InputSize, hidden = HiddenSize, actions = ActionCount,
            w1 = (float[])w1.Clone(), b1 = (float[])b1.Clone(),
            w2 = (float[])w2.Clone(), b2 = (float[])b2.Clone(),
            wp = (float[])wp.Clone(), bp = (float[])bp.Clone(),
            wv = (float[])wv.Clone(), bv = (float[])bv.Clone(),
        };
    }

    public void Import(Weights w)
    {
        if (w == null || w.input != InputSize || w.hidden != HiddenSize || w.actions != ActionCount) return;
        Array.Copy(w.w1, w1, w1.Length); Array.Copy(w.b1, b1, b1.Length);
        Array.Copy(w.w2, w2, w2.Length); Array.Copy(w.b2, b2, b2.Length);
        Array.Copy(w.wp, wp, wp.Length); Array.Copy(w.bp, bp, bp.Length);
        Array.Copy(w.wv, wv, wv.Length); Array.Copy(w.bv, bv, 1);
    }
}