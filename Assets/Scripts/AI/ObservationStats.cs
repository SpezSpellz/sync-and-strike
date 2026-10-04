using System;
using UnityEngine;

/// <summary>
/// Per-feature running mean and variance over the observation stream, for normalising inputs.
///
/// Capture-time scaling (see AIDecisionContext) puts every feature into roughly the same numeric
/// range, which is the bulk of the problem. This handles what scaling cannot: features that are
/// naturally rare or spiky still have a very different mean and spread from the rest, and a network
/// that spends most of its weights separating "target on the ground" from "target in the air" has less
/// capacity left for the things that actually decide a fight.
///
/// Deliberately a SEPARATE, optional stage rather than folding it into Capture. It introduces a
/// train/inference skew the moment it is enabled, because the statistics differ between a fresh
/// network and a trained one, so any consumer of the normalised vector has to use the same
/// statistics. Keeping it explicit means enabling it is a decision, not a silent change to the
/// observation contract.
///
/// Statistics are persisted alongside the policy weights. Without that, every restart would begin
/// normalising with statistics gathered from a different network's behaviour - the first few hundred
/// turns after a reload would be fed nonsense, which is exactly the warm-start window.
/// </summary>
[Serializable]
public class ObservationStats
{
    /// <summary>Number of observations folded in. Below a few hundred the variance is not trustworthy.</summary>
    public int count;

    public float[] mean;
    public float[] varSum;   // sum of squared deviations; converted to variance on read

    /// <summary>Variance floor, so a feature that has never varied does not divide by zero.</summary>
    private const float MinVariance = 1e-4f;

    public void EnsureSize(int size)
    {
        if (mean != null && mean.Length == size) return;
        mean = new float[size];
        varSum = new float[size];
        count = 0;
    }

    /// <summary>
    /// Fold one observation into the running statistics.
    ///
    /// Welford's online algorithm rather than accumulating a sum and a sum of squares: the naive form
    /// loses precision catastrophically over a long run because the sum of squares grows without
    /// bound while the variance of the data stays small, and float32 runs out of mantissa long before
    /// a training run finishes.
    /// </summary>
    public void Update(float[] x)
    {
        if (x == null) return;
        EnsureSize(x.Length);
        count++;
        float n = count;
        float invN = 1f / n;
        for (int i = 0; i < x.Length; i++)
        {
            float delta = x[i] - mean[i];
            mean[i] += delta * invN;
            varSum[i] += delta * (x[i] - mean[i]);
        }
    }

    /// <summary>
    /// Normalise in place using the current statistics.
    ///
    /// Clamped to +/-CLAMP_LIMIT because a feature that has barely varied produces enormous z-scores,
    /// and a single outlier of 50 would otherwise saturate every downstream ReLU and lose the gradient
    /// for the whole layer. Clamping keeps a rare extreme input influential but bounded.
    /// </summary>
    private const float CLAMP_LIMIT = 5f;

    public void Normalize(float[] x)
    {
        if (x == null || mean == null || mean.Length != x.Length) return;
        // Before enough samples, the variance estimate is noise. Passing the vector through untouched is
        // better than dividing by an unreliable std.
        if (count < MinSamplesForNormalizing) return;

        for (int i = 0; i < x.Length; i++)
        {
            float v = varSum[i] / Mathf.Max(count - 1, 1);
            float sd = Mathf.Sqrt(Mathf.Max(v, MinVariance));
            float z = (x[i] - mean[i]) / sd;
            x[i] = Mathf.Clamp(z, -CLAMP_LIMIT, CLAMP_LIMIT);
        }
    }

    /// <summary>
    /// Minimum observations before normalisation is trusted.
    ///
    /// Set past the warm-start window on purpose: the first turns are played by the rule-based expert,
    /// whose state distribution is not the policy's own, so folding those in would bias the statistics
    /// toward expert behaviour and leave the policy normalising against the wrong distribution.
    /// </summary>
    public const int MinSamplesForNormalizing = 500;

    public bool ShouldNormalize => count >= MinSamplesForNormalizing;

    /// <summary>Copy statistics out for persistence.</summary>
    public ObservationStats Clone()
    {
        return new ObservationStats
        {
            count = count,
            mean = (float[])mean?.Clone(),
            varSum = (float[])varSum?.Clone(),
        };
    }

    /// <summary>Restore persisted statistics, tolerating a feature-count change.</summary>
    public void CopyFrom(ObservationStats other)
    {
        if (other?.mean == null) return;
        count = other.count;
        mean = (float[])other.mean.Clone();
        varSum = (float[])other.varSum?.Clone();
    }
}
