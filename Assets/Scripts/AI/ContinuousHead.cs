using System;
using UnityEngine;

/// <summary>
/// One scalar output the policy chooses directly, as a squashed Gaussian.
///
/// A continuous action cannot be sampled from a categorical distribution, and a PPO importance ratio
/// needs a probability DENSITY to compare against. So each scalar gets its own univariate Gaussian,
/// and the move head's categorical log-prob is added to the sum of these to form the joint log-prob.
/// Dropping the Gaussian terms would leave the ratio describing only the move choice, and the scalar
/// heads would receive no gradient at all while appearing to train.
///
/// The squash exists because each scalar has a hard gameplay range (see <see cref="ActionScalars"/>)
/// and the game's own setters clamp out-of-range values. Squashing into that range directly means the
/// network's raw output is unbounded but the action it produces never needs clamping.
///
/// That squashing has a consequence which is easy to get wrong and which this class exists to make
/// impossible to omit: the log-prob of the ACTION is not the log-prob of the raw sample. Since
/// a = lo + range * sigmoid(z), the density of a is the density of z times the inverse Jacobian
/// 1/(da/dz). Omitting that term produces a log-prob that is wrong by a state-dependent amount, which
/// silently biases every continuous update - no error, no NaN, just a policy that learns worse than
/// the expert it was cloned from.
/// </summary>
[Serializable]
public class ContinuousHead
{
    /// <summary>Lower bound of the gameplay range this head maps onto.</summary>
    public float lo;
    public float hi;

    /// <summary>Raw (pre-squash) mean from the network's continuous output head.</summary>
    public float mu;

    /// <summary>Log standard deviation. State-independent, so this is one learned parameter, not a
    /// function of the observation.</summary>
    public float logSd;

    private float sd => Mathf.Exp(logSd);

    /// <summary>
    /// Initial log-sd. log(0.6) ~= -0.51, which puts the first samples a moderate distance from the
    /// mean without being wide enough to saturate the range on the first turn.
    /// </summary>
    public const float DefaultLogSd = -0.51f;

    /// <summary>
    /// Bounds on log-sd.
    ///
    /// The lower bound matters most: sd approaching zero makes the log-density spike, so the
    /// importance ratio explodes and a single sample can dominate a PPO update. The upper bound stops
    /// the distribution going flat, where exploration is pure noise and nothing can ever be learned.
    /// </summary>
    public const float MinLogSd = -2.5f;
    public const float MaxLogSd = 0.7f;

    public ContinuousHead(float lo, float hi)
    {
        this.lo = lo;
        this.hi = hi;
        logSd = DefaultLogSd;
    }

    public void ClampLogSd() => logSd = Mathf.Clamp(logSd, MinLogSd, MaxLogSd);

    /// <summary>Sample z ~ N(mu, sd). Returns the RAW sample, which is what must be stored for
    /// recomputing the log-prob later.</summary>
    public float SampleZ(System.Random rng)
    {
        // Box-Muller. Both uniforms are guarded against exactly 0, since log(0) is -Infinity and would
        // put a NaN into the sample - which then poisons the stored log-prob for the whole buffer.
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return mu + (float)(sd * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>Map a raw sample into the gameplay range.</summary>
    public float Squash(float z) => lo + (hi - lo) * Sigmoid(z);

    /// <summary>
    /// Inverse of <see cref="Squash"/>. Needed to express a TARGET action (from the expert, or from a
    /// saved transition) as the raw z that log-prob is evaluated at.
    /// </summary>
    public float Unsquash(float a)
    {
        float t = Mathf.Clamp01((a - lo) / (hi - lo));
        // Clamped away from 0 and 1 so the log never sees log(0) or log(1-in-epsilon).
        t = Mathf.Clamp(t, 1e-4f, 1f - 1e-4f);
        return Mathf.Log(t / (1f - t));
    }

    private static float Sigmoid(float z)
    {
        // Branched to avoid overflow: Mathf.Exp of a large negative is fine, but of a large positive
        // returns Infinity and then Infinity/Infinity is NaN.
        if (z >= 0f) return 1f / (1f + Mathf.Exp(-z));
        float e = Mathf.Exp(z);
        return e / (1f + e);
    }

    /// <summary>
    /// Log-density of a raw sample under N(mu, sd), plus the log-Jacobian correction.
    ///
    /// This is the term that must appear in the policy's log-prob. Callers should use
    /// <see cref="LogProbForAction"/> rather than calling this directly, so the Jacobian cannot be
    /// left off by accident at a call site.
    /// </summary>
    public float LogProbRaw(float z)
    {
        float s = sd;
        float d = (z - mu) / s;
        // log N(z) - log(da/dz). da/dz = (hi-lo) * s(1-s), whose log is
        // log(hi-lo) + log(s) + log(1-s) with s the SQUASH output, not the standard deviation.
        float sq = Sigmoid(z);
        float logJacobian = Mathf.Log(Mathf.Max(hi - lo, 1e-6f))
                          + Mathf.Log(Mathf.Max(sq * (1f - sq), 1e-6f));
        return -0.5f * d * d - Mathf.Log(s) - 0.9189385f - logJacobian;
    }

    /// <summary>Log-prob of a gameplay-range ACTION. Converts to z internally.</summary>
    public float LogProbForAction(float a) => LogProbRaw(Unsquash(a));

    /// <summary>
    /// d(logProb)/d(mu) for a raw sample. For a Gaussian this is (z - mu) / sd^2.
    ///
    /// Note this is the gradient with respect to the RAW sample's log-density, which includes the
    /// Jacobian term. The Jacobian depends on z, not on mu, so it contributes nothing to d/dmu - but it
    /// DOES contribute to d/dz, which matters when the same head is also being regression-cloned
    /// toward a target.
    /// </summary>
    public float DLogProbDMu(float z)
    {
        float s = sd;
        return (z - mu) / (s * s);
    }

    /// <summary>d(logProb)/d(logSd) for a raw sample.</summary>
    public float DLogProbDLogSd(float z)
    {
        float s = sd;
        // d/dlogSd of [-0.5*d^2 - log(s) - jacobian] where d = (z-mu)/s and the jacobian carries log(s').
        // d/ds of -0.5*((z-mu)/s)^2 = (z-mu)^2/s^3, and ds/dlogSd = s.
        float num = z - mu;
        return (num * num) / (s * s) - 1f;
    }
}

/// <summary>
/// The scalar outputs the policy chooses, and their gameplay ranges.
///
/// These ranges are the game's own, taken from CharacterController.setJumpInfo and setKnockbackInfo.
/// Keeping them here rather than reading them from the controller means the network's output space is
/// fixed and checkable, and it means a change to the game's clamp values shows up as a mismatch to
/// resolve rather than as a silent rescaling of what the policy learned.
/// </summary>
public static class ActionScalars
{
    public const int Count = 4;

    public const int JumpPower = 0;
    public const int JumpAngle = 1;
    public const int DiPower = 2;
    public const int DiAngle = 3;

    // setJumpInfo: power clamped to 0.5..1, direction clamped to 0.5236..2.618 rad (30..150 degrees,
    // measured from straight up).
    public const float JumpPowerLo = 0.5f;
    public const float JumpPowerHi = 1f;
    public const float JumpAngleLo = 0.5235987755982988f;
    public const float JumpAngleHi = 2.6179938779914944f;

    // setKnockbackInfo: power clamped to 0..1, direction clamped to 0..2pi.
    public const float DiPowerLo = 0f;
    public const float DiPowerHi = 1f;
    public const float DiAngleLo = 0f;
    public const float DiAngleHi = 6.283185307179586f;

    public static ContinuousHead[] CreateAll()
    {
        return new[]
        {
            new ContinuousHead(JumpPowerLo, JumpPowerHi),
            new ContinuousHead(JumpAngleLo, JumpAngleHi),
            new ContinuousHead(DiPowerLo, DiPowerHi),
            new ContinuousHead(DiAngleLo, DiAngleHi),
        };
    }

    public static float Lo(int i) => All()[i].lo;
    public static float Hi(int i) => All()[i].hi;

    private static ContinuousHead[] cached;
    private static ContinuousHead[] All() => cached ??= CreateAll();
}
