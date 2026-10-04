using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The observation normaliser attached to a single role's brain.
///
/// Kept separate from <see cref="PolicyLearner"/> so the normaliser can be exercised, inspected and
/// persisted without touching the reward or PPO paths, and so a role can have normalisation enabled
/// while another does not.
///
/// Normalisation is OFF by default. It is not free: it introduces a train/inference skew, because the
/// statistics depend on how long the network has been training. A network that has just been created
/// and one that has trained for a hundred thousand turns normalise the same observation differently.
/// Enabling it is therefore a decision that has to be made deliberately, with the statistics
/// persisted alongside the weights.
/// </summary>
public class ObservationNormalizer
{
    private readonly Dictionary<string, ObservationStats> byRole = new Dictionary<string, ObservationStats>();

    /// <summary>Master switch. Overridable per run from the command line.</summary>
    public bool enabled;

    /// <summary>When true, statistics are persisted with the weights so a reload keeps them.</summary>
    public bool persist = true;

    public ObservationStats StatsFor(string role)
    {
        if (!byRole.TryGetValue(role, out var stats) || stats == null)
        {
            stats = new ObservationStats();
            byRole[role] = stats;
        }
        return stats;
    }

    /// <summary>
    /// Fold a raw observation into the running statistics for this role.
    ///
    /// Always fed the RAW capture, never an already-normalised vector: normalising before updating the
    /// statistics would make the statistics describe their own output and converge to a fixed point
    /// rather than describing the world.
    /// </summary>
    public void Observe(string role, float[] rawObs)
    {
        if (!enabled || rawObs == null) return;
        StatsFor(role).Update(rawObs);
    }

    /// <summary>
    /// Return the vector to actually feed the network: normalised if enabled and trustworthy, otherwise
    /// the raw vector untouched.
    /// </summary>
    public float[] Prepare(string role, float[] rawObs)
    {
        if (rawObs == null) return null;
        if (!enabled) return rawObs;

        var stats = StatsFor(role);
        if (!stats.ShouldNormalize) return rawObs;

        // Copy, because the caller also keeps the raw observation for the rollout buffer. Storing the
        // normalised version and re-normalising it on the next turn would compound the transform and
        // drift further from the true distribution every turn.
        var copy = (float[])rawObs.Clone();
        stats.Normalize(copy);
        return copy;
    }

    public void Clear(string role) => byRole.Remove(role);

    public void ClearAll() => byRole.Clear();
}
