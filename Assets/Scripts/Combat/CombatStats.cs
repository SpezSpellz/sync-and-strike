using System;

/// <summary>
/// Per-fighter running combat counters, used to score a training match and to drive the training
/// telemetry CSV.
///
/// Exists because the two numbers the reward actually depends on were not being tracked anywhere.
/// The PPO learner only ever saw health DELTAS between turns, which is enough to compute a per-turn
/// reward but not enough to answer "did this fighter win the fight by itself or did its ally carry
/// it". Dividing the shared win bonus by damage contribution needs a running total, and a whiff rate
/// needs the number of attacks attempted as well as the number that landed.
///
/// These are monotonic within a match. <see cref="Reset"/> is called when the match resets so a
/// long training run does not accumulate totals across thousands of matches and drown out the
/// per-match numbers.
/// </summary>
[Serializable]
public class CombatStats
{
    /// <summary>Attacks thrown. Denominator of the whiff rate.</summary>
    public int attacksThrown;

    /// <summary>Attacks whose hitbox connected, blocked or not.</summary>
    public int hitsLanded;

    /// <summary>Health points dealt to opponents.</summary>
    public float damageDealt;

    /// <summary>Health points received from opponents.</summary>
    public float damageTaken;

    /// <summary>Largest single hit dealt. A useful early-warning signal for degenerate policies
    /// that learn to spam one big move instead of playing the fight.</summary>
    public float biggestHit;

    public float WhiffRate => attacksThrown > 0 ? 1f - (float)hitsLanded / attacksThrown : 0f;

    /// <summary>Reset for a new match. Called from the training runner, not from gameplay code, so
    /// the shipped game keeps whole-session totals.</summary>
    public void Reset()
    {
        attacksThrown = 0;
        hitsLanded = 0;
        damageDealt = 0f;
        damageTaken = 0f;
        biggestHit = 0f;
    }

    /// <summary>Called when a fighter commits to an attack move, whether or not it ever connects.
    /// Counting these is what makes a whiff rate possible at all.</summary>
    public void RecordAttack() => attacksThrown++;

    /// <summary>Called from HitReaction when a hitbox connects. <paramref name="damage"/> is the
    /// health actually removed, so a fully blocked hit correctly contributes no damage.</summary>
    public void RecordHit(float damage)
    {
        hitsLanded++;
        damageDealt += damage;
        if (damage > biggestHit) biggestHit = damage;
    }

    public void RecordDamageTaken(float damage) => damageTaken += damage;

    public override string ToString() =>
        $"{attacksThrown} attacks, {hitsLanded} hits ({WhiffRate:P0} whiff), " +
        $"{damageDealt:0} dealt, {damageTaken:0} taken";
}
