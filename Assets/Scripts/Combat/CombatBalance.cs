using UnityEngine;

/// <summary>
/// Central balance constants for the one-vs-two matchup, applied to the lone ENEMY fighter.
///
/// The enemy fights two bodies with one, so it is deliberately given more staying power and more
/// damage output than a single ally. Both multipliers are RUNTIME values rather than authored into
/// the scene, for one specific reason: the headless training arena builds its fighters from code and
/// would otherwise never see values that live on an authored prefab. Applying them here means the
/// enemy the policies train against is the enemy that ships - a train/serve mismatch on combat
/// numbers is invisible in every win rate and only shows up as a trained policy that plays badly.
///
/// Applied from CharacterController.Start for Team == Enemy, BEFORE the fighter registers its
/// start-of-match SaveData, so the scaled health is what every match reset restores.
///
/// NOTE ON REWARDS: the PPO reward weights damageDealt/damageTaken are PER HEALTH POINT, so raising
/// enemy damage makes the ally's damageTaken penalty larger and raising enemy health makes the
/// enemy's damageDealt reward smaller per point. If the eval win rates move in a way that does not
/// track perceived skill, re-check those weights (or normalise them by maxHealth) rather than this
/// file.
/// </summary>
public static class CombatBalance
{
    /// <summary>Damage dealt by the enemy, multiplied. 1 is parity with an ally.</summary>
    public static float enemyDamageMultiplier = 1.9f;

    /// <summary>Enemy starting/max health, multiplied. 1 is parity with an ally.</summary>
    public static float enemyHealthMultiplier = 1.9f;

    /// <summary>Effective combat power the two knobs produce together, for logging.</summary>
    public static float EnemyPowerMultiplier => enemyDamageMultiplier * enemyHealthMultiplier;

    /// <summary>
    /// Apply a <code>-enemyDamage=</code> / <code>-enemyHealth=</code> command-line override.
    /// Non-positive or non-finite values are rejected so a typo cannot silently zero the enemy's
    /// health (which would look like an instantly-winning policy).
    /// </summary>
    public static bool TrySet(string field, float value)
    {
        if (!(value > 0f) || float.IsNaN(value) || float.IsInfinity(value)) return false;
        switch (field)
        {
            case "enemyDamage": enemyDamageMultiplier = value; return true;
            case "enemyHealth": enemyHealthMultiplier = value; return true;
            default: return false;
        }
    }

    /// <summary>One line for the run banner, so the shipped balance is stated rather than assumed.</summary>
    public static string Describe() =>
        $"enemy damage x{enemyDamageMultiplier:0.###}, health x{enemyHealthMultiplier:0.###} "
        + $"(combined power x{EnemyPowerMultiplier:0.###})";
}
