using UnityEngine;

/// <summary>
/// The enemy fighter. Now driven by the same rule-based <see cref="FighterAI"/> brain as the
/// companion, so it picks its move automatically without needing a trained ML-Agents policy.
/// </summary>
public class EnemyController : AIController
{
    [Header("Enemy")]
    [Tooltip("Preferred target. If empty it picks the closest hostile fighter each turn.")]
    [SerializeField]
    private CharacterController target;

    protected override CharacterController ResolveTarget()
    {
        if (target != null && !target.IsDead())
            return target;
        return FindClosestEnemy();
    }

    protected override void ConfigurePersonality(FighterAI.Personality profile)
    {
        // The enemy is a touch more patient than the companion: it likes to block
        // and punish rather than mash.
        profile.aggression = Mathf.Clamp(profile.aggression, 0.35f, 1f);
        profile.blockChance = Mathf.Clamp(profile.blockChance, 0.25f, 1f);
    }

    private CharacterController FindClosestEnemy()
    {
        var all = FindObjectsByType<CharacterController>(FindObjectsSortMode.None);
        CharacterController best = null;
        float bestDistance = float.PositiveInfinity;
        Vector2 selfPos = GetPosition();
        foreach (var candidate in all)
        {
            if (candidate == this || candidate.IsDead()) continue;
            if (!CombatTeamUtility.AreEnemies(Team, candidate.Team)) continue;
            float d = Vector2.Distance(selfPos, candidate.GetPosition());
            if (d < bestDistance)
            {
                bestDistance = d;
                best = candidate;
            }
        }
        return best;
    }
}