using UnityEngine;

/// <summary>
/// The enemy fighter. Ships FROZEN: outside a <c>-training</c> run it either uses the rule-based
/// <see cref="FighterAI"/> brain, or a policy trained offline and exported by
/// <see cref="PolicyLearner.ExportRole"/>. It never adapts while a player is playing.
///
/// To train it, run the game with the <c>-training</c> switch; the enemy then uses the same PPO
/// machinery as the companion and is allowed to learn for that run only.
/// </summary>
public class EnemyController : AIController
{
    [Header("Enemy")]
    [Tooltip("Preferred target. If empty it picks the closest hostile fighter each turn.")]
    [SerializeField]
    private CharacterController target;

    [Tooltip("Play with the offline-trained PPO policy instead of the rule-based brain. " +
             "Leave off until trained weights have been exported; without them the rule-based brain is used.")]
    [SerializeField]
    private bool useTrainedPolicy;

    protected override FighterPolicy SelectBrain()
    {
        // In a -training run the enemy always uses the trainable policy, which is how it gets
        // trained offline in the first place. Outside training it uses a trained policy when the build
        // actually ships one, or when the checkbox forces it; otherwise it stays on the rule-based brain.
        // Auto-detecting the shipped file means a build that exported its frozen enemy needs no manual
        // step, and a build that did not export one can never end up running an untrained network.
        if (!useTrainedPolicy && !TrainingMode.enabled
            && !PolicyLearner.HasShippedWeights(PolicyLearner.RoleEnemy)) return ruleBrain;
        // learnOnline: false -> outside a training run this role is frozen and never adapts during
        // a player's match. PolicyLearner unfreezes it only when TrainingMode.enabled.
        return PolicyLearner.CreatePolicy(this, ruleBrain, PolicyLearner.RoleEnemy, learnOnline: false);
    }

    public override void RequestDecision()
    {
        base.RequestDecision();
        PolicyLearner.NotifyDecision(this);
    }

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
        // Scoped to this fighter's own arena. A scene-wide search would happily pick an opponent
        // from a different arena, so during a multi-arena training run each fighter would be
        // playing against the wrong opponent and the results would be meaningless.
        var all = Arena.FightersNear(this);
        CharacterController best = null;
        float bestDistance = float.PositiveInfinity;
        Vector2 selfPos = GetPosition();
        foreach (var candidate in all)
        {
            if (candidate == null || candidate == this || candidate.IsDead()) continue;
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