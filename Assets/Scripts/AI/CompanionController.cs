using UnityEngine;

/// <summary>
/// The player's ally. Plays the same moveset as the player, targets only the enemy,
/// never collides with or damages the player, and is voted on by the player after
/// each of its turns.
/// </summary>
public class CompanionController : AIController
{
    [Header("Companion")]
    [Tooltip("The character the companion fights. If empty it is resolved automatically.")]
    [SerializeField]
    private CharacterController enemyTarget;

    [Tooltip("The player. Used only for spacing; the companion passes through it.")]
    [SerializeField]
    private CharacterController playerAlly;

    [Tooltip("Seconds the vote prompt stays on screen before auto-resolving as 'good'.")]
    private float voteTimeout = 12f;

    protected override void ConfigurePersonality(FighterAI.Personality profile)
    {
        // The companion plays around the player rather than through them.
        profile.avoidAlly = true;
        profile.aggression = Mathf.Max(profile.aggression, 0.5f);
    }

    protected override CharacterController ResolveTarget()
    {
        if (enemyTarget != null && !enemyTarget.IsDead())
            return enemyTarget;
        return FindFirstEnemy();
    }

    protected override CharacterController ResolveAlly()
    {
        if (playerAlly != null)
            return playerAlly;
        var players = FindAllPlayers();
        foreach (var candidate in players)
        {
            if (candidate != this && candidate.Team == Team)
                return candidate;
        }
        return null;
    }

    private CharacterController FindFirstEnemy()
    {
        var all = FindAllPlayers();
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

    private CharacterController[] FindAllPlayers()
    {
        var all = FindObjectsByType<CharacterController>(FindObjectsSortMode.None);
        return all;
    }

    /// <summary>Public accessor so the vote manager can read the companion's current target.</summary>
    public CharacterController ResolveTargetForUI()
    {
        return ResolveTarget();
    }

    /// <summary>
    /// The companion plays on an on-device PPO policy (see <see cref="CompanionLearner"/>) that keeps
/// training on the player's own machine from their votes and match outcomes. The rule-based brain is
    /// still used during the warm start and to supply the jump/DI geometry each turn.
    /// </summary>
    protected override FighterPolicy SelectBrain()
    {
        return CompanionLearner.CreatePolicy(this, ruleBrain);
    }

    /// <summary>Public accessor for the on-device learner so it can build the next observation.</summary>
    public CharacterController ResolveTargetForTraining() => ResolveTarget();

    /// <summary>Public accessor for the on-device learner so it can build the next observation.</summary>
    public CharacterController ResolveAllyForTraining() => ResolveAlly();

    public override void RequestDecision()
    {
        base.RequestDecision();
        // Capture the committed transition and fold in any pending PPO update.
        CompanionLearner.NotifyDecision(this);
    }

    /// <summary>Called by TurnManager when the turn starts so the companion can vote-prompt later.</summary>
    public void PrepareVoting()
    {
        CompanionVoteManager.Instance?.BeginVoteFor(this, CurrentDecision, DecisionContext, voteTimeout);
    }
}