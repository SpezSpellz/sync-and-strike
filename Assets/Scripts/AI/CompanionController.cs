using System.Collections.Generic;
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

    /// <summary>
    /// Point this fighter at a specific ally. The training arenas build their fighters from code, so
    /// the serialized <see cref="playerAlly"/> reference can never be authored for them and would
    /// otherwise stay null. Without an ally the FighterAI spacing rules are inert and the allyX/allyY
    /// observation features are always zero, so the companion could not learn to coordinate.
    /// </summary>
    public void SetAllyForTraining(CharacterController ally) => playerAlly = ally;

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

    /// <summary>
    /// Fighters in the same arena as this one.
    ///
    /// Scoped deliberately. This used to be a scene-wide FindObjectsByType, which is harmless with a
    /// single arena but catastrophic during a parallel training run: with N arenas alive every
    /// companion would happily target a fighter belonging to a different arena, so the matches
    /// would be meaningless. Mirrors EnemyController.FindClosestEnemy.
    /// </summary>
    private List<CharacterController> FindAllPlayers()
    {
        return Arena.FightersNear(this);
    }

    /// <summary>Public accessor so the vote manager can read the companion's current target.</summary>
    public CharacterController ResolveTargetForUI()
    {
        return ResolveTarget();
    }

    /// <summary>
    /// The companion plays on an on-device PPO policy (see <see cref="PolicyLearner"/>) that keeps
/// training on the player's own machine from their votes and match outcomes. The rule-based brain is
    /// still used during the warm start and to supply the jump/DI geometry each turn.
    /// </summary>
    /// <summary>
    /// The ally this fighter shares one network with. In the shipped game only the companion plays a
    /// policy, so the player is left null and the companion gets its own. In a 2v1 training arena
    /// both allies are policy-driven and must share one network, so the builder points the second
    /// ally at the first.
    /// </summary>
    private AIController brainSharer;

    /// <summary>Join <paramref name="other"/>'s network instead of creating a separate one.</summary>
    public void SharePolicyWith(AIController other) => brainSharer = other;

    protected override FighterPolicy SelectBrain()
    {
        return PolicyLearner.CreatePolicy(this, ruleBrain, PolicyLearner.RoleCompanion,
                                          learnOnline: true, shareWith: brainSharer);
    }

    public override void RequestDecision()
    {
        base.RequestDecision();
        // Capture the committed transition and fold in any pending PPO update.
        PolicyLearner.NotifyDecision(this);
    }

    /// <summary>Called by TurnManager when the turn starts so the companion can vote-prompt later.</summary>
    public void PrepareVoting()
    {
        CompanionVoteManager.Instance?.BeginVoteFor(this, CurrentDecision, DecisionContext, voteTimeout);
    }
}