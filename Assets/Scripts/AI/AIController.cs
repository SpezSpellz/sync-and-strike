using UnityEngine;

/// <summary>
/// Base for every fighter that plays itself. Holds the <see cref="FighterAI"/> brain,
/// resolves who the current opponent is and applies the decision to the controller.
/// The player-facing <see cref="PlayerController"/> does NOT derive from this.
/// </summary>
public abstract class AIController : CharacterController
{
    [SerializeField]
    private FighterAI.Personality personality = new FighterAI.Personality();

    [SerializeField]
    private int seed = 12345;

    [Tooltip("How much the AI wanders its seed each match so repeated runs are not identical.")]
    private int seedJitter = 0;

    protected FighterAI ruleBrain;
    private FighterPolicy brain;
    // Arena and its _arena cache are declared once on CharacterController. Redeclaring them here
    // would shadow the base field and make Unity serialize the same name twice.
    private TurnManager TurnMgr => ArenaFor(this) != null ? ArenaFor(this).TurnManager : TurnManager.Instance;
    private FighterAI.Personality runtimePersonality;

    /// <summary>The decision made for the current turn. Read by the voting UI and the trainer.</summary>
    public AIDecision CurrentDecision { get; private set; }
    public FighterPolicy Brain => brain;
    /// <summary>The rule-based brain, always available even when the active brain is a PPO policy.</summary>
    public FighterAI RuleBrain => ruleBrain;
    public FighterAI.Personality Profile => runtimePersonality;

    /// <summary>Snapshot taken when the decision was made, used to score the decision later.</summary>
    public AIDecisionContext DecisionContext { get; private set; }

    protected virtual void InitializeAI()
    {
        // FighterAI.Personality is a plain serializable class, not a UnityEngine.Object,
        // so copy it field by field rather than using Instantiate.
        runtimePersonality = CopyPersonality(personality);
        ConfigurePersonality(runtimePersonality);
        ruleBrain = new FighterAI(runtimePersonality, seed + Random.Range(0, Mathf.Max(1, seedJitter)));
        brain = SelectBrain();
    }

    /// <summary>
    /// Choose the active brain. Defaults to the rule-based brain, which is what the frozen enemy
    /// ships as. Subclasses override to return a learned policy (see <see cref="PolicyLearner"/>).
    /// </summary>
    protected virtual FighterPolicy SelectBrain()
    {
        return ruleBrain;
    }

    /// <summary>
    /// Public accessor so a policy learner can rebuild observations for this fighter without
    /// knowing the concrete controller type.
    /// </summary>
    public CharacterController TargetForTraining => ResolveTarget();

    /// <summary>Public accessor for the friendly fighter, used only for companion spacing.</summary>
    public CharacterController AllyForTraining => ResolveAlly();

    /// <summary>
    /// The PPO policy currently driving this fighter, or null when it is using the rule-based brain
    /// (before initialization, or for a fighter that opted out of learning). Read by the telemetry
    /// writer so it does not need to know how the session/policy indirection is arranged.
    /// </summary>
    public NeuralPolicy PolicyForTelemetry => PolicyLearner.PolicyFor(this);

    private static FighterAI.Personality CopyPersonality(FighterAI.Personality from)
    {
        return new FighterAI.Personality
        {
            attackRange = from.attackRange,
            approachRange = from.approachRange,
            aggression = from.aggression,
            blockChance = from.blockChance,
            jumpChance = from.jumpChance,
            punishChance = from.punishChance,
            avoidAlly = from.avoidAlly,
            allyBuffer = from.allyBuffer,
        };
    }

    /// <summary>Hook for subclasses to tune the deserialized personality before the brain is built.</summary>
    protected virtual void ConfigurePersonality(FighterAI.Personality profile)
    {
    }

    protected virtual CharacterController ResolveTarget()
    {
        return null;
    }

    /// <summary>Fighter on the same team, used by the companion to keep its distance.</summary>
    protected virtual CharacterController ResolveAlly()
    {
        return null;
    }

    public override void Start()
    {
        base.Start();
        InitializeAI();
    }

    public override void RequestDecision()
    {
        if (IsDead())
        {
            TurnMgr.ResetState();
            return;
        }

        var target = ResolveTarget();
        CurrentDecision = brain.Decide(this, target, ResolveAlly());
        DecisionContext = AIDecisionContext.Capture(this, target, ResolveAlly(), CurrentDecision);

        ApplyDecision(CurrentDecision);
        TurnMgr.SubmitMove(this);
    }

    /// <summary>
    /// Translates an <see cref="AIDecision"/> into controller state. Uses the non-preview
    /// overloads of setJumpInfo/setKnockbackInfo because AI moves are committed, not previewed.
    /// </summary>
    protected void ApplyDecision(AIDecision decision)
    {
        Flip(decision.flipped);

        // Committed values (previewOnly = false) so the executed move uses them.
        setKnockbackInfo(decision.diPower, decision.diAngle, false);
        setJumpInfo(decision.jumpPower, decision.jumpAngle, false);

        if (!TrySelectMove(decision.moveId))
        {
            // Fall back to something always legal rather than stalling the turn.
            SelectMove("idle");
        }
        LastSubmittedMove = SelectedMove;
    }
}