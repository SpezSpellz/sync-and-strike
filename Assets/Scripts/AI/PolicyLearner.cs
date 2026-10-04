using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Shared PPO learner for any AI fighter. Records one transition per turn (observation, the action
/// played, and a reward derived from what happened), runs PPO updates, and persists the policy to
/// disk so learned behaviour survives sessions.
///
/// Two roles use it:
/// <list type="bullet">
/// <item><b>companion</b> — trains on the player's own machine from the turn outcome plus the
/// player's good/bad vote. The vote arrives one step after the reward, so it is applied as a late
/// adjustment to the previous transition.</item>
/// <item><b>enemy</b> — trains OFFLINE against the rule-based baseline, then ships frozen: the
/// trained weights are loaded but <c>learnOnline</c> stays false, so the shipped enemy never
/// changes while the player is playing.</item>
/// </list>
///
/// Because both roles share this class, the enemy and the companion can later be trained against
/// each other (cross-play) with no duplicated training machinery.
/// </summary>
public class PolicyLearner
{
    /// <summary>Reward weights for a resolved turn.</summary>
    [Serializable]
    public class RewardConfig
    {
        public float hitLanded = 0.5f;
        public float damageDealt = 0.04f;  // per health point
        public float damageTaken = 0.02f;  // per health point
        public float killBonus = 5f;

        /// <summary>
        /// Reward per world unit of distance closed on the opponent this turn.
        ///
        /// Without this there is NO gradient at all until a hit actually lands: from the starting
        /// separation every action, including doing nothing, scores exactly 0. A policy in that
        /// situation cannot tell good from bad, and empirically collapsed onto "block" and stayed
        /// there forever. Closing distance is the first thing a fighter has to learn.
        ///
        /// Per WORLD UNIT, and the fighters start roughly 20 units apart.
        ///
        /// Three runs have now bracketed this value, which is worth recording because both extremes were
        /// measured rather than assumed:
        ///
        ///   0.4  - too strong. Worth +8 per match and up to +0.4 in a single turn, twenty times a landed
        ///          hit. Policies learned to sit where their attacks could not reach: whiff 0.993, ~10
        ///          damage per match, 99% of matches timing out.
        ///   0.05 - too weak. With the policy also being destroyed by an oversized Adam step, nobody
        ///          closed the gap: median separation stayed at 16.1 units against a hitbox that reaches
        ///          about 2.3, so nothing could connect at all.
        ///   0.15 - chosen. Still only ~+3 per match for closing the whole gap, so it cannot dominate the
        ///          damage terms, but large enough to be worth doing over several turns from 16 units.
        ///
        /// The structural fact this exposes: horizontal_slash reaches ~2.3 units (offsetX 1.3 + half of
        /// width 2.0) and the rule brain only attacks within attackRange 1.35, so NOTHING can hit from
        /// where the fighters actually stand. If this value still fails to close the gap, the problem is
        /// the distance itself - the arena or walkf's reach - not the reward, and more tuning here would
        /// just be chasing it.
        /// </summary>
        public float approach = 0.15f;

        /// <summary>
        /// Reward for a turn that achieved nothing: no damage dealt and no ground gained.
        ///
        /// This is the fix for the flat-reward deadlock. Until now every action that neither
        /// connected nor was hit scored EXACTLY zero - not approximately zero, precisely zero - so
        /// blocking, idling and swinging at air were all mathematically tied. With ties everywhere
        /// there is no gradient to descend, and the policy is free to drift onto whichever action
        /// happened to be sampled most often. Empirically that was "block", and the run stalled
        /// with every fighter guarding until the turn cap turned matches into draws.
        ///
        /// Kept small on purpose. It only has to break a tie between doing nothing and trying; if it
        /// ever outweighs landing a hit it teaches the policy to avoid committing to attacks, which
        /// is the opposite of what we want.
        ///
        /// It is per TURN, though, and that makes it a term that compounds: at 0.05 over a match
        /// that timed out at 400 turns it summed to -20, which is larger than the entire win bonus and
        /// larger than all the damage terms together. The per-turn penalty was therefore the single
        /// loudest signal in the return, and every policy that minimised it correctly learned never to
        /// commit. Halved to 0.02 so it stays a tie-breaker rather than the objective.
        /// </summary>
        public float stalemate = 0.02f;

        public float voteGood = 1.5f;
        public float voteBad = 1.5f;
    }

    /// <summary>
    /// The shared half of a policy: weights, optimiser, rollout buffer, and the warm-start clone
    /// queue. One of these per ROLE.
    ///
    /// Split out from <see cref="Session"/> because 2v1 puts two fighters on one network, and the two
    /// halves have opposite lifetimes. Weights are genuinely shared (that is the point), but the
    /// per-turn bookkeeping is not: each fighter has its own observation, its own chosen action and
    /// its own health readings, so a single set of pending fields would have one fighter's decision
    /// overwritten by the other's before the turn resolved.
    /// </summary>
    private class SharedBrain
    {
        public string role;
        public NeuralNetwork net;
        public PPOTrainer trainer;
        public bool warmStart = true;
        public int turns = 0;
        public int updates = 0;
        public float lastReward;

        /// <summary>
        /// Expert samples pending a clone batch. Per brain, not global: a single shared list would mix
        /// arenas' observations together when several arenas train in one process.
        /// </summary>
        public readonly List<float[]> cloneObs = new List<float[]>();
        public readonly List<int> cloneActions = new List<int>();

        // Expert jump/DI values as RAW z targets for the clone regression. One entry per cloneObs, and a
        // null entry means "the expert supplied no geometry for this sample", so a policy without scalars
        // degrades to cloning the move alone instead of failing.
        public readonly List<float[]> cloneScalarZ = new List<float[]>();

    /// <summary>Monte-Carlo return target per clone sample, for value pretraining. See EstimatedReturn.</summary>
    public readonly List<float> cloneReturns = new List<float>();

        // Range descriptors for turning an expert gameplay value into a raw z.
        public ContinuousHead[] cloneHeads;
    }

    /// <summary>
    /// The per-fighter half of a policy. One per AI controller, even when several fighters share a
    /// single <see cref="SharedBrain"/>.
    /// </summary>
    private class Session
    {
        public AIController owner;
        public SharedBrain brain;

        /// <summary>
        /// This fighter's own view of the shared network.
        ///
        /// Deliberately a separate NeuralPolicy per fighter rather than one shared instance: the
        /// policy object holds LastObs / LastAction / LastLogProb / LastValue for the pending
        /// transition, and two fighters sharing one of those would clobber each other every turn.
        /// The underlying NeuralNetwork IS shared, which is the part that has to be shared.
        /// </summary>
        public NeuralPolicy policy;

        public bool hasPending;
        public float pendingSelfHealth;
        public float pendingTargetHealth;
        /// <summary>Distance to the target when the move was committed, for approach shaping.</summary>
        public float pendingDistance;

        // Convenience passthroughs, so the reward/clone code below reads naturally.
        public string role => brain.role;
        public NeuralNetwork net => brain.net;
        public PPOTrainer trainer => brain.trainer;
        public bool warmStart => brain.warmStart;

        /// <summary>Set once a NaN warning has been logged, so it is not repeated every turn.</summary>
        public bool poisonWarned;
        public int turns { get => brain.turns; set => brain.turns = value; }
        public int updates { get => brain.updates; set => brain.updates = value; }
        public float lastReward { get => brain.lastReward; set => brain.lastReward = value; }
        public List<float[]> cloneObs => brain.cloneObs;
        public List<int> cloneActions => brain.cloneActions;
        public List<float[]> cloneScalarZ => brain.cloneScalarZ;
    public List<float> cloneReturns => brain.cloneReturns;

    /// <summary>
    /// Running discounted reward sum during warm start, so each expert turn's return reflects
    /// everything banked so far this match rather than just the latest turn. Used as the critic's
    /// value-pretraining target.
    ///
    /// Per FIGHTER, not per brain - and this is a correction. It used to live on the SharedBrain,
    /// because keeping it per-fighter was thought to let the two allies overwrite each other's
    /// trajectory. The opposite happened: BOTH allies accumulate into it every turn, so the target
    /// the critic regressed onto was the sum of two independent trajectories rather than either one.
    /// In 2v1 the critic was therefore pretrained to predict roughly TWICE the return it was ever
    /// asked for, which is a direct explanation for the enormous return and advantage scale measured
    /// in the last run (return std ~98 against a value std of 6, advantages up to 1e31).
    ///
    /// Per fighter is also simply correct: a return belongs to the trajectory that earned it, and
    /// each fighter has its own.
    /// </summary>
    public float warmStartReturn;
        public ContinuousHead[] cloneHeads { get => brain.cloneHeads; set => brain.cloneHeads = value; }
    }

    private static readonly Dictionary<int, Session> sessions = new Dictionary<int, Session>();

    /// <summary>
    /// Optional observation normalisation, per role.
    ///
    /// Off by default because it is not free: the statistics change as a network trains, so a
    /// network that has just been created and one that has trained for a hundred thousand turns
    /// normalise the same observation differently. Enabling it is a decision, and the statistics have
    /// to travel with the weights.
    /// </summary>
    private static readonly ObservationNormalizer normalizer = new ObservationNormalizer();

    /// <summary>The role-level normaliser, exposed for the command line and for tests.</summary>
    public static ObservationNormalizer Normalizer => normalizer;

    // Warm start: for the first N turns the companion plays the rule-based expert and clones it,
    // so it begins competent rather than random, then PPO takes over.
    private const int WarmStartTurns = 200;
    private const int CloneBatchSize = 32;
    private const int SaveEveryUpdates = 25;
    private const int Hidden = 32;

    // The critic's trunk is deliberately WIDER than the policy's.
    //
    // At 32/32 the critic's explained variance sat at ~0 - it matched a constant and nothing more, so
    // every advantage carried almost no signal about which action was better. The two heads are not
    // asking the same question: the policy ranks discrete actions and can get by on 32 units, while
    // the critic has to integrate reward over a ~100-turn effective horizon (gamma 0.99) and resolve
    // small differences in expected outcome. Sizing them together made the critic the smaller of the
    // two by accident rather than by choice.
    //
    // Overridable via -ppo.valueHidden so the width can be swept rather than argued about.
    private static int valueHidden = 64;
    private static int ValueHidden => Mathf.Clamp(valueHidden, 8, 512);

    // Gaussian scalar heads per policy: jump power, jump angle, DI power, DI angle.
    //
    // Overridable so the hybrid policy can be A/B measured against the previous move-only behaviour by
    // passing -ppo.scalars=0, and so a network trained without scalars can still be loaded and used.
    private static int scalarsPerPolicy = ActionScalars.Count;
    private static int ScalarsPerPolicy => Mathf.Clamp(scalarsPerPolicy, 0, ActionScalars.Count);
    private const float CloneLearningRate = 1e-3f;

    /// <summary>
    /// Roles that ship frozen. A frozen role still loads and uses a trained policy, but records no
    /// transitions and runs no updates, so it cannot drift during a player's match.
    ///
    /// A <c>-training</c> run is the one exception: that is where the enemy is trained offline.
    /// Outside training mode the enemy is always frozen.
    /// </summary>
    private static readonly HashSet<string> FrozenRoles = new HashSet<string> { RoleEnemy };

    /// <summary>
    /// The session that owns each shareable role's single network. See CreatePolicy's shareWith note.
    /// </summary>
    private static readonly Dictionary<string, int> sessionRoleIndex = new Dictionary<string, int>();

    /// <summary>
    /// Whether a role is currently allowed to record transitions and run updates. Frozen roles are
    /// unfrozen only inside an explicit training run.
    /// </summary>
    private static bool IsLearning(string role) => !FrozenRoles.Contains(role) || TrainingMode.enabled;

    /// <summary>Role key for the companion: trains on the player's machine.</summary>
    public const string RoleCompanion = "companion";

    /// <summary>Role key for the enemy: trained offline, shipped frozen.</summary>
    public const string RoleEnemy = "enemy";

    private static RewardConfig rewardConfig = new RewardConfig();

    /// <summary>Reward weights, exposed so they can be tuned from the inspector in play.</summary>
    public static RewardConfig Config => rewardConfig;

    /// <summary>
    /// The PPO hyperparameter set shared by every brain, so a command-line sweep can retune the
    /// learner without recompiling. One instance for all brains on purpose: the hyperparameters belong
    /// to the algorithm, not to an individual network.
    /// </summary>
    private static readonly PPOTrainer.Hyper sharedHyper = new PPOTrainer.Hyper();

    /// <summary>
    /// Set one PPO hyperparameter by name, for <c>-ppo.name=value</c> command-line overrides.
    /// Returns false for an unknown field so the caller can warn rather than pretend it applied.
    ///
    /// These take effect on brains created AFTER the override is applied, which for a command-line run
    /// means every brain in the process.
    /// </summary>
    public static bool TrySetHyper(string field, float value)
    {
        switch (field)
        {
            case "scalars": scalarsPerPolicy = Mathf.Clamp(Mathf.RoundToInt(value), 0, ActionScalars.Count); return true;
            case "normalizer": normalizer.enabled = value != 0f; return true;
            case "normalizePersist": normalizer.persist = value != 0f; return true;
            case "learningRate": sharedHyper.learningRate = value; return true;
            case "gamma": sharedHyper.gamma = value; return true;
            case "lambda": sharedHyper.lambda = value; return true;
            case "valueLambda": sharedHyper.valueLambda = Mathf.Clamp01(value); return true;
            case "clipEpsilon": sharedHyper.clipEpsilon = value; return true;
            case "valueCoef": sharedHyper.valueCoef = value; return true;
            case "entropyCoef": sharedHyper.entropyCoef = value; return true;
            case "epochs": sharedHyper.epochs = Mathf.Max(1, Mathf.RoundToInt(value)); return true;
            case "minibatch": sharedHyper.minibatch = Mathf.Max(2, Mathf.RoundToInt(value)); return true;
            case "bufferSize": sharedHyper.bufferSize = Mathf.Max(8, Mathf.RoundToInt(value)); return true;
            case "batchSize": sharedHyper.batchSize = Mathf.Max(8, Mathf.RoundToInt(value)); return true;
            case "maxGradNorm": sharedHyper.maxGradNorm = value; return true;
            case "targetKl": sharedHyper.targetKl = value; return true;
            case "useAdam": sharedHyper.useAdam = value != 0f; return true;
            case "valueHidden": valueHidden = Mathf.Max(8, Mathf.RoundToInt(value)); return true;
            default:
                Debug.LogWarning($"[Training] unknown ppo field '{field}'; ignored.");
                return false;
        }
    }

    /// <summary>
    /// Create (or fetch) a PPO policy for an AI fighter and wire it to that fighter's turn loop.
    /// </summary>
    /// <param name="learnOnline">
    /// True for the companion (keeps learning on the player's machine). False for the shipped enemy,
    /// which loads trained weights but must never change during play.
    /// </param>
    public static NeuralPolicy CreatePolicy(AIController owner, FighterAI ruleBrain,
                                             string role, bool learnOnline)
    {
        return CreatePolicy(owner, ruleBrain, role, learnOnline, shareWith: null);
    }

    /// <summary>
    /// Create or fetch a policy that shares one network with other fighters of the same role.
    ///
    /// The 2v1 training arena has TWO allies (player and companion) that are deliberately driven by
    /// one brain. They must not merely load the same weights file: two independently created
    /// networks would each write enemy/companion_policy.json on their own schedule, so whichever
    /// saved last would silently overwrite the other's progress, and at match end the two bodies
    /// would be playing with different weights than they trained. Sharing the Session itself means
    /// one rollout buffer, one update, one save.
    /// </summary>
    /// <param name="shareWith">
    /// A fighter already set up for this role. When non-null its session is reused and this fighter
    /// simply joins it. Its OWN FighterAI rule brain is ignored in favour of the sharer's, because
    /// one network can only have been warm-started from one expert.
    /// </param>
    public static NeuralPolicy CreatePolicy(AIController owner, FighterAI ruleBrain,
                                             string role, bool learnOnline,
                                             AIController shareWith)
    {
        var key = owner.GetInstanceID();
        if (sessions.TryGetValue(key, out var existing))
            return existing.policy;

        // Resolve which brain this fighter joins. Explicit sharer first, then any existing brain for
        // the same role, then a brand new one.
        SharedBrain brain;
        if (shareWith != null && sessions.TryGetValue(shareWith.GetInstanceID(), out var sharerSession))
        {
            brain = sharerSession.brain;
        }
        else if (sessionRoleIndex.TryGetValue(role, out int existingKey)
                 && sessions.TryGetValue(existingKey, out var roleSession))
        {
            brain = roleSession.brain;
        }
        else
        {
            brain = null;
        }

        if (brain != null)
        {
            // Join an existing brain. A fresh NeuralPolicy over the SHARED network, so this fighter
            // keeps its own pending transition, and its own warm-start flag mirror below.
            var joined = new NeuralPolicy(owner, ruleBrain, brain.net,
                                          new System.Random(owner.GetInstanceID()), ScalarsPerPolicy);
            joined.Trainer = brain.trainer;
            joined.Role = role;
            joined.ForceExpert = brain.warmStart;
            joined.Turns = brain.turns;
            joined.Updates = brain.updates;

            var joinSession = new Session { owner = owner, brain = brain, policy = joined };
            sessions[key] = joinSession;

            // Subscribe PER FIGHTER, not per brain. Sharing one subscription would mean only the
            // first ally's transition was ever recorded.
            Arena joinedArena = Arena.Of(owner);
            TurnManager joinedTurns = joinedArena != null ? joinedArena.TurnManager : TurnManager.Instance;
            joinedTurns.TurnResolved += () => OnTurnResolved(key);
            if (learnOnline)
            {
                CompanionVoteManager.VoteResolved += (isGood, timedOut) => OnVoteResolved(key, isGood, timedOut);
            }
            return joined;
        }

        int obsSize = AIDecisionContext.FeatureNames().Length;
        // The action count depends on the unique move ids, so read it off a throwaway policy.
        var probe = new NeuralPolicy(owner, ruleBrain, new NeuralNetwork(obsSize, Hidden, 2, 1, ScalarsPerPolicy), new System.Random(1));
        int actionCount = probe.ActionCount;

        var net = new NeuralNetwork(obsSize, Hidden, actionCount, owner.GetInstanceID(), ScalarsPerPolicy, ValueHidden);
        // Weights load whenever they exist, from persistent data or from the build. They used to require
        // the -resume flag, which meant a shipped companion never carried its learning across sessions
        // and a shipped enemy never used its trained policy unless the player passed a training switch.
        WeightSource source = LoadWeights(net, role);
        bool resumed = source != WeightSource.None;
        Debug.Log(resumed
            ? $"[AI] {role} loaded {source} weights (max|weight| {net.MaxAbsWeight:0.###})."
            : $"[AI] {role} has no trained weights; starting from the rule-based warm start.");
        WarnIfNoPretrainedCompanion(source, role);

        // Statistics travel with the weights, and only with the player's own. They describe the input
        // distribution of one specific network, so pairing SHIPPED weights with local statistics - or the
        // reverse - feeds the first few hundred turns nonsense, which is exactly the window the warm
        // start was protecting.
        if (source == WeightSource.Persistent) LoadNormalizerStats(role);

        var trainer = new PPOTrainer(net, sharedHyper, new System.Random(owner.GetInstanceID()));
        var policy = new NeuralPolicy(owner, ruleBrain, net, new System.Random(owner.GetInstanceID()), ScalarsPerPolicy);
        policy.Trainer = trainer;

        // A fresh policy that will be trained needs the rule-based warm start, otherwise it starts random
        // and spends its first turns flailing. But a policy resumed from disk already holds trained
        // weights, and cloning the expert over it would throw that training away — so a resumed run
        // goes straight to PPO.
        bool willLearn = learnOnline || TrainingMode.enabled;
        bool warmStart = willLearn && !resumed;
        policy.ForceExpert = warmStart;

        var session = new Session
        {
            owner = owner, policy = policy,
            brain = new SharedBrain
            {
                role = role, net = net, trainer = trainer, warmStart = warmStart,
            },
        };
        policy.Role = role;
        sessions[key] = session;

        var turnMgr = Arena.Of(owner) != null ? Arena.Of(owner).TurnManager : TurnManager.Instance;
        turnMgr.TurnResolved += () => OnTurnResolved(key);

        // Index this session by role so a later fighter asking for the same role joins THIS network
        // instead of creating a rival one that would overwrite the same weights file. Only the
        // companion role is shared; a second enemy would be a genuinely different fighter.
        if (role == RoleCompanion) sessionRoleIndex[role] = key;
        if (learnOnline)
        {
            // Only the online companion is scored by the player's vote.
            CompanionVoteManager.VoteResolved += (isGood, timedOut) => OnVoteResolved(key, isGood, timedOut);
        }

        return policy;
    }

    /// <summary>Per-brain learner diagnostics, for the training health line and the CSV.</summary>
    public struct BrainHealth
    {
        public int updates;
        public int buffered;
        public float explainedVariance;
        public float approxKl;
        public float clipFraction;
        public float valueLoss;
        public float advantageMagnitude;
        public int rejected;
        public int nonfinite;
        public int clippedSteps;
        public int clippedCriticSteps;
        public float valueStd, returnStd, valueReturnCorr, chainedFraction;
        public bool poisoned;
        public float maxAbsWeight;
        public int epochsRun;
        public int minibatchesRun;
        public float klAll;
    }

    /// <summary>
    /// One entry per shared brain, i.e. per ROLE rather than per fighter.
    ///
    /// With 2v1 there are two allies on one brain, so listing that network once per fighter would read
    /// as two independent learners and invite tuning one of them into the ground.
    /// </summary>
    public static Dictionary<string, BrainHealth> BrainStats()
    {
        var result = new Dictionary<string, BrainHealth>();
        var seen = new HashSet<string>();
        foreach (var kv in sessions)
        {
            var b = kv.Value.brain;
            if (b == null || b.trainer == null || !seen.Add(b.role)) continue;
            result[b.role] = new BrainHealth
            {
                updates = b.updates,
                buffered = b.trainer.Buffered,
                explainedVariance = b.trainer.ExplainedVariance,
                approxKl = b.trainer.ApproxKl,
                clipFraction = b.trainer.ClipFraction,
                valueLoss = b.trainer.ValueLoss,
                advantageMagnitude = b.trainer.AdvantageMagnitude,
                rejected = b.trainer.RejectedTransitionCount,
                nonfinite = b.trainer.NonFiniteSteps,
                clippedSteps = b.trainer.ClippedSteps,
                clippedCriticSteps = b.trainer.ClippedCriticSteps,
                valueStd = b.trainer.ValueStd,
                returnStd = b.trainer.ReturnStd,
                valueReturnCorr = b.trainer.Correlation,
                chainedFraction = b.trainer.ChainedFraction,
                poisoned = b.trainer.NetworkIsPoisoned,
                maxAbsWeight = b.trainer.MaxAbsWeight,
                epochsRun = b.trainer.EpochsRun,
                minibatchesRun = b.trainer.MinibatchesRun,
                klAll = b.trainer.MeanKlAcrossEpochs,
            };
        }
        return result;
    }

    /// <summary>
    /// The live policy backing this fighter, or null if it runs the rule-based brain.
    /// Exposed so the telemetry writer can read entropy and action share without knowing how the
    /// session/policy indirection is arranged.
    /// </summary>
    public static NeuralPolicy PolicyFor(AIController owner)
    {
        if (owner == null) return null;
        return sessions.TryGetValue(owner.GetInstanceID(), out var s) ? s.policy : null;
    }

    /// <summary>Called right after the fighter commits a move for the turn.</summary>
    public static void NotifyDecision(AIController owner)
    {
        if (!sessions.TryGetValue(owner.GetInstanceID(), out var s)) return;

        var p = s.policy;

        if (s.warmStart)
        {
            // Clone into the SHARED queue. Both allies' expert samples train the one network, which is
            // what we want, but each fighter contributes its own observations.
            s.cloneObs.Add(p.LastObs);
            s.cloneActions.Add(p.LastExpertAction);
            s.cloneScalarZ.Add(ExpertScalarTargets(s));
            // Monte-Carlo return target for value pretraining, from arXiv:2503.01491.
            //
            // During warm start the expert is playing, so the trajectory's outcome is knowable in closed
            // form - which is exactly the "train the value model on Monte-Carlo returns under a fixed
            // policy" recipe the paper prescribes for fixing a collapsed critic. Fitting V before PPO's
            // first policy update is what stops the critic and the policy gradient from bootstrapping
            // each other off a bad initialisation.
            //
            // Recorded here, at DECISION time, on the expert's chosen action. The turn's reward is not
            // known until OnTurnResolved, so the running sum is advanced there instead; see
            // AccumulateWarmStartReturn.
            s.cloneReturns.Add(s.warmStartReturn);
            if (s.cloneObs.Count >= CloneBatchSize)
            {
                CloneBatch(s);
                s.cloneObs.Clear();
                s.cloneActions.Clear();
                s.cloneScalarZ.Clear();
                s.cloneReturns.Clear();
            }
            s.hasPending = true;
            s.pendingSelfHealth = p.LastSelfHealth;
            s.pendingTargetHealth = p.LastTargetHealth;
            s.pendingDistance = DistanceToTarget(owner);
            return;
        }

        // A frozen policy is observation-only: it never records or learns, it just fights.
        if (!IsLearning(s.role)) return;

        // Apply any pending PPO update now, after the previous turn's vote has been folded in.
        // Guarded on the brain's counter, not this fighter's, so two fighters sharing a network do
        // not both fire an update for the same buffer.
        if (s.trainer.ReadyToUpdate())
        {
            s.trainer.Update();
            s.updates++;
            if (s.updates % SaveEveryUpdates == 0) SaveWeights(s.net, s.role);
        }

        s.hasPending = true;
        s.pendingSelfHealth = p.LastSelfHealth;
        s.pendingTargetHealth = p.LastTargetHealth;
        s.pendingDistance = DistanceToTarget(owner);
    }

    private static float DistanceToTarget(AIController owner)
    {
        var target = owner.TargetForTraining;
        if (owner == null || target == null) return 0f;
        return Vector2.Distance(owner.GetPosition(), target.GetPosition());
    }

    private static void OnTurnResolved(int key)
    {
        if (!sessions.TryGetValue(key, out var s) || !s.hasPending) return;
        s.hasPending = false;
        if (!IsLearning(s.role)) return;

        var self = s.owner;
        if (self == null) return;

        var target = self.TargetForTraining;
        var ally = self.AllyForTraining;
        float selfNow = self.GetHealth();
        float targetNow = target != null ? target.GetHealth() : 0f;

        float dmgDealt = Mathf.Max(0f, s.pendingTargetHealth - targetNow);
        float dmgTaken = Mathf.Max(0f, s.pendingSelfHealth - selfNow);

        float reward = 0f;
        if (dmgDealt > 0f) reward += rewardConfig.hitLanded;
        reward += rewardConfig.damageDealt * dmgDealt;
        reward -= rewardConfig.damageTaken * dmgTaken;
        // Paid once, on the turn the kill actually lands. Gating on dmgDealt > 0 means the bonus is only
        // paid when the target's health DROPPED this turn, i.e. the turn that killed them. Testing only
        // IsDead() pays +5 on every subsequent turn of the same match, which with gamma=0.99 chains
        // into a large spurious terminal return - and a dead opponent is exactly the situation that
        // happens most often at the end of a won match.
        if (target != null && target.IsDead() && dmgDealt > 0f) reward += rewardConfig.killBonus;

        bool done = self.IsDead() || (target != null && target.IsDead());

        // Approach shaping: reward closing the gap, so there is a gradient before any hit lands.
        float closed = 0f;
        if (target != null)
        {
            float now = Vector2.Distance(self.GetPosition(), target.GetPosition());
            closed = s.pendingDistance - now;
            reward += rewardConfig.approach * closed;
        }

        // Stalemate penalty: a turn that dealt no damage AND gained no ground was a wasted turn.
        // Applied only on non-terminal turns. When the match is resolving on its own terms the
        // outcome is already unambiguous, and charging for the final turn of a lost fight adds noise
        // to the return instead of signal. Also skipped while the expert is driving, since the
        // expert does not stall and the penalty would just offset every warm-start reward.
        if (!done && !s.warmStart && dmgDealt <= 0f && closed <= 0f)
            reward -= rewardConfig.stalemate;

        // Value pretraining: fold this turn's reward into the running Monte-Carlo return that the
        // clone batches regress the critic against. Warm start only, because after it the trajectories
        // are the LEARNER's and their outcomes are not knowable in closed form.
        if (s.warmStart) AccumulateWarmStartReturn(s, reward);

        var nextObs = s.policy.CaptureObservation(self, target, ally);
        // The raw scalar samples ride along with the transition. Cloned rather than referenced, because
        // the policy reuses and overwrites that array on the very next decision and the buffer has to
        // keep describing the action that was actually played.
        float[] rawZ = s.policy.ScalarCount > 0 ? (float[])s.policy.LastRawZ.Clone() : null;
        s.trainer.Add(s.policy.LastObs, nextObs, s.policy.LastAction,
                      s.policy.LastLogProb, s.policy.LastValue, reward, done, rawZ,
                      agentId: self.GetInstanceID(), episode: s.trainer.CurrentEpisode);
        WarnIfPoisoned(s);
        s.lastReward = reward;
        s.turns++;
        // Mirror the counters onto this fighter's policy so telemetry can read them without session
        // access. Entropy and top-1 share are NOT mirrored: each policy measures its own sampling, and
        // copying the brain's numbers here would make two fighters report one body's statistics - which
        // is exactly the misreading that made the shared-brain bug look like two agreeing agents.
        s.policy.LastReward = reward;
        s.policy.Turns = s.turns;
        s.policy.Updates = s.updates;

        // Warm start is a property of the BRAIN, so when it ends, every fighter using this network
        // must stop deferring to its own expert. Only this fighter's policy is reachable here, so the
        // others are swept below.
        if (s.warmStart && s.turns >= WarmStartTurns)
        {
            s.brain.warmStart = false;
            EndWarmStartForRole(s.role);
        }

        // A training run must not stop at the turn cap of a single match: keep the weights current
        // so a long headless run keeps improving instead of holding whatever it had at match one.
        if (TrainingMode.enabled && s.trainer.ReadyToUpdate())
        {
            s.trainer.Update();
            s.updates++;
            if (s.updates % SaveEveryUpdates == 0) SaveWeights(s.net, s.role);
        }
    }

    /// <summary>
    /// Stop every fighter on <paramref name="role"/> deferring to its rule-based expert.
    ///
    /// Warm start ends per BRAIN, but each fighter owns a separate NeuralPolicy (so its pending
    /// transition is its own). Clearing only the fighter that happened to cross the threshold would
    /// leave the other one permanently stuck playing the expert while still recording transitions, so
    /// it would look like it was learning while never actually driving.
    /// </summary>
    private static void EndWarmStartForRole(string role)
    {
        foreach (var kv in sessions)
            if (kv.Value.role == role) kv.Value.policy.ForceExpert = false;
    }

    // Convert the rule-based expert's gameplay-space jump/DI values into raw z targets for the clone
    // regression. Returns null when there is nothing usable, so the caller falls back to cloning the
    // move alone rather than injecting a bad target.
    private static float[] ExpertScalarTargets(Session s)
    {
        if (s.owner == null || !s.owner.HasExpertGeometry) return null;
        AIDecision expert = s.owner.ExpertGeometry;

        var heads = s.cloneHeads ?? (s.cloneHeads = ActionScalars.CreateAll());
        var z = new float[heads.Length];
        z[ActionScalars.JumpPower] = heads[ActionScalars.JumpPower].Unsquash(expert.jumpPower);
        z[ActionScalars.JumpAngle] = heads[ActionScalars.JumpAngle].Unsquash(expert.jumpAngle);
        z[ActionScalars.DiPower] = heads[ActionScalars.DiPower].Unsquash(expert.diPower);
        z[ActionScalars.DiAngle] = heads[ActionScalars.DiAngle].Unsquash(expert.diAngle);

        // Refuse non-finite targets. A garbage expert value would otherwise put a NaN straight into the
        // clone gradient, which is precisely how an earlier training run killed every policy at once.
        for (int i = 0; i < z.Length; i++)
            if (float.IsNaN(z[i]) || float.IsInfinity(z[i])) return null;
        return z;
    }

    private static void OnVoteResolved(int key, bool isGood, bool timedOut)
    {
        if (!sessions.TryGetValue(key, out var s)) return;
        // A timeout is not a signal: the player did not actually judge the move, so it must not
        // push the policy either way. Only a real good/bad vote counts.
        if (timedOut) return;
        float delta = isGood ? rewardConfig.voteGood : -rewardConfig.voteBad;
        s.trainer.AdjustLastReward(delta);
    }

    /// <summary>Supervised clone of the rule-based expert's moves onto the network.</summary>
    private static void CloneBatch(Session s)
    {
        if (s.cloneObs.Count == 0) return;
        if (s.cloneHeads == null) s.cloneHeads = ActionScalars.CreateAll();

        // Never clone onto a poisoned network. Warm start runs at the very start of a session, so this
        // is cheap insurance rather than a routine check.
        if (!s.net.WeightsAreFinite())
        {
            Debug.LogWarning($"[Training] {s.role}: network has non-finite weights; skipping clone batch.");
            s.cloneObs.Clear();
            s.cloneActions.Clear();
            s.cloneScalarZ.Clear();
            return;
        }

        s.net.ClearGradients();
        var heads = s.cloneHeads;
        var dMu = new float[ActionScalars.Count];
        var dLogSd = new float[ActionScalars.Count];
        for (int i = 0; i < s.cloneObs.Count; i++)
        {
            s.net.Forward(s.cloneObs[i]);
            // Maximising log pi(expert) is exactly dLogProb = 1 on the expert action.
            float[] targetZ = s.cloneScalarZ.Count > i ? s.cloneScalarZ[i] : null;
            if (targetZ != null && heads != null)
            {
                // Pull each scalar mean toward the EXPERT's value, as a regression rather than as a
                // log-prob. Maximising log N(z_expert; mu, sd) would drag sigma toward zero as well as
                // mu toward the target, collapsing the head's spread; this moves only the mean and
                // leaves exploration intact.
                var means = s.net.ScalarMeans;
                for (int k = 0; k < heads.Length; k++) dMu[k] = means[k] - targetZ[k];
                Array.Clear(dLogSd, 0, dLogSd.Length);
            }
            else
            {
                Array.Clear(dMu, 0, dMu.Length);
                Array.Clear(dLogSd, 0, dLogSd.Length);
            }

            // Value pretraining rides along with the behaviour clone.
            //
            // The critic is regressed onto the Monte-Carlo return of the expert trajectory this sample
            // came from, with dValue = V - G. That is ordinary supervised regression on a known target,
            // not a bootstrapped one, so it cannot be self-reinforcing: however wrong V starts, the
            // gradient always points at a real number.
            //
            // This is the "address the value initialisation bias by value pretraining" step from
            // arXiv:2503.01491, and it is why warm start is the right place for it - the expert plays a
            // fixed policy, so its returns are exactly the thing the paper says to fit against.
            float dValue = 0f;
            if (s.cloneReturns.Count > i)
            {
                s.net.Forward(s.cloneObs[i]);
                dValue = s.net.Value - s.cloneReturns[i];
            }

            s.net.Backprop(s.cloneObs[i], s.cloneActions[i], 1f, dValue, 0f, dMu, dLogSd);
        }
        s.net.ApplyGradients(CloneLearningRate / s.cloneObs.Count);
    }

    /// <summary>
    /// Monte-Carlo return of the expert turn that just resolved, used as the value-pretraining target.
    ///
    /// Deliberately a plain discounted sum of what this fighter has actually banked so far this match,
    /// with the terminal win/loss bonus folded in on the deciding turn. It is the reward stream the
    /// reward config already defines, so the critic is trained on the same units the policy is
    /// optimised against rather than on some rescaled variant.
    ///
    /// Stored per clone sample rather than recomputed at fit time because the trajectory is only
    /// complete once it ends; by the time a batch is fitted, early samples would have to be
    /// reconstructed from state that has since been overwritten.
    /// </summary>
    /// <summary>
    /// Advance the warm-start discounted return by one resolved expert turn.
    ///
    /// Called from OnTurnResolved while warm start is active, so the value-pretraining targets
    /// recorded at decision time describe the reward actually earned on that turn. sharedHyper is the
    /// live PPO config, so the pretraining target uses the SAME discount the critic will later be
    /// trained with; a mismatch would make the critic's target inconsistent with its own bootstrapping,
    /// which is the very problem being fixed.
    /// </summary>
    private static void AccumulateWarmStartReturn(Session s, float reward)
    {
        s.warmStartReturn = s.warmStartReturn * sharedHyper.gamma + reward;
    }

    // --- Persistence ---------------------------------------------------------

    /// <summary>Each role persists to its own file so the enemy and companion cannot clobber each other.</summary>
    private static string SavePath(string role) => Path.Combine(Application.persistentDataPath, role + "_policy.json");

    /// <summary>
    /// True when this brain's network has been poisoned by non-finite weights.
    ///
    /// Worth surfacing loudly and once per brain: a NaN network keeps sampling, so nothing about the
    /// run looks broken from the outside - it just always makes the same choice. Without an explicit
    /// check the only symptom is a policy that mysteriously stopped learning.
    /// </summary>
    private static void WarnIfPoisoned(Session s)
    {
        if (s == null || s.net == null || s.net.WeightsAreFinite()) return;
        if (s.poisonWarned) return;
        s.poisonWarned = true;
        Debug.LogError($"[Training] {s.role}: policy weights contain NaN/Inf. This policy is dead and "
            + "cannot recover - delete its weights file to restart the role from a fresh network. "
            + "(Non-finite gradients are now skipped, so a fresh run should not reach this state.)");
    }

    private static void SaveWeights(NeuralNetwork net, string role)
    {
        try
        {
            File.WriteAllText(SavePath(role), JsonUtility.ToJson(net.Export()));
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not save {role} policy: {e.Message}");
        }
    }

    /// <summary>Where a role's weights came from.</summary>
    private enum WeightSource
    {
        /// <summary>Nothing usable on disk; start from the rule-based warm start.</summary>
        None,

        /// <summary>The player's own saved progress in persistentDataPath.</summary>
        Persistent,

        /// <summary>The frozen baseline bundled inside the build.</summary>
        Shipped,
    }

    /// <summary>Frozen weights bundled with the build. Written by Tools > Training > Export ...</summary>
    private static string ShippedPath(string role) =>
        Path.Combine(Application.streamingAssetsPath, "AI", role + "_policy.json");

    /// <summary>True when this build carries a frozen policy for the role.</summary>
    public static bool HasShippedWeights(string role)
    {
        try { return File.Exists(ShippedPath(role)); }
        catch (Exception) { return false; }
    }

    /// <summary>
    /// Load a role's weights, from whichever source is correct for that role.
    ///
    /// A FROZEN role reads the shipped file FIRST: the enemy that plays must be the enemy that shipped,
    /// and it must never silently pick up a stray enemy_policy.json that a training run left in a
    /// player's persistent data. A LEARNING role reads its own persistent file first, because that is the
    /// progress the player earned and the shipped baseline is only a starting point.
    ///
    /// A file that parses but carries no trained signal - all zeros, or non-finite - is treated as absent
    /// so the caller falls back to the rule-based warm start. That guard is load-bearing: an earlier,
    /// barely trained file silently disabled the warm start and the policy collapsed onto one move.
    /// </summary>
    private static WeightSource LoadWeights(NeuralNetwork net, string role)
    {
        bool frozen = FrozenRoles.Contains(role);
        var paths = frozen
            ? new[] { ShippedPath(role), SavePath(role) }
            : new[] { SavePath(role), ShippedPath(role) };

        foreach (var path in paths)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) continue;
            try
            {
                var weights = JsonUtility.FromJson<NeuralNetwork.Weights>(File.ReadAllText(path));
                if (!net.Import(weights))
                {
                    Debug.LogWarning($"[AI] {role} weights at {path} do not match this network's shape; ignored.");
                    continue;
                }
                net.RecomputeMaxWeight();
                if (!net.WeightsAreFinite() || net.MaxAbsWeight < 1e-4f)
                {
                    Debug.LogWarning($"[AI] {role} weights at {path} carry no trained signal; ignored.");
                    continue;
                }
                // The moments describe the trajectory that PRODUCED these weights, which may have come
                // from a different network or a different optimiser. Keeping them would push the first
                // few steps in a direction unrelated to what was just loaded.
                net.ResetOptimizerState();
                return path == SavePath(role) ? WeightSource.Persistent : WeightSource.Shipped;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not load {role} policy from {path}: {e.Message}");
            }
        }
        return WeightSource.None;
    }

    /// <summary>Set once the "no pretrained companion" warning has been logged, so 2v1 building two
    /// fighters on one network does not print the same warning twice.</summary>
    private static bool warnedNoPretrainedCompanion;

    /// <summary>
    /// Tell the player ONCE that the companion has no pretrained weights, in the shipped game.
    ///
    /// Silent fallback is the wrong default here. The companion works perfectly well without weights - it
    /// plays the rule-based brain and warm-starts from it - so nothing looks broken, and that is the
    /// problem. What the player cannot see is that the companion they are getting is NOT the trained one,
    /// and that it will not become trained until they have played a few hundred turns. So this states it
    /// plainly and names both paths the weights were supposed to have arrived by.
    ///
    /// Suppressed during a training run, where starting with no weights is the normal state rather than a
    /// deployment mistake.
    /// </summary>
    private static void WarnIfNoPretrainedCompanion(WeightSource source, string role)
    {
        if (warnedNoPretrainedCompanion) return;
        if (role != RoleCompanion) return;
        if (source != WeightSource.None) return;
        if (TrainingMode.enabled) return;

        warnedNoPretrainedCompanion = true;
        Debug.LogWarning(
            "[AI] The companion has NO PRETRAINED WEIGHTS. It will play the rule-based brain and begin "
            + "learning from your votes and match outcomes, reaching roughly the trained quality only "
            + "after a few hundred turns of play.\nIf you expected a trained companion its weights are "
            + "missing. Looked for:\n  shipped:  "
            + Path.Combine(Application.streamingAssetsPath, "AI", RoleCompanion + "_policy.json")
            + "\n            (create with Tools > Training > Export Companion Baseline)\n  your save: "
            + $"{Path.Combine(Application.persistentDataPath, RoleCompanion + "_policy.json")}");
    }

    /// <summary>Force-save every live policy (e.g. on quit).</summary>
    public static void SaveAll()
    {
        foreach (var s in sessions.Values)
            if (!FrozenRoles.Contains(s.role)) SaveWeights(s.net, s.role);
        SaveNormalizerStats();
    }

    /// <summary>
    /// Persist observation statistics next to the weights, per role.
    ///
    /// Written as a separate file rather than folded into the weights document so that a build shipping
    /// a frozen enemy does not accidentally require a statistics file to exist, and so a missing or
    /// stale statistics file degrades to "normalisation starts fresh" rather than failing the load.
    ///
    /// Only saved when normalisation is actually on. Persisting statistics for a run that never
    /// normalised anything would create a file that implies it did.
    /// </summary>
    private static void SaveNormalizerStats()
    {
        if (!normalizer.enabled || !normalizer.persist) return;
        foreach (var role in new[] { RoleCompanion, RoleEnemy })
        {
            var stats = normalizer.StatsFor(role);
            if (stats?.count == 0) continue;
            try
            {
                File.WriteAllText(StatsPath(role), JsonUtility.ToJson(stats));
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not save {role} observation stats: {e.Message}");
            }
        }
    }

    private static string StatsPath(string role) =>
        Path.Combine(Application.persistentDataPath, role + "_obsstats.json");

    /// <summary>Restore observation statistics for a role, if any were persisted.</summary>
    private static void LoadNormalizerStats(string role)
    {
        if (!normalizer.enabled || !normalizer.persist) return;
        try
        {
            var path = StatsPath(role);
            if (!File.Exists(path)) return;
            var stats = JsonUtility.FromJson<ObservationStats>(File.ReadAllText(path));
            if (stats?.count > 0) normalizer.StatsFor(role).CopyFrom(stats);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not load {role} observation stats: {e.Message}");
        }
    }

    /// <summary>
    /// <summary>
    /// Tell every brain that the current match has ended, so GAE does not chain across the reset: the
    /// final states of one match are not the predecessors of the opening states of the next.
    ///
    /// Called by the training runner. The shipped game has no rollout buffer to invalidate.
    /// </summary>
    public static void NotifyMatchReset()
    {
        foreach (var kv in sessions)
        {
            var s = kv.Value;
            s.brain.trainer.BeginEpisode();
            // The warm-start return accumulator is per fighter and must not carry across matches.
            // Left running, it discounts the previous match's rewards into this one's value target,
            // so after a few matches the critic is pretrained against a return inflated by everything
            // that ever happened before it.
            s.warmStartReturn = 0f;
        }
    }

    /// Credit a match result to a fighter's most recent transition. The per-turn reward is dense but
    /// says nothing about whether the match was actually won, so a sparse terminal bonus is added to
    /// the last transition of the match. This is what lets a policy learn to finish a fight rather
    /// than just trade evenly.
    /// </summary>
    public static void ApplyTerminalBonus(AIController owner, float bonus)
    {
        if (owner == null || bonus == 0f) return;
        if (!sessions.TryGetValue(owner.GetInstanceID(), out var s)) return;
        if (!IsLearning(s.role)) return;
        s.trainer.AdjustLastReward(bonus);
    }

    /// <summary>
    /// Flush every live policy to disk. Call before quitting a training run, otherwise the last
    /// batch of updates is lost. Frozen roles are skipped because their weights never change.
    /// </summary>
    public static void FlushAll()
    {
        foreach (var s in sessions.Values)
        {
            if (!IsLearning(s.role)) continue;
            SaveWeights(s.net, s.role);
            Debug.Log($"Flushed {s.role} policy after {s.updates} updates ({s.turns} turns).");
        }
    }

    /// <summary>Per-role training counters, for a training run's progress reporting.</summary>
    public static string Stats()
    {
        var sb = new System.Text.StringBuilder();
        foreach (var s in sessions.Values)
        {
            sb.Append($"{s.role}: {s.turns} turns, {s.updates} updates, warmStart={s.warmStart}, ");
        }
        return sb.ToString();
    }

    /// <summary>Export a role's weights to an explicit path, for shipping a frozen enemy build.</summary>
    public static bool ExportRole(string role, string path)
    {
        foreach (var s in sessions.Values)
        {
            if (s.role != role) continue;
            try
            {
                File.WriteAllText(path, JsonUtility.ToJson(s.net.Export()));
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Could not export {role} policy: {e.Message}");
                return false;
            }
        }
        return false;
    }
}
