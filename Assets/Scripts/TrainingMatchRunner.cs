using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Headless self-play driver for training runs. Spawns <see cref="arenaCount"/> independent arenas,
/// lets both fighters in each play themselves, and when a match ends records the win/loss as a
/// terminal bonus so the policies actually learn to finish fights instead of just trading evenly.
///
/// Runs only under <c>-training</c>. In the editor it does nothing, so it cannot disturb play.
/// </summary>
public class TrainingMatchRunner : MonoBehaviour
{
    [Header("Arenas")]
    [Tooltip("Number of independent arenas to run in parallel inside this one process.")]
    [Range(1, 64)] public int arenaCount = 8;

    [Tooltip("Turns after which a match is abandoned and treated as a draw. Stops a stalemate from "
             + "running forever and gives the value function a real episode boundary.")]
    public int maxMatchTurns = 400;

    [Header("Reward")]
    [Tooltip("Terminal reward for winning a match, and the negation for losing.")]
    public float winBonus = 20f;
    public float lossPenalty = -20f;

    [Header("Reporting")]
    [Tooltip("Log a progress line this often, in matches. 1 logs every match.")]
    public int reportEvery = 1;

    [Header("Sweep overrides")]
    [Tooltip("Optional reward/PPO overrides as key=value pairs, so a parameter sweep can be run "
             + "without editing code. Read from the command line, e.g. "
             + "-training -reward.approach=0.8 -reward.stalemate=0.1 -ppo.learningRate=1e-4. "
             + "Unrecognised keys are logged and ignored rather than silently doing nothing.")]
    public string parameterOverrideArg = "";

    private readonly List<string> appliedOverrides = new List<string>();

    [Tooltip("Seconds between heartbeat lines. The heartbeat proves the turn loop is advancing even "
             + "when no match finishes, which is how a deadlock gets told apart from slow progress.")]
    public float heartbeatSeconds = 2f;

    [Tooltip("Deactivate the scene's own Player/Enemy/Companion during a training run. They would "
             + "otherwise keep fighting in a second, scene-level arena and muddy the log.")]
    public bool deactivateSceneFighters = true;

    /// <summary>
    /// One arena's running match state.
    ///
    /// Shaped for 2v1 rather than 1v1: the win is a TEAM outcome, so there is no "winner index" and
    /// no per-slot win counter. <see cref="allies"/> holds the two Player-team fighters and
    /// <see cref="enemy"/> the single Enemy-team fighter. Ally order is resolved by team rather than
    /// by index, because arena.FightersInArena() returns registration order and attributing a
    /// reward to the wrong ally would be silently wrong rather than obviously broken.
    /// </summary>
    private class ArenaRun
    {
        public int index;
        public Arena arena;
        public List<CharacterController> allies = new List<CharacterController>();
        public CharacterController enemy;
        public int turns;
        public int wins;
        public int losses;
        public int draws;

        /// <summary>
        /// Which role this arena trains. Derived from the arena index, never from a shared match counter.
        ///
        /// Per-arena rather than a global odd/even tally because arenas finish matches at different
        /// times: a single counter would make "which role was learning here" depend on the order arenas
        /// happened to complete in, and arenas could drift so that one role held every learner slot.
        /// Index parity splits evenly by construction and survives any completion order.
        /// </summary>
        public bool trainsCompanion;

        /// <summary>What the ally team was playing this match. All three fighters see the same value.</summary>
        public BehaviourSource allySource = BehaviourSource.Live;

        /// <summary>What the enemy was playing this match.</summary>
        public BehaviourSource enemySource = BehaviourSource.Live;

        /// <summary>
        /// True when this match measured rather than trained. Nothing recorded, so nothing was learned
        /// from it; its only job is to produce an uncontaminated win rate for the curriculum's gate.
        /// </summary>
        public bool isEval;

        /// <summary>Which role the eval slice was measuring, and against which fixed opponent.</summary>
        public string evalRole;
        public BehaviourSource evalOpponent = BehaviourSource.Rule;

        /// <summary>
        /// Simulated frames the last turn in this match consumed. The single most useful number for
        /// spotting a stalled turn system: a healthy turn finishes in tens of frames, while anything
        /// sitting at TurnManager.maxTurnFrames means some fighter has no completion path.
        /// </summary>
        public int framesLastMatch;

        /// <summary>Health of each participant when the current match started, for damage credit.</summary>
        public readonly Dictionary<CharacterController, float> startHealth =
            new Dictionary<CharacterController, float>();

        public IEnumerable<CharacterController> Everyone()
        {
            foreach (var a in allies) if (a != null) yield return a;
            if (enemy != null) yield return enemy;
        }
    }

    private readonly List<ArenaRun> runs = new List<ArenaRun>();

    /// <summary>Arenas built but not yet wired up (still waiting for their fighters to Start).</summary>
    private readonly List<ArenaRun> pending = new List<ArenaRun>();

    private int totalMatches;

    /// <summary>Total matches started across all arenas. Drives the evaluation slice cadence.</summary>
    private int evalMatches;

    /// <summary>Rotates which (role x opponent) pair the next eval match measures.</summary>
    private int evalSlot;

    /// <summary>
    /// Per-(learner role x opponent x train/eval) W/L/D, so a win rate can be read against a SPECIFIC
    /// opponent instead of only in aggregate.
    ///
    /// A single win rate mixes random, rule and live opponents, which is exactly the ambiguity the
    /// curriculum exists to remove - a policy can look flat overall while improving against rule and
    /// regressing against live. Keyed by the learner's role and the source of the side it faced.
    /// </summary>
    private class Wl { public int wins, losses, draws; }
    private readonly Dictionary<string, Wl> wlByOpponent = new Dictionary<string, Wl>();

    private void Start()
    {
        if (!TrainingMode.enabled)
        {
            // Loud on purpose. A silent no-op here looks exactly like "the training code is broken"
            // when in fact training was never switched on.
            Debug.LogWarning(
                "[Training] Not starting: TrainingMode is off. Launch with the -training command-line "
                + "switch, or use Tools > Training > Start Training In Editor.");
            return;
        }
        Debug.Log($"[Training] Starting. fast-forward timestep {TrainingMode.secondsPerFrame}s/frame.");
        runStartTime = Time.realtimeSinceStartup;
        ApplyParameterOverrides();
        StartCoroutine(RunTraining());
    }

    /// <summary>
    /// Apply <c>-key=value</c> overrides from the command line to the reward and PPO weights.
    ///
    /// Written so a parameter sweep does not need a recompile per data point, which is the whole point
    /// of having telemetry to graph. Every applied value is echoed, because a silently-ignored override
    /// would make a sweep look like it tested something it did not.
    /// </summary>
    private void ApplyParameterOverrides()
    {
        var args = Environment.GetCommandLineArgs();
        int applied = 0;

        // -arenas=N is handled separately from the numeric overrides below, because it is an INT applied
        // to this component rather than a weight applied to the learner.
        foreach (var raw in args)
        {
            if (string.IsNullOrEmpty(raw) || !raw.StartsWith("-arenas=")) continue;
            if (int.TryParse(raw.Substring(8), out int n) && n > 0)
            {
                arenaCount = Mathf.Clamp(n, 1, 64);
                Debug.Log($"[Training] arena count set to {arenaCount} from the command line.");
                applied++;
                // The rendered camera only makes sense for one arena; extra arenas share the viewport.
                var builder = GetComponent<TrainingArenaBuilder>();
                if (builder != null && arenaCount > 1 && builder.renderFighters)
                    Debug.Log("[Training] note: only arena " + builder.renderArenaIndex
                        + " renders; the rest simulate headlessly in the same process.");
            }
        }

        foreach (var raw in args)
        {
            if (string.IsNullOrEmpty(raw) || raw[0] != '-') continue;
            int eq = raw.IndexOf('=');
            if (eq <= 1) continue;
            string key = raw.Substring(1, eq - 1);
            string value = raw.Substring(eq + 1);
            if (key == "arenas") continue;   // handled above

            if (!float.TryParse(value, System.Globalization.NumberStyles.Float,
                                 System.Globalization.CultureInfo.InvariantCulture, out float f))
            {
                Debug.LogWarning($"[Training] override '{raw}' is not a number; ignored.");
                continue;
            }

            if (key.StartsWith("reward."))
            {
                if (!TrySetReward(key.Substring("reward.".Length), f)) continue;
            }
            else if (key.StartsWith("ppo."))
            {
                PolicyLearner.TrySetHyper(key.Substring("ppo.".Length), f);
            }
            else if (key.StartsWith("mix."))
            {
                if (!PolicyLearner.TrySetMix(key.Substring("mix.".Length), f)) continue;
            }
            else
            {
                continue;   // some other -flag, not ours
            }

            appliedOverrides.Add($"{key}={value}");
            Debug.Log($"[Training] override applied: {key} = {value}");
            applied++;
        }
        if (applied > 0) Debug.Log($"[Training] {applied} parameter override(s) in effect.");
    }

    /// <summary>Set one RewardConfig field by name. Returns false for an unknown field.</summary>
    private static bool TrySetReward(string field, float value)
    {
        var cfg = PolicyLearner.Config;
        switch (field)
        {
            case "hitLanded": cfg.hitLanded = value; return true;
            case "damageDealt": cfg.damageDealt = value; return true;
            case "damageTaken": cfg.damageTaken = value; return true;
            case "killBonus": cfg.killBonus = value; return true;
            case "approach": cfg.approach = value; return true;
            case "stalemate": cfg.stalemate = value; return true;
            case "voteGood": cfg.voteGood = value; return true;
            case "voteBad": cfg.voteBad = value; return true;
            default:
                Debug.LogWarning($"[Training] unknown reward field '{field}'; ignored.");
                return false;
        }
    }

    /// <summary>
    /// Two phases, with a frame boundary between them. This ordering is load-bearing.
    ///
    /// <c>AddComponent</c> runs <c>Awake</c> immediately, but Unity calls <c>Start</c> on a
    /// runtime-created object just before the NEXT frame's update, and <c>Update</c> before
    /// <c>Start</c>. A fighter only registers itself with its arena's TurnManager from
    /// <c>CharacterController.Start()</c>, so creating the fighters and checking registration in the
    /// same phase always sees 0/2. Yielding between build and wire is what makes it work.
    /// </summary>
    private System.Collections.IEnumerator RunTraining()
    {
        var builder = GetComponent<TrainingArenaBuilder>();
        if (builder == null)
        {
            Debug.LogError("TrainingMatchRunner needs a TrainingArenaBuilder on the same GameObject.");
            yield break;
        }

        DeactivateSceneFighters(builder);
        BuildArenas(builder);

        // Let every fighter's Awake/Start complete so they have registered with their TurnManager.
        yield return null;

        WireArenas();

        // After WireArenas, not before: the bars can only be pointed at fighters that already exist, and
        // the arena clones are created by BuildArenas above.
        RebindHudToArenaFighters();
    }

    /// <summary>Phase 1: create the arena roots, their managers, the stage and the two fighters.</summary>
    private void BuildArenas(TrainingArenaBuilder builder)
    {
        for (int i = 0; i < arenaCount; i++)
        {
            var arena = builder.Build(i);
            if (arena == null) continue;
            pending.Add(new ArenaRun { arena = arena, index = i });
        }
        Debug.Log($"[Training] Built {pending.Count} arena(s); waiting a frame for fighters to register.");
    }

    /// <summary>Phase 2: resolve fighters, verify registration, and subscribe to match completion.</summary>
    private void WireArenas()
    {
        foreach (var run in pending)
        {
            // Resolve by TEAM, not by index. FightersInArena() is registration-ordered, so index 0
            // was never guaranteed to be the player; with two allies on one team an index-based split
            // would also have put both allies on the same side of the reward.
            foreach (var f in run.arena.FightersInArena())
            {
                if (f == null) continue;
                if (f.Team == CombatTeam.Player) run.allies.Add(f);
                else if (f.Team == CombatTeam.Enemy) run.enemy = f;
            }

            if (run.allies.Count != 2 || run.enemy == null)
            {
                Debug.LogError($"Arena {run.index} did not come out 2v1 "
                    + $"({run.allies.Count} allies, {(run.enemy == null ? 0 : 1)} enemy); skipping.");
                continue;
            }

            // Every fighter must have registered with its arena's TurnManager by now. If not, the turn
            // loop will silently idle, so say so loudly rather than appearing to train nothing.
            run.arena.TurnManager.TurnResolved += () => run.framesLastMatch = run.arena.TurnManager.FramesLastTurn;
            int registered = run.arena.TurnManager.GetAllPlayers().Count;
            if (registered < 3)
            {
                Debug.LogError($"Arena {run.index}: only {registered}/3 fighters registered with the turn "
                    + "manager. Training cannot run in this arena.");
                continue;
            }

            SnapshotMatchStart(run);
            runs.Add(run);
            // Opponents are assigned per arena rather than drawn from a shared counter, so this happens
            // once and stays fixed for the arena's whole life.
            run.trainsCompanion = (run.index % 2) == 0;
            AssignBehaviour(run);
            run.arena.TurnManager.TurnResolved += () => OnMatchEnd(run);
        }
        pending.Clear();

        if (runs.Count == 0)
        {
            Debug.LogError("[Training] No arenas were built. Check that TrainingArenaBuilder is on "
                + "the same GameObject and that fighterPrefab (if set) carries an AIController.");
            return;
        }
        int companionArenas = 0, enemyArenas = 0;
        foreach (var r in runs)
        {
            if (r.trainsCompanion) companionArenas++; else enemyArenas++;
        }
        Debug.Log($"TrainingMatchRunner: {runs.Count} arenas, up to {arenaCount} matches in flight "
            + $"(2v1 each). Learner split by arena index parity: {companionArenas} train the companion, "
            + $"{enemyArenas} train the enemy."
            + (TrainingMode.curriculum
                ? $" Curriculum on, evaluation slice 1 in {TrainingMode.evalEvery} matches."
                : " Curriculum OFF (-noCurriculum)."));
        // Logged once, not per arena. The stage is invisible in every number the run reports, so if it
        // ever drifts from the shipped scene again this line is the only thing that would say so.
        // Resolved rather than passed in: WireArenas is a separate phase from RunTraining, and the builder
        // is the only thing that knows the stage geometry.
        var stageBuilder = GetComponent<TrainingArenaBuilder>();
        if (stageBuilder != null) Debug.Log($"[Training] stage: {stageBuilder.StageSummary()}");
        // Stated on the banner because it changes every reward the run sees. A balance change that is
        // not in the log makes two segments' CSVs look like a policy regression.
        Debug.Log($"[Training] combat balance: {CombatBalance.Describe()}.");
    }

    /// <summary>
    /// Record every fighter's starting health and clear its combat counters, so damage can be
    /// attributed per fighter when the match ends.
    ///
    /// Health deltas rather than CombatStats.damageDealt for the damage credit itself: the enemy is
    /// hit by two allies at once, so "what did this ally contribute" is only answerable from how the
    /// ENEMY's health fell. CombatStats remains the source for attempts and hits, which have no
    /// health-drop equivalent.
    /// </summary>
    private static void SnapshotMatchStart(ArenaRun run)
    {
        run.startHealth.Clear();
        foreach (var f in run.Everyone())
        {
            run.startHealth[f] = f.GetHealth();
            f.CombatStats.Reset();
        }
    }

    /// <summary>
    /// Disable the scene's authored fighters so a training run only contains the arenas we built.
    ///
    /// Without this they keep running against each other in the scene's own TurnManager, which is the
    /// one that owns the static Instance, and their activity shows up in the same console as the
    /// training run.
    /// </summary>
    private void DeactivateSceneFighters(TrainingArenaBuilder builder)
    {
        if (!deactivateSceneFighters) return;
        foreach (var f in FindObjectsByType<CharacterController>(FindObjectsSortMode.None))
        {
            if (f.GetComponentInParent<Arena>() != null) continue; // one of ours

            // Stop the preview ghosts BEFORE deactivating. PreviewManager drives its registered previews
            // from Update() by calling Step() directly, so a deactivated fighter's preview is still
            // being stepped -- with a preview physics body that never got Initialize()d, which threw
            // a NullReferenceException every single frame.
            foreach (var pc in f.GetComponentsInChildren<PreviewController>(true))
                pc.StopPreview();

            f.gameObject.SetActive(false);
        }
        foreach (var cam in FindObjectsByType<CameraFollow>(FindObjectsSortMode.None))
            cam.enabled = false;
    }

    /// <summary>
    /// Point every HUD health bar at ONE arena's three fighters, matched by role.
    ///
    /// This is presentation only - it has no effect on damage, rewards or telemetry - but it was badly
    /// wrong, and in a way that read as a game bug rather than a UI bug.
    ///
    /// It used to infer each bar's team from <c>bar.name</c>, which is the HealthBar GameObject's own
    /// name: the inner fill rectangle, called "Inner". The actual bar containers are "Left Health",
    /// "Right Health" and "Companion Health". So the enemy bar never matched "Enemy" and fell through to
    /// the player branch, and both unbound bars were then handed the same deterministically-chosen ally.
    /// Two HUD bars were pointing at one fighter, which is why every health bar appeared to move in
    /// lockstep. (Compounding it, neither of those two bars even had an Owner set - they still use the
    /// older serialized characterData field - so the name test was the only thing that ran at all.)
    ///
    /// Matching by ROLE inside ONE arena fixes both that and a second fault: the companion bar used to
    /// resolve through TurnManager.Instance, i.e. the last arena created, so it displayed a different
    /// fight from the other two bars. One arena, three roles, three bars.
    ///
    /// The arena chosen is the one the training camera renders, so the HUD and the viewport always agree.
    /// </summary>
    private void RebindHudToArenaFighters()
    {
        var builder = GetComponent<TrainingArenaBuilder>();
        int arenaIndex = builder != null ? builder.renderArenaIndex : 0;

        CharacterController playerFighter = null, companionFighter = null, enemyFighter = null;

        foreach (var a in FindObjectsByType<Arena>(FindObjectsSortMode.None))
        {
            if (a == null || a.gameObject.name != $"TrainingArena_{arenaIndex}") continue;
            foreach (var f in a.GetComponentsInChildren<CharacterController>(true))
            {
                // TrainingArenaBuilder names its three fighters, which is what makes a per-slot
                // binding possible at all. Team cannot do it: BOTH allies are CombatTeam.Player in a
                // training arena, so a team-based mapping has nothing to distinguish them with.
                switch (f.name)
                {
                    case "Player": playerFighter = f; break;
                    case "Companion": companionFighter = f; break;
                    case "Enemy": enemyFighter = f; break;
                }
            }
            break;
        }

        if (playerFighter == null && enemyFighter == null)
        {
            Debug.LogWarning($"[Training] No arena named 'TrainingArena_{arenaIndex}' with the expected "
                + "Player/Companion/Enemy fighters; leaving the HUD bound to the scene's own bars.");
            return;
        }

        // The companion bar is bound through its owner component rather than by scanning, because it is
        // created at runtime and is the one bar that knows it is the companion bar.
        var companionUi = FindFirstObjectByType<CompanionHealthBarUI>();
        if (companionUi != null) companionUi.BindTo(companionFighter);

        var bars = FindObjectsByType<HealthBar>(FindObjectsSortMode.None);
        foreach (var bar in bars)
        {
            if (companionUi != null && bar == companionUi.Bar) continue;
            // Structural, not name-based: the enemy bar is anchored to the right edge of the canvas and
            // the player bar to the left. CompanionHealthBarUI already relies on this same convention to
            // find the bar it clones, so it is the established discriminator in this HUD.
            var target = IsRightAnchored(bar) ? enemyFighter : playerFighter;
            if (target != null) bar.Bind(target);
        }

        Debug.Log($"[Training] HUD bound to arena {arenaIndex}: "
            + $"player={playerFighter?.name ?? "none"}, "
            + $"companion={companionFighter?.name ?? "none"}, "
            + $"enemy={enemyFighter?.name ?? "none"}.");
    }

    /// <summary>
    /// True when a bar hangs off the right edge of the canvas, i.e. it is the enemy bar.
    /// </summary>
    private static bool IsRightAnchored(HealthBar bar)
    {
        var rect = bar.transform as RectTransform;
        if (rect == null) return false;
        return rect.anchorMin.x > 0.5f || rect.anchorMax.x > 0.5f;
    }

    /// <summary>
    /// Read-only views for the telemetry writer, rebuilt on demand rather than cached, so a handle
    /// can never go stale relative to the run it describes.
    /// </summary>
    private List<TrainingTelemetry.ArenaRunHandle> handles
    {
        get
        {
            var list = new List<TrainingTelemetry.ArenaRunHandle>(runs.Count);
            foreach (var r in runs)
                list.Add(new TrainingTelemetry.ArenaRunHandle(r.index, r.turns, r.framesLastMatch,
                                                             () => r.Everyone()));
            return list;
        }
    }

    private void OnMatchEnd(ArenaRun run)
    {
        var enemy = run.enemy;
        bool enemyDead = enemy == null || enemy.IsDead();

        // Team wipe is the loss condition, NOT "one ally died". With two allies, losing one is
        // survivable, and treating it as a loss would train both of them to play so cautiously that
        // neither ever takes the risk of setting up the other's combo.
        int alliesAlive = 0;
        foreach (var a in run.allies)
            if (a != null && !a.IsDead()) alliesAlive++;

        if (!enemyDead && alliesAlive > 0)
        {
            run.turns++;
            // Abandoned match: nobody won, so neither side gets a bonus. Scored as a draw rather
            // than a forced loss, otherwise the policy is pushed toward whatever ends matches fast.
            if (run.turns >= maxMatchTurns) FinishMatch(run, MatchResult.Draw);
            return;
        }

        if (enemyDead && alliesAlive == 0) FinishMatch(run, MatchResult.Draw);   // mutual wipe
        else if (enemyDead) FinishMatch(run, MatchResult.AlliesWin);
        else FinishMatch(run, MatchResult.AlliesLose);
    }

    private enum MatchResult { AlliesWin, AlliesLose, Draw }

    /// <summary>
    /// Resolve the match: award the terminal bonus, reset the arena, and log progress.
    /// <paramref name="winnerIndex"/> is 0 for A, 1 for B, or -1 for a draw / abandoned match, which
    /// awards nothing to either side.
    /// </summary>
    private void FinishMatch(ArenaRun run, MatchResult result)
    {
        // Credit each ally with its share of the damage it actually contributed to the enemy, so the
        // win bonus can be divided by contribution. Without this both allies get the full bonus and
        // one of them learns that idling in a corner is free, because the pair still wins.
        float totalAllyDamage = 0f;
        foreach (var a in run.allies)
            if (a != null && run.startHealth.TryGetValue(run.enemy, out float e0))
                totalAllyDamage += Mathf.Max(0f, e0 - run.enemy.GetHealth());

        if (result == MatchResult.AlliesWin)
        {
            foreach (var a in run.allies)
            {
                var ai = a as AIController;
                if (ai == null || a.IsDead()) continue;
                float share = 1f;
                if (totalAllyDamage > 0.01f && run.startHealth.TryGetValue(run.enemy, out float e0))
                    share = Mathf.Max(0f, e0 - run.enemy.GetHealth()) / totalAllyDamage;
                PolicyLearner.ApplyTerminalBonus(ai, winBonus * share);
            }
            PolicyLearner.ApplyTerminalBonus(run.enemy as AIController, lossPenalty);
            run.wins++;
        }
        else if (result == MatchResult.AlliesLose)
        {
            // A loss is charged in full to every survivor: an ally that was still standing when the
            // team was wiped down shares the blame, and splitting it by damage here would reward
            // having done nothing.
            foreach (var a in run.allies)
            {
                var ai = a as AIController;
                if (ai == null) continue;
                PolicyLearner.ApplyTerminalBonus(ai, a.IsDead() ? lossPenalty * 0.5f : lossPenalty);
            }
            PolicyLearner.ApplyTerminalBonus(run.enemy as AIController, winBonus);
            run.losses++;
        }
        else
        {
            run.draws++;
        }

        RecordLearnerResult(run, result);

        totalMatches++;
        evalMatches++;
        RecordTelemetry(run, result);
        TickCurricula(run, result);

        if (reportEvery <= 1 || totalMatches % reportEvery == 0)
        {
            Report();
            LogLearnerHealth();
        }
        // Checked here too, not only in the heartbeat: a run bounded by matches should stop the moment
        // it reaches the limit, and the heartbeat is up to heartbeatSeconds late.
        CheckRunLimits();

        run.turns = 0;
        // ResetState reloads each fighter's start-of-match SaveData, so health, position and combo
        // counters all return to the opening state for the next match.
        run.arena.TurnManager.ResetState();
        // Stamp a new episode onto every brain so GAE does not chain the final states of this match into
        // the opening states of the next.
        //
        // This call was MISSING entirely before the curriculum work. Without it trainer.BeginEpisode()
        // never ran, CurrentEpisode stayed 0 for the whole process, and every transition carried
        // episode 0 - so the (agentId, episode) grouping that keeps the two allies' trajectories apart
        // was silently doing nothing, and GAE discounted each fighter's last turn toward the OTHER
        // fighter's first turn of the following match. It is exactly the cross-fighter bootstrapping
        // bug that grouping exists to prevent, just across a match boundary instead of a turn one.
        PolicyLearner.NotifyMatchReset();
        // Re-snapshot: health was just restored, and the next match's damage credit is measured
        // against this baseline rather than the previous match's starting health.
        SnapshotMatchStart(run);
        // Next match's opponents. Assigned AFTER the reset so the new sources are in place for the very
        // first turn of the next match rather than one match late.
        AssignBehaviour(run);
    }

    /// <summary>
    /// Heartbeat. Proves the loop is advancing independently of match completion, which is the only
    /// way to tell "training is slow" apart from "training is deadlocked".
    /// </summary>
    private void Update()
    {
        if (!TrainingMode.enabled || runs.Count == 0) return;
        heartbeatTimer += Time.unscaledDeltaTime;
        if (heartbeatTimer < heartbeatSeconds) return;
        heartbeatTimer = 0f;
        CheckRunLimits();

        var run = runs[0];
        string state = "";
        foreach (var f in run.Everyone())
        {
            if (f == null) { state += " [missing]"; continue; }
            var ai = f as AIController;
            string why = ai != null && ai.CurrentDecision.rationale != null
                ? $" why='{ai.CurrentDecision.rationale}'"
                : "";
            state += $" [{f.name} hp={f.GetHealth():0} pos={f.GetPosition()} state={f.State} "
                   + $"move='{f.SelectedMove}'{why}]";
        }

        // Throughput, on every heartbeat. Previously the only per-turn counter in the log had no
        // timestamp, so "how fast is this actually running" could not be answered from the data at all -
        // which is the wrong thing to be unable to answer when deciding whether to add arenas.
        //
        // Turns and simulated frames are both reported, and they are different questions. A turn is one
        // round of all three fighters; a frame is one simulation tick. Turns per second says how fast
        // matches complete, frames per second says how much simulation is actually happening - and a
        // policy that stalls makes frames-per-turn balloon, which moves the second number without moving
        // the first.
        float elapsed = Time.realtimeSinceStartup - runStartTime;
        // Both throughput rates must come from MONOTONIC counters, differenced over the sample span.
        //
        // The previous version summed run.turns, which is the CURRENT match's turn count and is reset
        // to 0 on every FinishMatch. Dividing that by TOTAL elapsed time produced a number with no
        // units: it read 0.04-1.45 turns/s while the run was actually completing ~75, and
        // avgFramesPerTurn then divided a cumulative frame counter by it, manufacturing the impossible
        // 5,000-100,000 frames/turn readings. The CSV's per-turn frame count was always 26-36.
        long simFramesTotal = 0, turnsTotal = 0;
        foreach (var r in runs)
        {
            simFramesTotal += r.arena.TurnManager.TotalSimFrames;
            turnsTotal += r.arena.TurnManager.TotalTurns;
        }
        float turnsPerSec = 0f, framesPerSec = 0f;
        float sampleSpan = elapsed - lastSampleElapsed;
        if (sampleSpan > 0.01f)
        {
            turnsPerSec = (float)(turnsTotal - turnsAtLastSample) / sampleSpan;
            framesPerSec = (float)(simFramesTotal - simFramesAtLastSample) / sampleSpan;
        }
        simFramesAtLastSample = simFramesTotal;
        turnsAtLastSample = turnsTotal;
        lastSampleElapsed = elapsed;
        // Now meaningful: both sides are cumulative, so this is the true average simulation frames a
        // turn costs. It is the number that distinguishes a stalled turn (pinned at maxTurnFrames)
        // from a genuinely cheap one.
        avgFramesPerTurn = turnsTotal > 0 ? (float)simFramesTotal / turnsTotal : 0f;

        // Where the wall-clock actually goes. turns/s alone cannot say whether the bottleneck is the
        // physics step, the fighter state machines, or the PPO update - and each needs a different fix.
        // Timing only OBSERVES; it feeds no RNG and no branch of the sim, so the run stays deterministic.
        double simSec = 0, fighterSec = 0, hitboxSec = 0;
        foreach (var r in runs)
        {
            simSec += r.arena.TurnManager.SimSecondsTotal;
            fighterSec += r.arena.TurnManager.FighterStepSecondsTotal;
            hitboxSec += r.arena.TurnManager.HitboxStepSecondsTotal;
        }
        double ppoSec = 0; int ppoCalls = 0;
        foreach (var kv in PolicyLearner.BrainStats())
        {
            ppoSec += kv.Value.updateSeconds;
            ppoCalls += kv.Value.updateCalls;
        }
        double simMsPerTurn = turnsTotal > 0 ? simSec / turnsTotal * 1000.0 : 0.0;
        double fighterMsPerTurn = turnsTotal > 0 ? fighterSec / turnsTotal * 1000.0 : 0.0;
        double ppoMsPerUpdate = ppoCalls > 0 ? ppoSec / ppoCalls * 1000.0 : 0.0;
        double ppoShare = simSec + ppoSec > 0.0 ? ppoSec / (simSec + ppoSec) : 0.0;

        // avgFramesPerTurn separates "few turns, each expensive" from "many turns, each cheap".
        // Without it a low turns/s is ambiguous: the turn count alone cannot distinguish a policy that
        // stalls (turns hit the frame cap) from a genuinely slow simulation, and those need opposite fixes.
        Debug.Log($"[Training] heartbeat: {totalMatches} matches done, arena0 at turn {run.turns}, "
            + $"phase={run.arena.TurnManager.Phase} "
            + $"[{elapsed:0}s {turnsPerSec:0.00} turns/s {framesPerSec:0} simframes/s "
            + $"{avgFramesPerTurn:0.0} frames/turn "
            + $"sim {simMsPerTurn:0.00}ms/turn (fighter {fighterMsPerTurn:0.00}, hitbox {hitboxSec / Mathf.Max(1f, (float)turnsTotal) * 1000.0:0.00}) "
            + $"ppo {ppoMsPerUpdate:0.0}ms/update {ppoShare:P0} "
            + $"({runs.Count} arenas, {appliedOverrides.Count} overrides)]{state}");
    }

    private float heartbeatTimer;
    private float runStartTime;

    // Baseline for the differenced throughput rate. Storing the previous counter value and elapsed time,
    // rather than dividing by total elapsed, keeps the reported rate describing the CURRENT rate instead
    // of a running average that can only ever lag.
    private long simFramesAtLastSample;
    private long turnsAtLastSample;
    private float lastSampleElapsed;
    private float avgFramesPerTurn;

    /// <summary>
    /// Stop a bounded run once it has hit <c>-maxMatches</c> or <c>-maxSeconds</c>.
    ///
    /// Both the policies and the telemetry are flushed BEFORE quitting, because OnApplicationQuit is
    /// not guaranteed to fire under -batchmode and a run that ended without writing its weights or its
    /// CSV would leave nothing to analyse.
    /// </summary>
    private void CheckRunLimits()
    {
        if (!TrainingMode.enabled) return;
        bool hitMatches = TrainingMode.maxMatches > 0 && totalMatches >= TrainingMode.maxMatches;
        bool hitSeconds = TrainingMode.maxSeconds > 0
                          && Time.realtimeSinceStartup - runStartTime >= TrainingMode.maxSeconds;
        if (!hitMatches && !hitSeconds) return;

        Debug.Log($"[Training] run limit reached ({(hitMatches ? "matches" : "time")}): "
            + $"{totalMatches} matches in {Time.realtimeSinceStartup - runStartTime:0}s. Flushing.");
        PolicyLearner.FlushAll();
        TrainingTelemetry.End();
        LogLearnerHealth();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    /// <summary>
    /// Open the telemetry CSV from Start, so a run that never builds an arena still leaves a file
    /// explaining why rather than nothing at all.
    /// </summary>
    private void OnEnable() => TrainingTelemetry.Begin();

    /// <summary>
    /// Write this match's rows to the telemetry CSV, one per fighter.
    ///
    /// Damage credit is computed here rather than inside the telemetry writer because it needs the
    /// match's start-of-match health snapshot, which lives on the run.
    /// </summary>
    private void RecordTelemetry(ArenaRun run, MatchResult result)
    {
        var credit = new Dictionary<CharacterController, float>();
        float enemyStart = run.startHealth.TryGetValue(run.enemy, out var e0) ? e0 : 0f;
        float enemyLost = run.enemy != null ? Mathf.Max(0f, enemyStart - run.enemy.GetHealth()) : 0f;

        // Attribute the enemy's missing health to the allies that were still standing. A dead ally
        // cannot have contributed to the final blow, and giving it credit for the whole total would
        // let a policy farm the shared bonus by trading itself off early.
        float credited = 0f;
        foreach (var a in run.allies)
        {
            if (a == null) continue;
            float share = 0f;
            if (!a.IsDead() && enemyLost > 0f) share = enemyLost * (a.CombatStats.damageDealt / Mathf.Max(0.01f, TotalAllyDamage(run, enemyLost)));
            credit[a] = share;
            credited += share;
        }
        if (run.enemy != null) credit[run.enemy] = credited;

        var handle = new TrainingTelemetry.ArenaRunHandle(
            run.index, run.turns, run.framesLastMatch, () => run.Everyone());
        TrainingTelemetry.RecordMatch(handle, result.ToString(), totalMatches, credit,
                                   run.allySource, run.enemySource, run.isEval);
    }

    /// <summary>
    /// Total damage the allies landed on the enemy this match, used as the denominator when splitting
    /// the enemy's missing health between them. Guarded against a zero denominator so a match that
    /// ended without any connecting hit cannot divide by zero.
    /// </summary>
    private static float TotalAllyDamage(ArenaRun run, float enemyLost)
    {
        float total = 0f;
        foreach (var a in run.allies)
            if (a != null) total += a.CombatStats.damageDealt;
        return Mathf.Max(total, enemyLost * 0.0001f, 0.0001f);
    }

    /// <summary>
    /// Decide who each side plays in this arena's next match, and push the decision onto the fighters.
    ///
    /// Two independent things are decided here, and conflating them is the mistake worth avoiding:
    ///
    ///   1. WHICH SIDE LEARNS. Fixed per arena by index parity, not drawn. A random or alternating
    ///      choice per match would make the two roles' throughput depend on match length and completion
    ///      order, and the shorter side's brain would end up starved for reasons unrelated to learning.
    ///
    ///   2. WHAT EACH SIDE FACES. Drawn from that role's OWN curriculum, so the opponent difficulty a
    ///      role trains against is the difficulty it has actually earned its way up to, rather than the
    ///      same global schedule applied to both roles.
    ///
    /// The learner side is forced to <see cref="BehaviourSource.Live"/> and never to the curriculum
    /// draw: a role that was learning against random and then met the live opponent for one match would
    /// have that match's samples attributed to a distribution it almost never samples from.
    /// </summary>
    private void AssignBehaviour(ArenaRun run)
    {
        // The learner side always plays itself; the opponent side plays whatever the curriculum says.
        // The opponent is drawn from the curriculum of the role being TRAINED, not from the role
        // playing the opponent. These two calls were cross-wired: the companion trained against
        // opponents from the ENEMY's curriculum and the enemy against opponents from the
        // COMPANION's. The two curricula agree only while both sit in phase 0, so the swap is
        // invisible early and then diverges sharply once one role advances. Measured from a run's CSV:
        // companion-learner opponents were random 508 / rule 64 (the enemy's random-then-rule mix)
        // while enemy-learner opponents were 100% random (the companion's stuck-at-phase-0 mix). The
        // consequence was the companion being fed rule-based opponents its own curriculum had not yet
        // promoted it to, so its eval-vs-rule stayed low, its curriculum could never advance, and its
        // critic sat near zero while the enemy trained on pure random and raced ahead.
        run.allySource = run.trainsCompanion
            ? BehaviourSource.Live
            : (TrainingMode.curriculum ? PolicyLearner.CurriculumFor(PolicyLearner.RoleEnemy).Draw()
                                       : BehaviourSource.Live);
        run.enemySource = run.trainsCompanion
            ? (TrainingMode.curriculum ? PolicyLearner.CurriculumFor(PolicyLearner.RoleCompanion).Draw()
                                       : BehaviourSource.Live)
            : BehaviourSource.Live;

        // The evaluation slice. Every Nth match, neither side records and the opponents are fixed, which
        // is the only way to get a win rate that is not measured on data the policy trained on.
        //
        // The evalMatches > 0 guard matters only at startup: AssignBehaviour is called once per arena
        // during wiring, so with a bare modulo every arena would see 0 % evalEvery == 0 and ALL of them
        // would open with an evaluation match - burning one match per arena and skewing the first four
        // slots of the rotation. Requiring one completed match first also makes the slice fire from a
        // single arena: evalMatches increments once per completion, so exactly the arena that takes the
        // run past the multiple gets the evaluation match and its peers do not.
        run.isEval = TrainingMode.curriculum && TrainingMode.evalEvery > 0 && evalMatches > 0
                     && evalMatches % TrainingMode.evalEvery == 0;

        if (run.isEval)
        {
            // Rotate over (role x opponent) so all four curves accumulate at the same rate. Measuring
            // only the companion's curve would let the enemy's curriculum promote on no evidence at all.
            evalSlot++;
            bool evalCompanion = (evalSlot / 2) % 2 == 0;
            run.evalRole = evalCompanion ? PolicyLearner.RoleCompanion : PolicyLearner.RoleEnemy;
            run.evalOpponent = (evalSlot % 2) == 0 ? BehaviourSource.Random : BehaviourSource.Rule;

            if (evalCompanion)
            {
                run.allySource = BehaviourSource.LiveNoLearn;   // acts for real, records nothing
                run.enemySource = run.evalOpponent;
            }
            else
            {
                run.enemySource = BehaviourSource.LiveNoLearn;
                run.allySource = run.evalOpponent;
            }
        }
        else
        {
            run.evalRole = null;
        }

        foreach (var a in run.allies)
            if (a is AIController ai) ai.SetBehaviourSource(run.allySource);
        if (run.enemy is AIController enemyAi) enemyAi.SetBehaviourSource(run.enemySource);

        // New match begins here, so reset each fighter's terminal-bonus attribution. This is called at
        // wiring and again after every FinishMatch (post-bonus), which makes it the exact per-match
        // boundary. Done per fighter rather than as a global sweep because arenas share a role brain.
        foreach (var a in run.allies)
            if (a is AIController ai) PolicyLearner.NotifyMatchStart(ai);
        if (run.enemy is AIController eai) PolicyLearner.NotifyMatchStart(eai);
    }

    /// <summary>
    /// Feed one finished match's outcome into the curriculum's promotion and forgetting gates.
    ///
    /// Regress is checked before advance on purpose. Both read the same evaluation windows, so a run of
    /// unlucky matches can satisfy the advance threshold in the short window while the long window is
    /// still below the regress threshold; promoting on that reading would move the curriculum to a
    /// harder opponent at the exact moment the policy looks like it is regressing.
    /// </summary>
    private void TickCurricula(ArenaRun run, MatchResult result)
    {
        if (!TrainingMode.curriculum || !run.isEval || run.evalRole == null) return;

        bool alliesWon = result == MatchResult.AlliesWin;
        bool alliesLost = result == MatchResult.AlliesLose;

        // Draws carry no signal about whether the policy is improving, so they are dropped rather than
        // counted as losses - a stalemate-heavy run would otherwise be demoted for being inconclusive.
        bool won = alliesWon || alliesLost;
        if (!won) return;

        var curriculum = PolicyLearner.CurriculumFor(run.evalRole);
        if (curriculum == null) return;

        bool roleWon = run.evalRole == PolicyLearner.RoleCompanion ? alliesWon : alliesLost;
        curriculum.RecordEval(run.evalOpponent, roleWon);

        if (curriculum.TickRegress())
        {
            Debug.Log($"[Training] Curriculum {run.evalRole}: eval win rate vs rule has fallen below the "
                      + $"regress threshold; stepping back to {curriculum.DescribeShares()}.");
            return;
        }
        if (curriculum.TickAdvance())
            Debug.Log($"[Training] Curriculum {run.evalRole}: promoted to {curriculum.DescribeShares()}.");
    }

    /// <summary>Record the learner's result against the opponent it actually faced this match.</summary>
    private void RecordLearnerResult(ArenaRun run, MatchResult result)
    {
        string role;
        BehaviourSource opponent;
        if (run.isEval && run.evalRole != null)
        {
            role = run.evalRole;
            opponent = run.evalOpponent;
        }
        else
        {
            role = run.trainsCompanion ? PolicyLearner.RoleCompanion : PolicyLearner.RoleEnemy;
            opponent = run.trainsCompanion ? run.enemySource : run.allySource;
        }

        bool won = role == PolicyLearner.RoleCompanion ? result == MatchResult.AlliesWin
                                                       : result == MatchResult.AlliesLose;
        bool lost = role == PolicyLearner.RoleCompanion ? result == MatchResult.AlliesLose
                                                        : result == MatchResult.AlliesWin;

        string key = $"{role} vs {opponent.Token()}{(run.isEval ? " (eval)" : "")}";
        if (!wlByOpponent.TryGetValue(key, out var wl)) wlByOpponent[key] = wl = new Wl();
        if (won) wl.wins++;
        else if (lost) wl.losses++;
        else wl.draws++;
    }

    private void Report()
    {
        int wins = 0, losses = 0, draws = 0;
        foreach (var r in runs)
        {
            wins += r.wins;
            losses += r.losses;
            draws += r.draws;
        }
        float total = wins + losses;
        float rate = total > 0 ? 100f * wins / total : 0f;
        Debug.Log($"Training: {totalMatches} matches. Ally win rate {rate:0.0}% "
            + $"({wins}W/{losses}L, {draws} draws).");
        Debug.Log($"Training policies: {PolicyLearner.Stats()}");
        foreach (var kv in wlByOpponent)
        {
            var wl = kv.Value;
            float games = wl.wins + wl.losses;
            float wr = games > 0f ? 100f * wl.wins / games : 0f;
            Debug.Log($"[Training] win rate {kv.Key}: {wr:0.0}% "
                + $"({wl.wins}W/{wl.losses}L, {wl.draws} draws).");
        }
        TrainingTelemetry.LogSnapshot(handles, totalMatches);
    }

    /// <summary>
    /// The learner-health line.
    ///
    /// These are the numbers that say whether training is working, separated out from the win rate
    /// because the win rate alone cannot distinguish "learning" from "getting lucky". Specifically:
    /// explained variance below 0 means the critic is worse than a constant and every advantage is
    /// noise; clip fraction near 0 or near 1 means the step size is wrong; entropy near 0 means the
    /// policy has stopped exploring.
    /// </summary>
    private void LogLearnerHealth()
    {
        var lines = new List<string>();
        foreach (var kv in PolicyLearner.BrainStats())
        {
            lines.Add(
                $"{kv.Key}: updates={kv.Value.updates} buffered={kv.Value.buffered} " +
                $"lr={kv.Value.learningRate:0.###e+0} " +
                $"EV={kv.Value.explainedVariance:0.###} kl={kv.Value.approxKl:0.####} " +
                $"clip={kv.Value.clipFraction:0.###} vloss={kv.Value.valueLoss:0.###} " +
                $"|A|={kv.Value.advantageMagnitude:0.###} rejected={kv.Value.rejected} "
                + $"nonfinite={kv.Value.nonfinite} clipped={kv.Value.clippedSteps}/{kv.Value.clippedCriticSteps} vStd={kv.Value.valueStd:F2} rStd={kv.Value.returnStd:F2} corr={kv.Value.valueReturnCorr:F2} chained={kv.Value.chainedFraction:F3}"
                // mb= counts gradient steps ACTUALLY applied. Under Adam that is what sets how far the
                // policy can move in one update (~mb*lr), so when the KL overshoots this is the number to
                // read before touching the epoch count: fewer minibatches means the STEP SIZE is at fault.
                + $"w|max|={kv.Value.maxAbsWeight:0.###} ep={kv.Value.epochsRun}/{kv.Value.minibatchesRun}mb klAll={kv.Value.klAll:0.####}"
                + (kv.Value.poisoned ? " POISONED" : ""));
        }
        if (lines.Count == 0) return;
        Debug.Log("[Training] learner: " + string.Join(" | ", lines));

        // Curriculum state on its own line, because it changes on a completely different cadence from
        // the learner metrics and would otherwise be lost in them. The eval win rates are the only
        // generalisation signal in the run, so they belong on the console and not only in the CSV.
        if (!TrainingMode.curriculum)
        {
            Debug.Log("[Training] curriculum: OFF (-noCurriculum). Every match is plain self-play.");
            return;
        }
        var curricula = new List<string>();
        foreach (var c in PolicyLearner.Curricula) curricula.Add(c.ToString());
        if (curricula.Count > 0)
            Debug.Log($"[Training] curriculum (eval slice: 1 in {TrainingMode.evalEvery}, "
                      + $"promote after {PolicyLearner.CurriculumFor(PolicyLearner.RoleCompanion).AdvanceWindow} "
                      + $"graded evals): "
                      + string.Join(" | ", curricula));
    }

    /// <summary>
    /// Persist learned weights. Wired to the player's quit event so a long run that is stopped
    /// mid-way keeps everything it has learned instead of only the last autosave interval.
    /// </summary>
    private void OnApplicationQuit()
    {
        PolicyLearner.FlushAll();
        TrainingTelemetry.End();
    }

    private void OnDestroy()
    {
        PolicyLearner.FlushAll();
        // Also flush here, not just on quit: in the editor a run is normally ended by leaving play
        // mode, which does NOT raise OnApplicationQuit, so the tail of every in-editor run would
        // otherwise never reach disk.
        TrainingTelemetry.End();
    }
}