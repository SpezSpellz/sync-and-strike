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
            run.arena.TurnManager.TurnResolved += () => OnMatchEnd(run);
        }
        pending.Clear();

        if (runs.Count == 0)
        {
            Debug.LogError("[Training] No arenas were built. Check that TrainingArenaBuilder is on "
                + "the same GameObject and that fighterPrefab (if set) carries an AIController.");
            return;
        }
        Debug.Log($"TrainingMatchRunner: {runs.Count} arenas, up to {arenaCount} matches in flight "
            + "(2v1 each).");
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
    /// Point the HUD health bars at the fighters that are actually fighting.
    ///
    /// The bars are scene objects with SERIALIZED owner references to the original player and enemy,
    /// and the fighters that fight are clones spawned per arena. Once the originals are deactivated the
    /// bars keep reading their untouched CharacterData and stay full, so only the companion's bar moved -
    /// it is the one bar that re-resolves through TurnManager at runtime. This is presentation only: it
    /// has no effect on damage, rewards or telemetry.
    ///
    /// Matched by TEAM rather than by slot, because there is no single "the" player fighter once arenas
    /// exist. The team is read from the bar's CURRENT owner, which is still the original fighter at this
    /// point - that is the whole reason we can tell which bar is which. Where no owner is set the bar's
    /// name is used as a fallback.
    /// </summary>
    private void RebindHudToArenaFighters()
    {
        var bars = FindObjectsByType<HealthBar>(FindObjectsSortMode.None);
        if (bars.Length == 0) return;

        foreach (var bar in bars)
        {
            // The companion bar is deliberately left alone: CompanionHealthBarUI resolves it through
            // TurnManager and re-binds on its own schedule, and stomping it here would race that.
            if (bar.Owner != null && bar.Owner.Team == CombatTeam.Companion) continue;

            CombatTeam team;
            if (bar.Owner != null) team = bar.Owner.Team;
            else if (bar.transform.parent != null && bar.name.IndexOf("Enemy", StringComparison.OrdinalIgnoreCase) >= 0)
                team = CombatTeam.Enemy;
            else team = CombatTeam.Player;

            var fighter = LatestArenaFighter(team);
            if (fighter != null) bar.Bind(fighter);
        }
    }

    /// <summary>
    /// A fighter on the given team belonging to an arena, choosing the lowest instance id so the choice
    /// is STABLE across calls.
    ///
    /// Stability matters more than recency here. A single HUD bar cannot represent N simultaneous
    /// arenas, so this deliberately picks one deterministically instead of following whichever arena
    /// happened to act last - a bar that jumped between arenas every turn would be unreadable. The
    /// consequence is that the HUD shows one arena's fighters while the others run unwatched; the
    /// training CSV remains the per-arena record.
    /// </summary>
    private CharacterController LatestArenaFighter(CombatTeam team)
    {
        CharacterController best = null;
        int bestKey = int.MaxValue;
        foreach (var f in FindObjectsByType<CharacterController>(FindObjectsSortMode.None))
        {
            if (f == null || f.GetComponentInParent<Arena>() == null) continue;
            if (f.Data == null || f.Data.team != team) continue;
            int key = f.GetInstanceID();
            if (key < bestKey) { best = f; bestKey = key; }
        }
        return best;
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

        totalMatches++;
        RecordTelemetry(run, result);

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
        // Re-snapshot: health was just restored, and the next match's damage credit is measured
        // against this baseline rather than the previous match's starting health.
        SnapshotMatchStart(run);
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

        // avgFramesPerTurn separates "few turns, each expensive" from "many turns, each cheap".
        // Without it a low turns/s is ambiguous: the turn count alone cannot distinguish a policy that
        // stalls (turns hit the frame cap) from a genuinely slow simulation, and those need opposite fixes.
        Debug.Log($"[Training] heartbeat: {totalMatches} matches done, arena0 at turn {run.turns}, "
            + $"phase={run.arena.TurnManager.Phase} "
            + $"[{elapsed:0}s {turnsPerSec:0.00} turns/s {framesPerSec:0} simframes/s "
            + $"{avgFramesPerTurn:0.0} frames/turn "
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
        TrainingTelemetry.RecordMatch(handle, result.ToString(), totalMatches, credit);
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