using System;
using System.Collections.Generic;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }
    [HideInInspector] public Arena Arena;

    private HitboxManager Hitbox => Arena != null ? Arena.HitboxManager : HitboxManager.Instance;
    private UIManager UI => Arena != null ? Arena.UIManager : UIManager.Instance;
    private CompanionVoteManager VoteManager => Arena != null ? Arena.CompanionVoteManager : CompanionVoteManager.Instance;

    /// <summary>
    /// Fired at the end of every simulated turn, after fighters are force-finished but BEFORE the
    /// vote prompt pauses the loop. The companion's on-device PPO learner uses this to turn the
    /// resolved turn into a reward. Instance-scoped so multi-arena training can route per arena.
    /// </summary>
    public event Action TurnResolved;

    private float LOCAL_SECONDS_PER_FRAME => TrainingMode.enabled ? TrainingMode.secondsPerFrame : 1f / 60f;
    private const float DEFAULT_SECONDS_PER_FRAME = 1f / 60f; // SET GAME FRAME RATE TO 60 FPS. DO NOT CHANGE

    [SerializeField]
    private bool fastForward;

    /// <summary>
    /// Training-mode simulation cap per Unity frame.
    ///
    /// TurnManager's inner loop is bounded by this rather than running to completion, so one slow arena
    /// cannot consume an entire frame and stall every other arena sharing the process. The bound is what
    /// makes N arenas in one process scale: total throughput is (arenas x this) simulation frames per
    /// rendered frame, capped by however long a frame actually takes.
    ///
    /// Set from <see cref="TrainingMode.maxSimFramesPerUnityFrame"/> so a throughput run can be pushed
    /// harder without touching this component.
    /// </summary>
    private int MaxSimFramesThisFrame =>
        TrainingMode.enabled ? TrainingMode.maxSimFramesPerUnityFrame : 100;

    [Tooltip("Hard cap on how many frames a single turn may simulate before it is force-resolved.")]
    private int maxTurnFrames = 180;

    /// <summary>Simulation frames stepped during the current Unity frame, across all phases.</summary>
    public int SimFramesThisFrame { get; private set; }

    /// <summary>Total simulation frames stepped since this manager was created. Monotonic, unlike any
    /// per-match figure, so it can be differenced to get a real frames/second rate.</summary>
    public long TotalSimFrames { get; private set; }

    /// <summary>Total turns resolved since this manager was created. Monotonic for the same reason as
    /// <see cref="TotalSimFrames"/>, and the only correct basis for a turns/second rate: the per-match
    /// counter is reset on every match end, so dividing it by total elapsed time has no meaning.</summary>
    public long TotalTurns { get; private set; }

    [Tooltip("Backstop: resume the turn if the rating prompt has not resolved by now.")]
    private float maxVoteWaitSeconds = 30f;

    private float voteAwaitStartedAt = 0f;

    class PlayerTurnData
    {
        public CharacterController player;
        public CharacterController.SaveData saveData;
        public string submitted_move;
    }

    private IndexSet<PlayerTurnData> players = new();
    private int submittedMoves;
    private int completedCount = 0;
    private float ticksAwaiting = 0.0f;
    private int framesThisTurn = 0;

    [SerializeField] private float defaultTurnDuration = 20f; // Keep it for now even if unused

    private void Awake()
    {
        Instance = this;
        Arena = GetComponentInParent<Arena>();
        Phase = TurnPhase.Planning;
    }

    public void ForEachPlayer(Action<CharacterController> callback)
    {
        foreach (PlayerTurnData playerTurnData in players.getList())
            callback(playerTurnData.player);
    }

    public List<CharacterController> GetAllPlayers()
    {
        var result = new List<CharacterController>();
        foreach (var entry in players.getList())
            result.Add(entry.player);
        return result;
    }

    public CharacterController FindByTeam(CombatTeam team)
    {
        foreach (var entry in players.getList())
        {
            if (entry.player != null && entry.player.Team == team)
                return entry.player;
        }
        return null;
    }

    public CompanionController FindCompanion()
    {
        foreach (var entry in players.getList())
        {
            if (entry.player is CompanionController companion)
                return companion;
        }
        return null;
    }

    /// <summary>
    /// True while <see cref="ResolveTurn"/> is running, i.e. while a <see cref="TurnResolved"/>
    /// handler is on the stack.
    ///
    /// This flag exists because of a re-entrancy bug. The training runner resets the match from
    /// inside its TurnResolved handler, and ResetState used to call BeginPlanning() directly. Control
    /// then returned to ResolveTurn, whose tail ALSO calls BeginPlanning() - so planning ran twice
    /// per turn. RequestDecision was therefore called twice per fighter, the PPO transition was
    /// recorded twice and then overwritten (orphaning the first sample), and ExecuteMove ran twice,
    /// restarting every animation and resetting actionReported.
    ///
    /// Now a reset arriving mid-resolve only records intent, and ResolveTurn performs the single
    /// BeginPlanning at its tail.
    /// </summary>
    private bool resolvingTurn;

    public void ResetState()
    {
        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.Load(player_turn_data.saveData);
        }
        // Must come after load of other players
        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.ResetDecisionMetrics();
        }

        // The SaveData load above has already happened; only the new round is deferred, and
        // ResolveTurn reaches exactly one BeginPlanning on its way out.
        if (resolvingTurn) return;

        BeginPlanning();
    }

    public int RegisterPlayer(CharacterController p)
    {
        var plr_data = this.players.getOr(p.id, default);
        if (plr_data != null && plr_data.player == p)
            return p.id;
        return players.add(new PlayerTurnData{
            player = p,
            saveData = p.Save(),
        });
    }

    public bool IsSubmitted(CharacterController p)
    {
        return this.players.get(p.id).submitted_move != null;
    }

    public void SubmitMove(CharacterController p)
    {
        if (Phase != TurnPhase.Planning) return;
        var plr_data = this.players.get(p.id);
        if (plr_data.submitted_move == null)
            ++submittedMoves;
        plr_data.submitted_move = p.SelectedMove;

        // start the round once every fighter has locked in (submitted their move)
        if (submittedMoves >= players.getList().Count)
        {
            Phase = TurnPhase.Simulating;
            completedCount = 0;
            framesThisTurn = 0;
            ticksAwaiting = 0f;
            UI?.HideMoveUI();
            foreach (var player_turn_data in players.getList())
            {
                var player = player_turn_data.player;
                player.HideMovePreview();
                player.ExecuteMove(player_turn_data.submitted_move ?? "continue", () =>
                {
                    if (Phase != TurnPhase.Simulating) return;
                    completedCount++;
                });
            }
        }
    }

    private void BeginPlanning()
    {
        Phase = TurnPhase.Planning;
        submittedMoves = 0;
        completedCount = 0;
        framesThisTurn = 0;

        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.submitted_move = null;
            var player = player_turn_data.player;
            if (player == null) continue;

            // Unconditional: this used to sit inside the TargetPosition check, so fighters with no
            // target transform (the enemy and the companion) kept last turn's SelectedMove.
            player.ResetMove();

            // Auto-face the opponent, but never override a facing the player explicitly chose with
            // the Flip button, otherwise that choice was reverted at the start of every turn.
            if (player.TargetPosition != null && !player.FacingChosenByPlayer)
            {
                if (player.transform.localPosition.x <= player.TargetPosition.localPosition.x)
                    player.Flip(false);
                else player.Flip(true);
            }
            player.ResetPreviewScale();

            // The combo counters, the
            // grounded-hit counter that drives the launch rule and the wall-slam counter all
            // reset here; without it they accumulated across the whole match.
            player.ResetComboState();
            player.RequestDecision();
        }
        UI?.ShowMoveUI();
    }

    /// <summary>
    /// Called by <see cref="CompanionVoteManager"/> once the player has rated (or timed out on)
    /// the companion's move. Until this fires the game waits on the rating prompt.
    /// </summary>
    public void ResumeAfterVote()
    {
        BeginPlanning();
    }

    private void Update()
    {
        // Reset per-frame before any stepping, so a reader sampling this after Update() sees the whole
        // frame's work rather than the tail of it.
        SimFramesThisFrame = 0;

        UpdateVoteTimeout();
        switch(Phase)
        {
            case TurnPhase.Planning:
                {
                    foreach (var player_turn_data in players.getList())
                    {
                        if(player_turn_data.submitted_move == null)
                            player_turn_data.player.RequestDecision();
                    }
                    break;
                }
            case TurnPhase.Simulating:
                {
                    int budget = MaxSimFramesThisFrame;
                    // The accumulator is drained in fixed simulation steps rather than one step per unit
                    // of leftover real time. In training secondsPerFrame is 0.001s, so a 16ms Unity frame
                    // asks for ~16 sim frames; capping at `budget` and DISCARDING the remainder is
                    // deliberate, because carrying the surplus forward would make the backlog grow without
                    // bound under load and every arena would fall progressively further behind real time.
                    // Dropping it means a frame that overruns its budget simply simulates less, which is
                    // the correct trade for a throughput run.
                    // In normal play, simulated time tracks REAL time: a 16.7ms Unity frame buys one 1/60s
                    // step, so the game runs at real speed regardless of frame rate.
                    //
                    // In training that coupling is exactly wrong. It ties simulated time to wall-clock
                    // time, so throughput is capped near 1x no matter how high maxSimFramesPerUnityFrame
                    // is set - measured 1.04 turns/s against a budget of 64 that was never even reached,
                    // because a 16.7ms frame only ever asks for ~17 steps at a 1ms timestep. Raising
                    // -simFrames could not have helped; the accumulator, not the budget, was binding.
                    //
                    // Training therefore runs a FULL budget every Unity frame. Simulated time advances as
                    // fast as the CPU allows rather than in step with the wall clock, which is what makes
                    // multi-arena throughput runs possible at all. The physics the policies learn is
                    // affected in the same way -simSpeed already affects it, which is why that is also
                    // documented as training-only.
                    int steps;
                    if (TrainingMode.enabled)
                    {
                        ticksAwaiting = 0f;
                        steps = budget;
                    }
                    else
                    {
                        // A real-time accumulator: keep the REMAINDER instead of discarding it, and round
                        // DOWN rather than up.
                        //
                        // Both details are load-bearing, and together they are why the game ran fast
                        // whenever it was not vSync-locked to 60.
                        //
                        // CeilToInt with the accumulator zeroed meant any frame shorter than 1/60s still
                        // bought a WHOLE step. That is correct at exactly 60fps and wrong at every other
                        // rate: on a machine rendering at 200fps, deltaTime is 5ms, 5ms/16.7ms rounds up
                        // to 1 step, and the game therefore simulates 200 steps per second instead of 60 -
                        // over three times real speed. Discarding the leftover time made the surplus
                        // unrecoverable, so the error accumulated every single frame and could never
                        // correct itself.
                        //
                        // Keeping the remainder lets a faster machine take two steps on the frames that
                        // are long enough, and lets a slower one fall behind by a fraction of a step at
                        // most. Rounding down (never up) guarantees the simulation never runs ahead of
                        // the wall clock, which is the direction that looks like a bug to the player.
                        ticksAwaiting += Time.deltaTime;
                        int stepsWanted = Mathf.FloorToInt(ticksAwaiting / LOCAL_SECONDS_PER_FRAME);
                        ticksAwaiting -= stepsWanted * LOCAL_SECONDS_PER_FRAME;
                        // Only when there is genuinely time banked. Forcing a minimum of 1 here would
                        // reintroduce the run-fast bug at high frame rates.
                        steps = Mathf.Min(stepsWanted, budget);
                    }

                    while (steps-- > 0)
                    {
                        int totalPlayers = players.getList().Count;
                        if (completedCount >= totalPlayers || framesThisTurn >= maxTurnFrames)
                        {
                            if (completedCount < totalPlayers) WarnUnfinishedFighters();
                            ResolveTurn();
                            // ResolveTurn runs the whole hand-off synchronously: for AI fighters its tail
                            // BeginPlanning calls RequestDecision -> SubmitMove -> ExecuteMove, which
                            // leaves Phase back in Simulating with a fresh turn already assembled. So the
                            // remaining budget can be spent on the NEXT turn.
                            //
                            // The unconditional break that used to sit here threw that budget away and
                            // pinned throughput to one turn per rendered frame regardless of -simFrames:
                            // measured ~4,576 simframes/s against a 64-frame budget (≈71 rendered fps) is
                            // exactly one 50-frame turn per frame, i.e. ~75 turns/s with 95% of the budget
                            // unused. Continuing is what lets one Unity frame simulate many turns.
                            //
                            // Guarded on Phase rather than assumed, because a turn can also end in
                            // AwaitingVote (not in training) or be left unresolved if a fighter never
                            // submitted; in those cases stepping further would run the sim outside a turn.
                            if (Phase != TurnPhase.Simulating) break;
                        }
                        foreach (var player_turn_data in players.getList())
                        {
                            player_turn_data.player.Step();
                        }
                        Hitbox.Step();
                        framesThisTurn++;
                        SimFramesThisFrame++;
                        TotalSimFrames++;
                    }
                    break;
                }
        }
    }

    /// <summary>
    /// A fighter that never reports completion would otherwise freeze the turn for the whole frame
    /// cap with nothing in the log to explain it. Name them, the move they were given and the state
    /// they were stuck in.
    /// </summary>
    private void WarnUnfinishedFighters()
    {
        foreach (var entry in players.getList())
        {
            var p = entry.player;
            if (p == null || p.HasReportedAction) continue;
            Debug.LogWarning(
                $"Turn hit the {maxTurnFrames}-frame cap without '{p.name}' finishing its action " +
                $"(move '{entry.submitted_move}', state {p.State}). Force-resolving the turn.");
        }
    }

    /// <summary>
    /// Simulated frames the most recently resolved turn consumed, versus <see cref="maxTurnFrames"/>.
    ///
    /// Published because "the turn took the full frame cap" and "the turn finished normally" look
    /// identical in the console unless you know which happened. Training telemetry records it, and it
    /// is how a missing completion path in a fighter state gets spotted: the number pins at the cap.
    /// </summary>
    public int FramesLastTurn { get; private set; }

    private void ResolveTurn()
    {
        if (Phase != TurnPhase.Simulating) return;
        Phase = TurnPhase.Resolved;
        FramesLastTurn = framesThisTurn;
        TotalTurns++;

        // Everything from here to the single BeginPlanning at the tail runs with this set, so a
        // ResetState() from a TurnResolved handler cannot start a second planning phase.
        resolvingTurn = true;

        // Anyone still busy (e.g. frozen in hitstun) is force-cleared so the next turn can start.
        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.ForceFinishMove();
        }

        // Reward hook for the companion's on-device learner. Fired before the vote prompt pauses
        // the loop so the learner sees the resolved turn immediately.
        try { TurnResolved?.Invoke(); } catch (System.Exception e) { Debug.LogException(e); }

        // The reset itself already ran - ResetState loads every fighter's SaveData synchronously
        // before returning. Only the BeginPlanning was deferred, and exactly one of the two exits
        // below reaches it, so there is nothing left to carry across.
        resolvingTurn = false;

        var companion = FindCompanion();
        if (companion != null && !companion.IsDead())
        {
            // Ask the player to rate the companion's move before the next planning phase.
            var voteManager = VoteManager;
            if (voteManager != null && voteManager.VotingEnabled)
            {
                voteManager.PrepareForTurn(companion);
                Phase = TurnPhase.AwaitingVote;
                voteAwaitStartedAt = Time.time;
                companion.PrepareVoting();
                return;   // ResumeAfterVote / UpdateVoteTimeout calls BeginPlanning
            }
        }

        BeginPlanning();
    }

    /// <summary>
    /// Backstop for the rating prompt. If the prompt somehow never resolves (window not focused,
    /// UI event system missing) the turn resumes rather than soft-locking the game.
    /// </summary>
    private void UpdateVoteTimeout()
    {
        if (Phase != TurnPhase.AwaitingVote) return;
        if (Time.time - voteAwaitStartedAt > maxVoteWaitSeconds)
        {
            VoteManager?.ForceResolveAsSkipped();
            BeginPlanning();
        }
    }
}
