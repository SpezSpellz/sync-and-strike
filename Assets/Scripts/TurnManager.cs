using System;
using System.Collections.Generic;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    private const float SECONDS_PER_FRAME = 1f / 60f; // SET GAME FRAME RATE TO 60 FPS. DO NOT CHANGE
    // private const float SECONDS_PER_FRAME = 0.001f; // SET GAME FRAME RATE TO VERY HIGH FOR TRAINING

    [SerializeField]
    private bool fastForward;

    [Tooltip("Hard cap on how many frames a single turn may simulate before it is force-resolved.")]
    private int maxTurnFrames = 180;

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
            UIManager.Instance.HideMoveUI();
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
        UIManager.Instance.ShowMoveUI();
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
                    int count = 0;
                    ticksAwaiting += Time.deltaTime;
                    while (ticksAwaiting > 0 || (fastForward && ++count < 100))
                    {
                        ticksAwaiting -= SECONDS_PER_FRAME;
                        int totalPlayers = players.getList().Count;
                        if (completedCount >= totalPlayers || framesThisTurn >= maxTurnFrames)
                        {
                            if (completedCount < totalPlayers) WarnUnfinishedFighters();
                            ResolveTurn();
                            break;
                        }
                        foreach (var player_turn_data in players.getList())
                        {
                            player_turn_data.player.Step();
                        }
                        HitboxManager.Instance.Step();
                        framesThisTurn++;
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

    private void ResolveTurn()
    {
        if (Phase != TurnPhase.Simulating) return;
        Phase = TurnPhase.Resolved;

        // Anyone still busy (e.g. frozen in hitstun) is force-cleared so the next turn can start.
        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.ForceFinishMove();
        }

        var companion = FindCompanion();
        if (companion != null && !companion.IsDead())
        {
            // Ask the player to rate the companion's move before the next planning phase.
            var voteManager = CompanionVoteManager.Instance;
            if (voteManager != null && voteManager.VotingEnabled)
            {
                voteManager.PrepareForTurn(companion);
                Phase = TurnPhase.AwaitingVote;
                voteAwaitStartedAt = Time.time;
                companion.PrepareVoting();
                return;
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
            CompanionVoteManager.Instance?.ForceResolveAsSkipped();
            BeginPlanning();
        }
    }
}
