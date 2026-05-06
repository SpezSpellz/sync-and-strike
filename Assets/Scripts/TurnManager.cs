using System;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    private const float SECONDS_PER_FRAME = 0.016f; // SET GAME FRAME RATE TO 60 FPS. DO NOT CHANGE
    // private const float SECONDS_PER_FRAME = 0.001f; // SET GAME FRAME RATE TO VERY HIGH FOR TRAINING

    [SerializeField]
    private bool fastForward;

    class PlayerTurnData
    {
        public CharacterController player;
        public CharacterController.SaveData saveData;
        public string? submitted_move;
    }

    private IndexSet<PlayerTurnData> players = new();
    private int submittedMoves;
    private int completedCount = 0;
    private float ticksAwaiting = 0.0f;

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

    public void ResetState()
    {
        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.Load(player_turn_data.saveData);
        }
        // Must come after load of other players
        foreach (var player_turn_data in players.getList())
        {
            if (player_turn_data.player is EnemyController agent)
                agent.ResetMetrics();
        }
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
        // start the round of every player have locked-in (submitted their move)
        if (submittedMoves >= players.getList().Count)
        {
            Phase = TurnPhase.Simulating;
            completedCount = 0;
            foreach (var player_turn_data in players.getList())
            {
                player_turn_data.player.ExecuteMove(player_turn_data.submitted_move ?? "idle", () => {
                    completedCount++;
                });
            }
        }
    }

    private void Update()
    {
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
                        if (completedCount >= totalPlayers)
                        {
                            Phase = TurnPhase.Planning;
                            submittedMoves = 0;
                            foreach (var player_turn_data in players.getList())
                            {
                                player_turn_data.submitted_move = null;
                                player_turn_data.player.RequestDecision();
                            }
                            break;
                        }
                        foreach (var player_turn_data in players.getList())
                        {
                            player_turn_data.player.Step();
                        }
                        HitboxManager.Instance.Step();
                    }
                    break;
                }
        }
    }
}