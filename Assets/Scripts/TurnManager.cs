using System.Collections;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    class PlayerTurnData
    {
        public PlayerController player;
        public string? submitted_move;
    }

    private IndexSet<PlayerTurnData> players = new();
    private int submittedMoves;

    [SerializeField] private float defaultTurnDuration = 20f; // Keep it for now even if unused

    private void Awake()
    {
        Instance = this;
        Phase = TurnPhase.Planning;

        Application.targetFrameRate = 60; // SET GAME FRAME RATE TO 60 FPS. DO NOT CHANGE
    }

    public int RegisterPlayer(PlayerController p)
    {
        var plr_data = this.players.getOr(p.id, default);
        if (plr_data != null && plr_data.player == p)
            return p.id;
        return players.add(new PlayerTurnData{
            player = p,
        });
    }

    public void SubmitMove(PlayerController p)
    {
        if (Phase != TurnPhase.Planning) return;
        print("Get Player " + p.id);
        var plr_data = this.players.get(p.id);
        if (plr_data.submitted_move == null)
            ++submittedMoves;
        plr_data.submitted_move = p.SelectedMove;
        if (submittedMoves == players.getList().Count) // start the round of every player have locked-in (submitted their move)
            StartCoroutine(SimulateRound());
    }

    private IEnumerator SimulateRound()
    {
        Phase = TurnPhase.Simulating;

        int completedCount = 0;
        int totalPlayers = players.getList().Count;

        foreach (var player_turn_data in players.getList())
        {
            player_turn_data.player.ExecuteMove(player_turn_data.submitted_move, () => {
                completedCount++;
            });
        }

        // wait until all players finish their animation, CHANGE HERE FOR FRAME LOGIC
        yield return new WaitUntil(() => completedCount >= totalPlayers);

        CombatManager.Instance.ResolveAllHits();

        Phase = TurnPhase.Resolved;

        // yield return new WaitForSeconds(0.3f);

        submittedMoves = 0;
        foreach (var player_turn_data in players.getList())
            player_turn_data.submitted_move = null;
        Phase = TurnPhase.Planning;
    }
}