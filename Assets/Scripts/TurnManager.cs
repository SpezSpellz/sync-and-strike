using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    private Dictionary<PlayerController, string> submittedMoves = new();
    private List<PlayerController> players = new();

    [SerializeField] private float defaultTurnDuration = 20f; // Keep it for now even if unused

    private void Awake()
    {
        Instance = this;
        Phase = TurnPhase.Planning;

        Application.targetFrameRate = 60; // SET GAME FRAME RATE TO 60 FPS. DO NOT CHANGE
    }

    public void RegisterPlayer(PlayerController p)
    {
        if (!players.Contains(p))
            players.Add(p);
    }

    public void SubmitMove(PlayerController player, string moveId)
    {
        if (Phase != TurnPhase.Planning) return;

        submittedMoves[player] = moveId;

        if (submittedMoves.Count == players.Count) // start the round of every player have locked-in (submitted their move)
            StartCoroutine(SimulateRound());
    }

    private IEnumerator SimulateRound()
    {
        Phase = TurnPhase.Simulating;

        int completedCount = 0;
        int totalPlayers = submittedMoves.Count;

        foreach (var (player, moveId) in submittedMoves)
        {
            player.ExecuteMove(moveId, () => {
                completedCount++;
            });
        }

        // wait until all players finish their animation, CHANGE HERE FOR FRAME LOGIC
        yield return new WaitUntil(() => completedCount >= totalPlayers);

        CombatManager.Instance.ResolveAllHits();

        Phase = TurnPhase.Resolved;

        yield return new WaitForSeconds(0.3f);

        submittedMoves.Clear();
        Phase = TurnPhase.Planning;
    }
}