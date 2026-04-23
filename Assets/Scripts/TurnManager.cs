using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    private Dictionary<PlayerController, Move> submittedMoves = new();
    private List<PlayerController> players = new();

    [SerializeField] private float simulationDuration = 1.2f;

    private void Awake()
    {
        Instance = this;
        Phase = TurnPhase.Planning;
    }

    public void RegisterPlayer(PlayerController p)
    {
        if (!players.Contains(p))
            players.Add(p);
    }

    public void SubmitMove(PlayerController player, Move move)
    {
        if (Phase != TurnPhase.Planning) return;

        submittedMoves[player] = move;

        if (submittedMoves.Count == players.Count)
            StartCoroutine(SimulateRound());
    }

    private IEnumerator SimulateRound()
    {
        Phase = TurnPhase.Simulating;

        foreach (var (player, move) in submittedMoves)
            player.ExecuteMove(move);

        yield return new WaitForSeconds(simulationDuration);

        CombatManager.Instance.ResolveAllHits();

        Phase = TurnPhase.Resolved;

        // Small pause before next planning phase
        yield return new WaitForSeconds(0.3f);

        submittedMoves.Clear();
        Phase = TurnPhase.Planning;
    }
}