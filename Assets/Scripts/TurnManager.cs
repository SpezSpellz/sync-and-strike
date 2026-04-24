using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TurnManager : MonoBehaviour
{
    public static TurnManager Instance { get; private set; }
    public TurnPhase Phase { get; private set; }

    private Dictionary<PlayerController, string> submittedMoves = new();
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

        foreach (var (player, moveId) in submittedMoves)
            player.ExecuteMove(moveId);

        yield return new WaitForSeconds(simulationDuration);

        CombatManager.Instance.ResolveAllHits();

        Phase = TurnPhase.Resolved;

        // Small pause before next planning phase
        yield return new WaitForSeconds(0.3f);

        submittedMoves.Clear();
        Phase = TurnPhase.Planning;
    }
}