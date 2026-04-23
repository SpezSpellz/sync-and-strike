using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private TurnManager turnManager;
    private CharacterAnimation anim;
    private CharacterPhysics physics;

    private Move selectedMove = Move.Idle;

    private void Awake()
    {
        turnManager = TurnManager.Instance;
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
    }

    private void Start()
    {
        TurnManager.Instance.RegisterPlayer(this);
    }

    private void Update()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        if (Input.GetKeyDown(KeyCode.E)) selectedMove = Move.Punch;
        if (Input.GetKeyDown(KeyCode.Q)) selectedMove = Move.Kick;
        if (Input.GetKeyDown(KeyCode.W)) selectedMove = Move.Dodge;
        if (Input.GetKeyDown(KeyCode.R)) selectedMove = Move.Grab;

        if (Input.GetKeyDown(KeyCode.Return))
            TurnManager.Instance.SubmitMove(this, selectedMove);
    }

    public void ExecuteMove(Move move)
    {
        anim.PlayMove(move);
        physics.ApplyMoveImpulse(move);
    }
}