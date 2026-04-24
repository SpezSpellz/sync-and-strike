using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private CharacterAnimation anim;
    private CharacterPhysics physics;

    private string selectedMove = "idle";

    private void Awake()
    {
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

        if (Input.GetKeyDown(KeyCode.E)) selectedMove = "horizontal_slash";
        if (Input.GetKeyDown(KeyCode.Q)) selectedMove = "vertical_slash";
        if (Input.GetKeyDown(KeyCode.W)) selectedMove = "block";

        if (Input.GetKeyDown(KeyCode.Return))
            TurnManager.Instance.SubmitMove(this, selectedMove);
    }

    public void ExecuteMove(string moveId)
    {
        anim.PlayMove(moveId);
        physics.ApplyMoveImpulse(moveId);
    }
}