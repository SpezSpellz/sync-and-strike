using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private CharacterAnimation anim;
    private CharacterPhysics physics;

    public string SelectedMove { get; private set; } = "idle";

    private void Awake()
    {
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
    }

    private void Start()
    {
        TurnManager.Instance.RegisterPlayer(this);
    }

    public void SelectMove(string moveId)
    {
        SelectedMove = moveId;
        Debug.Log($"Move selected: {moveId}");
    }

    public void ExecuteMove(string moveId, System.Action onComplete = null)
    {
        anim.PlayMove(moveId, onComplete);
        physics.ApplyMoveImpulse(moveId);
    }
}