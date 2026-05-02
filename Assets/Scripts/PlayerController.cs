using UnityEngine;

[RequireComponent(typeof(CharacterAnimation))]
[RequireComponent(typeof(CharacterPhysics))]
[RequireComponent(typeof(CharacterData))]
public class PlayerController : MonoBehaviour
{
    private CharacterAnimation anim;
    private CharacterPhysics physics;
    private CharacterData characterData;

    public string SelectedMove { get; private set; } = "idle";

    private void Awake()
    {
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
        characterData = GetComponent<CharacterData>();
    }

    private void Start()
    {
        physics.Initialize(characterData);
        anim.Initialize(characterData.animations);
        TurnManager.Instance.RegisterPlayer(this);
    }

    public void SelectMove(string moveId)
    {
        SelectedMove = moveId;
        Debug.Log($"Move selected: {moveId}");
    }

    public void Flip(bool flipped)
    {
        if(flipped != transform.localScale.x < 0)
            transform.localScale = transform.localScale - new Vector3(transform.localScale.x * 2, 0, 0);
    }

    public void ExecuteMove(string moveId, System.Action onComplete = null)
    {
        anim.PlayMove(moveId, onComplete);
        var move = characterData.GetMove(moveId);
        if (move != null)
            physics.ApplyImpulse(move.impulse);
        physics.Step();
    }
}