using UnityEngine;

[RequireComponent(typeof(CharacterAnimation))]
[RequireComponent(typeof(CharacterPhysics))]
[RequireComponent(typeof(CharacterData))]
public class CharacterController : MonoBehaviour
{
    private CharacterAnimation anim;
    private CharacterPhysics physics;
    private CharacterData characterData;
    public int id {  get; private set; }
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
        anim.Initialize(characterData.animations, this.onFrameEvent);
        this.id = TurnManager.Instance.RegisterPlayer(this);
    }

    public void Damage(float damage)
    {
        this.characterData.health = Mathf.Max(0, this.characterData.health - damage);
    }

    private void onFrameEvent(FrameEventType frameEventType)
    {
        switch (frameEventType)
        {
            case FrameEventType.BasicAttack:
                HitboxManager.Instance.SubmitHitBox(
                    new HitBox(
                        this,
                        (target) => target.Damage(10),
                        transform.position.x - this.characterData.width / 1.2f,
                        transform.position.y - this.characterData.height / 2f,
                        transform.position.x + this.characterData.width / 1.2f,
                        transform.position.y + this.characterData.height / 2f
                    )
                );
                break;
        }
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
    public HurtBox getHurtBox()
    {
        return new HurtBox(
            this,
            transform.position.x - this.characterData.width / 2.5f,
            transform.position.y - this.characterData.height / 2.5f,
            transform.position.x + this.characterData.width / 2.5f,
            transform.position.y + this.characterData.height / 2.5f
        );
    }

    private void Update()
    {
        HitboxManager.Instance.SubmitHurtBox(this.getHurtBox());
    }

    public void ExecuteMove(string moveId, System.Action onComplete = null)
    {
        anim.PlayMove(moveId, onComplete);
        var move = characterData.GetMove(moveId);
        if (move != null)
            physics.ApplyImpulse(move.impulse);
    }

    void LateUpdate()
    {
        if(TurnManager.Instance.Phase == TurnPhase.Simulating)
            physics.Step();
    }
}