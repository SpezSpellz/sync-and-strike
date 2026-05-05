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

    public struct SaveData
    {
        public float health;
        public Vector2 pos;
        public Vector3 localScale;
        public Vector2 velocity;
    }

    private void Awake()
    {
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
        characterData = GetComponent<CharacterData>();
    }

    public virtual void Start()
    {
        physics.Initialize(characterData);
        anim.Initialize(characterData.animations, this.onFrameEvent);
        this.id = TurnManager.Instance.RegisterPlayer(this);
    }

    public void Damage(float damage)
    {
        this.characterData.health = Mathf.Max(0, this.characterData.health - damage);
    }
    public bool IsDead()
    {
        return this.characterData.health <= 0;
    }

    public float GetHealth()
    {
        return this.characterData.health;
    }

    public Vector2 GetPosition()
    {
        return this.physics.getPosition();
    }

    public SaveData Save()
    {
        return new SaveData
        {
            pos = this.physics.getPosition(),
            health = this.characterData.health,
            localScale = this.transform.localScale,
            velocity = this.physics.getVelocity(),
        };
    }

    public void Load(SaveData savedata)
    {
        this.physics.setPosition(savedata.pos.x, savedata.pos.y);
        this.physics.setVelocity(savedata.velocity.x, savedata.velocity.y);
        this.characterData.health = savedata.health;
        this.transform.localScale = savedata.localScale;
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

    public virtual void RequestDecision()
    {

    }

    public void Step()
    {
        anim.Step();
        physics.Step();
        HitboxManager.Instance.SubmitHurtBox(this.getHurtBox());
    }

    public void ExecuteMove(string moveId, System.Action onComplete = null)
    {
        anim.PlayMove(moveId, onComplete);
        var move = characterData.GetMove(moveId);
        if (move != null)
            physics.ApplyImpulse(move.impulse);
    }
}