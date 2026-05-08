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
        anim.Initialize(characterData.animations, this.onFrameEvent, physics);
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

    public Vector2 GetVelocity()
    {
        return this.physics.getVelocity();
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
    private void onFrameEvent(FrameEvent frameEvent)
    {
        switch (frameEvent.type)
        {
            case FrameEventType.SpawnHitbox:
                SpawnHitbox(frameEvent.hitboxData);
                break;
            case FrameEventType.SpawnVFX:
                // read from data later
                break;
            case FrameEventType.SpawnSFX:
                // read from data later
                break;
            case FrameEventType.Block:
                // maybe trigger block stun here or something? for now just a placeholder
                break;
        }
    }

    private void SpawnHitbox(HitboxData data)
    {
        if (data.damage == 0 && data.knockback == Vector2.zero) return;

        float facing = physics.FacingDirection().x;

        float minX = transform.position.x + facing * data.offsetX - data.width * 0.5f;
        float maxX = transform.position.x + facing * data.offsetX + data.width * 0.5f;
        float minY = transform.position.y + data.offsetY - data.height * 0.5f;
        float maxY = transform.position.y + data.offsetY + data.height * 0.5f;

        HitboxManager.Instance.SubmitHitBox(
            new HitBox(
                this,
                (target) =>
                {
                    target.physics.ApplyKnockback(new Vector2(data.knockback.x * facing, data.knockback.y));
                    target.Damage(data.damage);
                },
                minX, minY, maxX, maxY
            )
        );
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
            transform.position.x - this.characterData.width / 2f,
            transform.position.y - this.characterData.height / 2f,
            transform.position.x + this.characterData.width / 2f,
            transform.position.y + this.characterData.height / 2f
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
        if (move != null && !move.continuousImpulse) // if the move has an impulse and isn't continuous, apply it immediately. if it's continuous, the impulse will be applied in the CharacterAnimation's Step function.
            physics.ApplyImpulse(move.impulse);
    }
}