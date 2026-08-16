using NUnit.Framework;
using Unity.MLAgents.Integrations.Match3;
using Unity.VisualScripting;
using UnityEngine;

[RequireComponent(typeof(CharacterAnimation))]
[RequireComponent(typeof(CharacterPhysics))]
[RequireComponent(typeof(CharacterData))]
public class CharacterController : MonoBehaviour
{
    const string CONTINUE = "continue";
    private CharacterAnimation anim;
    private CharacterPhysics physics;
    private CharacterData characterData;
    private PreviewController previewController;
    private bool blocking = false;
    public int id {  get; private set; }
    public bool IsGrounded => physics != null && physics.IsGrounded;
    public AnimationData[] animations => characterData.animations;
    public bool IsBusy { get; private set; } = false;
    public bool IsKnockedBack { get; private set; } = false;
    public string SelectedMove { get; private set; } = CONTINUE;
    public float JumpDirection { get; private set; } = 1.5707963267948966f;
    public float JumpPower { get; private set; } = 1;
    public float PreviewJumpDirection { get; private set; } = 1.5707963267948966f;
    public float PreviewJumpPower { get; private set; } = 1f;
    public float KnockbackIndirectionDirection { get; private set; } = 0;
    public float KnockbackIndirectionPower { get; private set; } = 0;
    public float PreviewKnockbackIndirectionDirection { get; private set; } = 0f;
    public float PreviewKnockbackIndirectionPower { get; private set; } = 0f;
    public Vector3 PreviewScale { get; private set; }
    public Transform TargetPosition { get; protected set; }
    public bool IsPreviewReady => previewController != null && characterData != null && anim != null;

    // Direction in radians
    public void setJumpInfo(float power, float direction, bool previewOnly = true)
    {
        float clampedPower = Mathf.Clamp(power, 0.5f, 1f);
        // 30 degrees clamp
        float clampedDirection = Mathf.Clamp(direction, 0.5235987755982988f, 2.6179938779914944f);
        if (!previewOnly)
        {
            JumpPower = clampedPower;
            JumpDirection = clampedDirection;
        }
        PreviewJumpPower = clampedPower;
        PreviewJumpDirection = clampedDirection;
    }

    // Direction in radians
    public void setKnockbackInfo(float power, float direction, bool previewOnly = true)
    {
        float clampedPower = Mathf.Clamp(power, 0f, 1f);
        float clampedDirection = Mathf.Clamp(direction, 0f, 6.283185307179586f);
        if (!previewOnly)
        {
            KnockbackIndirectionPower = clampedPower;
            KnockbackIndirectionDirection = clampedDirection;
        }
        PreviewKnockbackIndirectionPower = clampedPower;
        PreviewKnockbackIndirectionDirection = clampedDirection;
    }

    public void ApplyKnockback(Vector2 knockback)
    {
        Debug.Log($"ApplyKnockback => {name}");
        IsKnockedBack = true;
        this.physics.AddVelocity(knockback + (new Vector2(Mathf.Cos(KnockbackIndirectionDirection), Mathf.Sin(KnockbackIndirectionDirection))) * (KnockbackIndirectionPower * knockback.magnitude * 0.98f)); // multipler 0.98
    }

    public struct SaveData
    {
        public float health;
        public Vector2 pos;
        public Vector3 localScale;
        public Vector2 velocity;
    }

    private void Awake()
    {
        Debug.Log(name + "Controller is awaken");
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
        characterData = GetComponent<CharacterData>();
        physics.Initialize(characterData);
        PreviewScale = transform.localScale;
        previewController = GetComponentInChildren<PreviewController>();
    }

    private void Update()
    {
        physics.DetectGround();
    }

    public virtual void Start()
    {
        if (previewController != null)
        {
            previewController.Initialize(characterData);
        }
        else
        {
            Debug.LogWarning("PreviewController not found on character children");
        }
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

    public virtual void Load(SaveData savedata)
    {
        this.physics.setPosition(savedata.pos.x, savedata.pos.y);
        this.physics.setVelocity(savedata.velocity.x, savedata.velocity.y);
        this.characterData.health = savedata.health;
        this.transform.localScale = savedata.localScale;
        this.IsKnockedBack = false;
        this.IsBusy = false;
        physics.DetectGround();
        HideMovePreview();
    }

    private void onFrameEvent(FrameEvent frameEvent)
    {
        switch (frameEvent.type)
        {
            case FrameEventType.SpawnHitbox:
                SpawnHitbox(frameEvent.hitboxData);
                break;
            case FrameEventType.Jump:
                this.physics.AddVelocity(new Vector2(Mathf.Cos(JumpDirection), Mathf.Sin(JumpDirection)) * JumpPower * 0.24f);
                break;
            case FrameEventType.SpawnVFX:
                // read from data later
                break;
            case FrameEventType.SpawnSFX:
                // read from data later
                break;
            case FrameEventType.Block:
                this.blocking = true;
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
                    target.ApplyKnockback(new Vector2(data.knockback.x * facing, data.knockback.y));
                    target.Damage(data.damage);
                },
                minX, minY, maxX, maxY
            )
        );
    }

    public bool CanUseMove(AnimationData moveData)
    {
        if (moveData.moveId == "idle") return true;
        if (moveData == null) return false;
        if (IsBusy) return false;
        if (moveData.onlyGrounded && !IsGrounded) return false;
        if (!moveData.usableInKnockedback && IsKnockedBack) return false;
        return true;
    }

    public void ResetMove()
    {
        SelectedMove = CONTINUE;
    }

    public void ResetPreviewScale()
    {
        PreviewScale = transform.localScale;
    }

    public bool TrySelectMove(string moveId)
    {
        if (moveId == CONTINUE)
        {
            SelectedMove = CONTINUE;
            return true;
        }
        if (string.IsNullOrEmpty(moveId))
        {
            SelectedMove = CONTINUE;
            return false;
        }

        var move = characterData.GetMove(moveId);
        if (move == null)
        {
            SelectedMove = CONTINUE;
            return false;
        }
        if (!CanUseMove(move))
        {
            SelectedMove = CONTINUE;
            return false;
        }

        SelectedMove = moveId;
        Debug.Log($"Move selected: {moveId}");
        return true;
    }

    public void SelectMove(string moveId)
    {
        TrySelectMove(moveId);
    }

    public void Flip(bool flipped, bool previewOnly = false)
    {
        if(flipped != PreviewScale.x < 0)
            PreviewScale -= new Vector3(PreviewScale.x * 2, 0, 0);
        if (!previewOnly)
        {
            if(flipped != transform.localScale.x < 0)
                transform.localScale -= new Vector3(transform.localScale.x * 2, 0, 0);
        }
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
        if(this.SelectedMove != "block")
            this.blocking = false;
        anim.Step();
        physics.Step();
        HitboxManager.Instance.SubmitHurtBox(this.getHurtBox());

        var velocity = physics.getVelocity();
        if (IsKnockedBack && velocity.magnitude < 0.05f && IsGrounded)
            IsKnockedBack = false;
        Debug.Log($"{name} Step — vel={physics.getVelocity()}, IsGrounded={IsGrounded}, IsKnockedBack={IsKnockedBack}, IsBusy={IsBusy}");
    }

    public void ExecuteMove(string moveId, System.Action onComplete = null)
    {
        if (moveId == CONTINUE) return;
        if (moveId != "idle") IsBusy = true;
        anim.PlayMove(moveId, () =>
        {
            IsBusy = false;
            onComplete?.Invoke();
        });

        var move = characterData.GetMove(moveId);
        if (move != null && !move.continuousImpulse) // if the move has an impulse and isn't continuous, apply it immediately. if it's continuous, the impulse will be applied in the CharacterAnimation's Step function.
            physics.ApplyImpulse(move.impulse);
    }

    public void ShowMovePreview(AnimationData moveData)
    {
        PreviewManager.Instance.RestartAllPreviews();
        previewController?.StopPreview();
        previewController?.StartPreview(moveData, this);
    }

    public void ResumePreview()
    {
        PreviewManager.Instance.RestartAllPreviews();
        previewController?.StopPreview();
        if (anim.HasActiveMove)
        {
            previewController?.StartPreview(anim.CurrentMove, this, anim.CurrentFrameIndex, true);
            return;
        }

        foreach (var move in characterData.animations)
        {
            if (move.moveId == "idle")
            {
                previewController?.StartPreview(move, this);
                return;
            }
        }
    }

    public void HideMovePreview()
    {
        previewController?.StopPreview();
    }
}