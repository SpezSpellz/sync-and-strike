using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(CharacterPhysics))]
public class PreviewController : MonoBehaviour
{
    [SerializeField] private SpriteRenderer previewRenderer;
    [SerializeField] private List<PhysicsCollider> physicsObjects;
    private CharacterPhysics previewPhysics;
    private CharacterData data;
    private AnimationData moveData;
    private int frameIndex;
    private float frameTimer;
    private bool active;
    private CharacterController owner;
    public CharacterController Owner => owner;
    private Vector2 lastVelocity;

    private const float FRAME_TIME = 0.08f;

    private void Awake()
    {
        previewPhysics = GetComponent<CharacterPhysics>();
        previewPhysics.skipPhysicsManagerRegistration = true;
    }

    public void Initialize(CharacterData characterData)
    {
        data = characterData;
        previewPhysics.Initialize(characterData);
        previewRenderer.color = new Color(0f, 0f, 0f, 0.35f);
        previewRenderer.gameObject.SetActive(false);
        foreach (var collider in physicsObjects)
        {
            PreviewPhysicsManager.Instance.Register(collider);
        }
    }

    private void Preview()
    {
        transform.SetParent(null);
        frameIndex = 0;
        frameTimer = 0f;
        active = true;
        previewRenderer.gameObject.SetActive(true);
        previewRenderer.sprite = moveData.frames[frameIndex];

        // ProcessPreviewFrameEvents(frameIndex);
        // SubmitPreviewHurtBox();

        // Reset preview physics to owner start
        Vector2 ownerPosition = owner.GetPosition();
        Vector2 ownerVelocity = owner.GetVelocity();
        transform.position = ownerPosition;
        transform.localScale = owner.transform.localScale;
        previewPhysics.setPosition(ownerPosition.x, ownerPosition.y);
        previewPhysics.setVelocity(ownerVelocity.x, ownerVelocity.y);
        if (moveData != null && !moveData.continuousImpulse) // if the move has an impulse and isn't continuous, apply it immediately. if it's continuous, the impulse will be applied in the CharacterAnimation's Step function.
            previewPhysics.ApplyImpulse(moveData.impulse);
        Debug.Log($"Current velo after impulse but before stepping: {previewPhysics.getVelocity()}");
    }

    public void StartPreview(AnimationData animationData, CharacterController owner)
    {
        this.owner = owner;
        moveData = animationData;
        Preview();
        PreviewManager.Instance.RegisterPreview(this);
        PreviewPhysicsManager.Instance.Register(previewPhysics);
    }

    public void StopPreview()
    {
        active = false;
        moveData = null;
        previewRenderer.gameObject.SetActive(false);
        PreviewManager.Instance.UnregisterPreview(this);
        PreviewPhysicsManager.Instance.Unregister(previewPhysics);
    }

    public void Restart()
    {
        if (owner == null || moveData == null) return;
        Preview();
    }

    public void Step(float deltaTime)
    {
        if (!active || moveData == null) return;

        frameTimer += deltaTime;
        if (frameTimer >= FRAME_TIME)
        {
            frameTimer -= FRAME_TIME;
            frameIndex = (frameIndex + 1) % moveData.frames.Length;
            previewRenderer.sprite = moveData.frames[frameIndex];
            ProcessPreviewFrameEvents(frameIndex);
        }

        // if (moveData.impulse != Vector2.zero)
        // {
        if (moveData.continuousImpulse)
        {
            previewPhysics.ApplyImpulse(moveData.impulse);
        }
        //     else if (frameIndex == 0)
        //     {
        //         previewPhysics.ApplyImpulse(moveData.impulse);
        //     }
        // }

        PreviewPhysicsManager.Instance.StepFor(previewPhysics);
        var currentVelo = previewPhysics.getVelocity();
        previewPhysics.setVelocity(currentVelo.x * 0.8f, currentVelo.y);
        // previewPhysics.Step();
        transform.position = previewPhysics.getPosition();

        SubmitPreviewHurtBox();
    }

    private void SubmitPreviewHurtBox()
    {
        if (data == null) return;
        Vector2 pos = previewPhysics.getPosition();
        PreviewHitboxManager.Instance.SubmitHurtBox(
            new HurtBox(
                owner,
                pos.x - data.width * 0.5f,
                pos.y - data.height * 0.5f,
                pos.x + data.width * 0.5f,
                pos.y + data.height * 0.5f
            )
        );
    }

    private void ProcessPreviewFrameEvents(int frame)
    {
        if (moveData.events == null) return;
        foreach (var e in moveData.events)
        {
            if (e.frame != frame) continue;
    
            switch (e.type)
            {
                case FrameEventType.SpawnHitbox:
                    SpawnPreviewHitbox(e.hitboxData);
                    break;
                case FrameEventType.Jump:
                    previewPhysics.AddVelocity(
                        new Vector2(
                            Mathf.Cos(owner.JumpDirection),
                            Mathf.Sin(owner.JumpDirection)
                        ) * owner.JumpPower * 0.24f
                    );
                    break;
            }
        }
    }

    private void SpawnPreviewHitbox(HitboxData data)
    {
        float facing = previewPhysics.FacingDirection().x;
        Vector2 pos = previewPhysics.getPosition();

        float minX = pos.x + facing * data.offsetX - data.width * 0.5f;
        float maxX = pos.x + facing * data.offsetX + data.width * 0.5f;
        float minY = pos.y + data.offsetY - data.height * 0.5f;
        float maxY = pos.y + data.offsetY + data.height * 0.5f;

        PreviewHitboxManager.Instance.SubmitHitBox(
            new HitBox(
                owner,
                (target) =>
                {
                    var targetPreview = PreviewManager.Instance.GetPreviewByOwner(target);
                    if (targetPreview != null)
                        targetPreview.ApplyPreviewKnockback(
                            new Vector2(data.knockback.x * facing, data.knockback.y)
                        );
                },
                minX, minY, maxX, maxY
            )
        );
    }

    public void ApplyPreviewKnockback(Vector2 knockback)
    {
        previewPhysics.AddVelocity(knockback);
    }

    public bool IsFinishedOneCycle => frameIndex == moveData.frames.Length - 1;
}