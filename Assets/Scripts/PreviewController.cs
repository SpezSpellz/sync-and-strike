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
    private int startFrame;
    private bool resume;
    private int frameIndex;
    private float frameTimer;
    private float previewTimer;
    private bool active;
    private bool processedEventsThisCycle;
    private CharacterController owner;
    public CharacterController Owner => owner;
    private const float FRAME_TIME = 1f / 60f; // 60 fps
    private const float PREVIEW_LIFETIME = 3f; // preview stays alive for 3 seconds

    private void Awake()
    {
        previewPhysics = GetComponent<CharacterPhysics>();
        previewPhysics.skipPhysicsManagerRegistration = true;
    }

    public void Initialize(CharacterData characterData)
    {
        data = characterData;
        previewPhysics.Initialize(characterData, true);
        previewRenderer.color = new Color(0f, 0f, 0f, 0.35f);
        previewRenderer.gameObject.SetActive(true);
        transform.SetParent(null);
        foreach (var collider in physicsObjects)
        {
            // Register real world object that preview can collide to
            PreviewPhysicsManager.Instance.Register(collider);
        }
    }

    private void Preview()
    {
        if (owner == null || moveData == null)
            return;

        frameIndex = Mathf.Clamp(startFrame, 0, moveData.frames.Length - 1);
        frameTimer = 0f;
        previewTimer = 0f;
        processedEventsThisCycle = false;
        active = true;
        previewRenderer.gameObject.SetActive(true);
        previewRenderer.sprite = moveData.frames[frameIndex];

        // Reset preview physics to owner start
        Vector2 ownerPosition = owner.GetPosition();
        Vector2 ownerVelocity = owner.GetVelocity();
        transform.position = ownerPosition;
        transform.localScale = owner.PreviewScale;
        previewPhysics.setPosition(ownerPosition.x, ownerPosition.y);
        previewPhysics.setVelocity(ownerVelocity.x, ownerVelocity.y);
        previewPhysics.DetectGround();
        if (moveData != null && !moveData.continuousImpulse && !resume) // if the move has an impulse and isn't continuous, apply it immediately. if it's continuous, the impulse will be applied in the CharacterAnimation's Step function.
            previewPhysics.ApplyImpulse(moveData.impulse);
    }

    public void StartPreview(AnimationData animationData, CharacterController owner, int startFrame = 0, bool resume = false)
    {
        this.resume = resume;
        this.startFrame = startFrame;
        this.owner = owner;
        moveData = animationData;
        
        if (owner == null || moveData == null || moveData.frames == null || moveData.frames.Length == 0)
        {
            Debug.LogWarning("Cannot start preview: invalid move data or owner.");
            return;
        }

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

    public void Step(float deltaTime = 1f / 60f)
    {
        if (!active || moveData == null) return;

        previewTimer += deltaTime;

        // Restart after 5s instead of frame-cycle completion
        if (previewTimer >= PREVIEW_LIFETIME)
        {
            previewTimer = 0f;
            PreviewManager.Instance.RestartAllPreviews();
            return;
        }

        frameTimer += deltaTime;
        if (frameTimer >= FRAME_TIME)
        {
            frameTimer -= FRAME_TIME;
        
            if (moveData.frames.Length == 1)
            {
                if (!processedEventsThisCycle)
                {
                    ProcessPreviewFrameEvents(frameIndex);
                    processedEventsThisCycle = true;
                }
            }
            else
            {
                int previousFrame = frameIndex;
                int lastFrame = moveData.frames.Length - 1;
                if (frameIndex < lastFrame)
                {
                    frameIndex++;
                    previewRenderer.sprite = moveData.frames[frameIndex];
                    if (frameIndex != previousFrame)
                    {
                        ProcessPreviewFrameEvents(frameIndex);
                    }
                }
                else
                {
                    // Hold on the final frame until lifetime expires
                    frameIndex = lastFrame;
                    previewRenderer.sprite = moveData.frames[frameIndex];
                }
            }
        }

        previewPhysics.DetectGround();

        bool hasJumpEventThisFrame = false;
        if (moveData.events != null)
        {
            foreach (var e in moveData.events)
            {
                if (e.frame == frameIndex && e.type == FrameEventType.Jump)
                {
                    hasJumpEventThisFrame = true;
                    break;
                }
            }
        }
        
        if (moveData.continuousImpulse && !hasJumpEventThisFrame)
        {
            previewPhysics.ApplyImpulse(moveData.impulse);
        }

        previewPhysics.Step();
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
                            Mathf.Cos(owner.PreviewJumpDirection),
                            Mathf.Sin(owner.PreviewJumpDirection)
                        ) * owner.PreviewJumpPower * 0.24f
                    );
                    break;
            }
        }
    }

    private void SpawnPreviewHitbox(HitboxData data)
    {
        if (data.damage == 0 && data.knockback == Vector2.zero) return;
        if (data.width <= 0f || data.height <= 0f) return;

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
        if (knockback == Vector2.zero)
            return;

        Vector2 adjustedKnockback = knockback;

        if (owner != null)
        {
            Vector2 diVector = new Vector2(
                Mathf.Cos(owner.PreviewKnockbackIndirectionDirection),
                Mathf.Sin(owner.PreviewKnockbackIndirectionDirection)
            );

            adjustedKnockback += diVector * (owner.PreviewKnockbackIndirectionPower * knockback.magnitude * 0.98f);
        }

        previewPhysics.AddVelocity(adjustedKnockback);
    }
}