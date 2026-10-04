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
    /// <summary>Simulated frames since the preview started, used to bound it to one move.</summary>
    private int simulatedFrames;
    /// <summary>Counts down while the ghost holds still between preview cycles.</summary>
    private float pauseTimer;
    private bool active;
    private bool processedEventsThisCycle;
    private CharacterController owner;
    public CharacterController Owner => owner;
    [HideInInspector] public Arena Arena;

    private PreviewPhysicsManager PreviewPhysics => Arena != null ? Arena.previewPhysicsManager : PreviewPhysicsManager.Instance;
    private PreviewHitboxManager PreviewHitbox => Arena != null ? Arena.previewHitboxManager : PreviewHitboxManager.Instance;
    private PreviewManager PreviewCtl => Arena != null ? Arena.previewManager : PreviewManager.Instance;

    private const float FRAME_TIME = 1f / 60f; // 60 fps

    /// <summary>
    /// Hard ceiling on how long any preview may simulate, 2 seconds at 60fps. The animation
    /// length bounds the move's own force; this bounds everything, so no move can keep a preview
    /// alive indefinitely no matter what it does.
    /// </summary>
    private const int PREVIEW_MAX_FRAMES = 120;

    /// <summary>
    /// How long the ghost holds its final pose before the preview replays. Without the pause the
    /// loop strobes and the eye cannot follow the move.
    /// </summary>
    private const float REPLAY_PAUSE_SECONDS = 0.5f;

    private void Awake()
    {
        previewPhysics = GetComponent<CharacterPhysics>();
        previewPhysics.skipPhysicsManagerRegistration = true;
        Arena = GetComponentInParent<Arena>();
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
            PreviewPhysics.Register(collider);
        }
    }

    private void Preview()
    {
        if (owner == null || moveData == null)
            return;

        frameIndex = Mathf.Clamp(startFrame, 0, moveData.frames.Length - 1);
        frameTimer = 0f;
        simulatedFrames = 0;
        pauseTimer = 0f;
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
        PreviewCtl.RegisterPreview(this);
        PreviewPhysics.Register(previewPhysics);
    }

    public void StopPreview()
    {
        active = false;
        moveData = null;
        previewRenderer.gameObject.SetActive(false);
        PreviewCtl.UnregisterPreview(this);
        PreviewPhysics.Unregister(previewPhysics);
    }

    public void Restart()
    {
        if (owner == null || moveData == null) return;
        Preview();
    }

    /// <summary>
    /// How many frames of <em>animation</em> this preview plays, which is what bounds the move's own
    /// force.
    ///
    /// This mirrors how long the real state runs: a looping hold plays one cycle
    /// (<c>firstActionable</c>), a committed move plays all of its frames. It is what stops a
    /// looping move's force being applied forever, which made the walk ghost cross the whole stage
    /// and then get teleported back to repeat it. It deliberately does not bound the physics:
    /// jump has a single animation frame, so bounding physics too would cut its arc off instantly.
    /// Total runtime is capped by <see cref="PREVIEW_MAX_FRAMES"/>.
    /// </summary>
    private int PreviewDuration
    {
        get
        {
            if (moveData == null) return 0;
            if (moveData.loop) return Mathf.Max(1, moveData.firstActionable);
            return moveData.frames != null ? moveData.frames.Length : 0;
        }
    }

    public void Step(float deltaTime = 1f / 60f)
    {
        if (!active || moveData == null) return;

        // Between cycles the ghost holds its final pose, then the move replays from the owner's
        // position. This is what makes the preview loop.
        if (pauseTimer > 0f)
        {
            pauseTimer -= deltaTime;
            if (pauseTimer <= 0f) Preview();
            return;
        }

        // The animation plays for the move's authored length, but physics keeps integrating
        // afterwards so a jump still shows its full arc and landing. Only the move's own force
        // stops when the animation ends. Bounding physics to the animation too would have cut a
        // jump off after one frame, since jump's animation is a single frame.
        bool animationDone = simulatedFrames >= PreviewDuration;

        if (!animationDone)
        {
            frameTimer += deltaTime;
            if (frameTimer >= FRAME_TIME)
            {
                frameTimer -= FRAME_TIME;
                AdvancePreviewFrame();
            }

            if (moveData.continuousImpulse && !HasJumpEventThisFrame())
                previewPhysics.ApplyImpulse(moveData.impulse);
        }

        // Step() only integrates the position; friction and gravity are applied explicitly here
        // so the preview matches the real fighter, where the state owns them.
        previewPhysics.DetectGround();
        bool grounded = previewPhysics.IsGrounded;
        previewPhysics.ApplyForces(
            grounded ? PhysicsConstants.GROUND_FRICTION_RATIO : PhysicsConstants.AIR_FRICTION_RATIO,
            PhysicsConstants.GRAVITY,
            grounded);

        previewPhysics.Step();
        transform.position = previewPhysics.getPosition();

        SubmitPreviewHurtBox();
        simulatedFrames++;

        // Close the cycle once the move has played out and the ghost has come to rest, then replay.
        //
        // Re-looping is safe now that each cycle is bounded: the old implementation restarted every
        // 3 seconds via RestartAllPreviews, which reset the ghost to the player and then applied a
        // looping move's continuousImpulse for the entire following cycle, so the walk ghost
        // crossed the whole stage over and over. PREVIEW_MAX_FRAMES still closes the cycle for a
        // ghost that never settles, such as one being knocked around by another preview.
        bool settled = previewPhysics.IsGrounded
                       && Mathf.Abs(previewPhysics.getVelocity().x) < PhysicsConstants.GROUND_SETTLE_SPEED;
        if (animationDone && (settled || simulatedFrames >= PREVIEW_MAX_FRAMES))
            pauseTimer = REPLAY_PAUSE_SECONDS;
    }

    /// <summary>Advances the ghost's sprite one animation frame, holding on the last one.</summary>
    private void AdvancePreviewFrame()
    {
        if (moveData.frames.Length == 1)
        {
            if (!processedEventsThisCycle)
            {
                ProcessPreviewFrameEvents(frameIndex);
                processedEventsThisCycle = true;
            }
            return;
        }

        int previousFrame = frameIndex;
        int lastFrame = moveData.frames.Length - 1;
        if (frameIndex < lastFrame)
        {
            frameIndex++;
            previewRenderer.sprite = moveData.frames[frameIndex];
            if (frameIndex != previousFrame)
                ProcessPreviewFrameEvents(frameIndex);
        }
        else
        {
            frameIndex = lastFrame;
            previewRenderer.sprite = moveData.frames[frameIndex];
        }
    }

    private bool HasJumpEventThisFrame()
    {
        if (moveData.events == null) return false;
        foreach (var e in moveData.events)
            if (e.frame == frameIndex && e.type == FrameEventType.Jump)
                return true;
        return false;
    }

    private void SubmitPreviewHurtBox()
    {
        if (data == null) return;
        Vector2 pos = previewPhysics.getPosition();
        PreviewHitbox.SubmitHurtBox(
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
                    // Use the real jump formula. This used to duplicate it with a hardcoded 0.24
                    // and without the Y modifier or the power curve, so the preview and the
                    // fighter launched at completely different heights and could never agree.
                    previewPhysics.AddVelocity(
                        owner.JumpVelocity(owner.PreviewJumpPower, owner.PreviewJumpDirection, moveData.JumpSpeed));
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

        PreviewHitbox.SubmitHitBox(
            new HitBox(
                owner,
                (target) =>
                {
                    var targetPreview = PreviewCtl.GetPreviewByOwner(target);
                    if (targetPreview != null)
                        targetPreview.ApplyPreviewKnockback(
                            new Vector2(data.knockback.x * facing, data.knockback.y)
                        );
                },
                minX, minY, maxX, maxY
            )
        );
    }

    /// <summary>
    /// Applies the same knockback + DI the real hit reaction would, so the preview trajectory
    /// matches what actually happens.
    ///
    /// DI uses the DiScaling ramp (1.0 to 6.0 with combo count). An earlier version used the
    /// player's stick power times a flat 0.98, which made the preview disagree with the real
    /// result.
    /// </summary>
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

            float scaling = PhysicsConstants.DiScaling(owner.ComboCount);
            bool grounded = previewPhysics.IsGrounded;

            float diStrength = grounded
                ? PhysicsConstants.DI_STRENGTH_GROUNDED
                : PhysicsConstants.DI_STRENGTH_AERIAL;

            Vector2 diForce = diVector
                * (scaling * owner.PreviewKnockbackIndirectionPower * diStrength);
            // A grounded victim only takes horizontal DI.
            if (grounded) diForce.y = 0f;

            adjustedKnockback += diForce;
        }

        previewPhysics.AddVelocity(adjustedKnockback);
    }
}
