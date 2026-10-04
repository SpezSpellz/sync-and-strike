using System;
using UnityEngine;

/// <summary>
/// Pure sprite/frame player.
///
/// This class deliberately has no physics responsibility. It used to read
/// <see cref="AnimationData.continuousImpulse"/> and push the fighter every frame, which meant a
/// looping move kept applying force even while the fighter was in hitstun and ignored its own
/// knockback. Movement force is now owned by the fighter state instead.
/// </summary>
public class CharacterAnimation : MonoBehaviour
{
    private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;

    private AnimationData current;
    public AnimationData CurrentMove => current;
    public bool HasActiveMove => current != null;
    public int CurrentFrameIndex => currentFrame;

    /// <summary>
    /// Frame events (spawn hitbox, jump burst, guard) are dispatched here but handled by the
    /// controller/state, keeping this class free of gameplay decisions.
    /// </summary>
    public Action<FrameEvent> OnFrameEvent;

    private int currentFrame;
    // Starts at 1 so the first actionable frame can be compared against it.
    private int frameCounter = 1;
    private bool playing;

    private void Awake()
    {
        // Fall back to a renderer on the same GameObject when none was assigned in the inspector.
        // Training spawns fighters from code, where a [SerializeField] reference cannot be authored,
        // so without this every spawned fighter would be invisible.
        if (spriteRenderer == null) spriteRenderer = GetComponent<SpriteRenderer>();
    }

    /// <summary>
    /// Attach a renderer at runtime. Used by the training spawner, which builds fighters in code and
    /// therefore cannot rely on the inspector-assigned reference.
    /// </summary>
    public void SetSpriteRenderer(SpriteRenderer renderer)
    {
        spriteRenderer = renderer;
    }

    public void Initialize(AnimationData[] data)
    {
        animations = data;
    }

    /// <summary>
    /// Advances one frame. Returns true on the frame the animation completes, so the owning
    /// state can transition out. Never applies force.
    /// </summary>
    public bool StepFrame()
    {
        if (current == null) return false;
        if (current.frames == null || current.frames.Length == 0) return false;

        AdvanceFrame();
        return completedThisFrame;
    }

    private bool completedThisFrame;

    public void PlayMove(AnimationData move, bool loop)
    {
        if (move == null || move.frames == null || move.frames.Length == 0)
        {
            current = null;
            playing = false;
            return;
        }
        current = move;
        currentFrame = 0;
        frameCounter = 1;
        playing = true;
        completedThisFrame = false;
        if (spriteRenderer != null && move.frames.Length > 0)
            spriteRenderer.sprite = move.frames[0];
    }

    public void PlayMove(string moveId, Action onComplete = null)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId)
            {
                PlayMove(anim, anim.loop);
                return;
            }
        }
        Debug.LogWarning($"No animation found for moveId: {moveId}");
    }

    private void AdvanceFrame()
    {
        completedThisFrame = false;

        if (currentFrame >= current.frames.Length)
        {
            if (current.loop)
            {
                // Hold on the final frame until the move reaches its first actionable frame.
                if (frameCounter >= current.firstActionable)
                {
                    currentFrame = 0;
                    frameCounter = 1;
                    if (spriteRenderer != null) spriteRenderer.sprite = current.frames[0];
                    return;
                }
                frameCounter++;
                return;
            }

            // Non-looping: hold the last frame as recovery until firstActionable.
            if (frameCounter < current.firstActionable)
            {
                frameCounter++;
                return;
            }

            frameCounter = 1;
            completedThisFrame = true;
            return;
        }

        if (spriteRenderer != null) spriteRenderer.sprite = current.frames[currentFrame];

        // Dispatch frame events on the frame they are authored on. currentFrame is the index of the
        // sprite being shown and has not been incremented yet, so it must be compared directly.
        // Comparing against currentFrame - 1 fired everything one frame late, which meant a
        // 1-frame animation with its event on frame 0 (jump) could never fire at all.
        if (current.events != null && OnFrameEvent != null)
        {
            foreach (var e in current.events)
            {
                if (e.frame == currentFrame) OnFrameEvent(e);
            }
        }

        currentFrame++;
    }

    /// <summary>
    /// Stops the current animation and freezes the sprite on its current frame. Used when a
    /// fighter is interrupted by hitstun, matching the hurt states which have no move of
    /// their own to play.
    /// </summary>
    public void Interrupt()
    {
        playing = false;
        currentFrame = 0;
        frameCounter = 1;
        completedThisFrame = false;
    }

    public void ResetToIdle()
    {
        Interrupt();
        current = null;
    }

    public void PlayIdle()
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == "idle")
            {
                PlayMove(anim, true);
                return;
            }
        }
    }
}