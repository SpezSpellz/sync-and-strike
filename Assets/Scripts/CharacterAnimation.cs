using System;
using UnityEngine;

public class CharacterAnimation : MonoBehaviour
{
    private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;
    private Action<FrameEvent> onFrameEvent;
    private CharacterPhysics physics;

    private AnimationData current;
    public AnimationData CurrentMove => current;
    public bool HasActiveMove => current != null;
    private int currentFrame;
    public int CurrentFrameIndex => currentFrame;
    private int frameCounter = 1;
    private Action onAnimationComplete;

    private void Awake()
    {
        Debug.Log($"SpriteRenderer: {spriteRenderer}");
    }

    public void Initialize(AnimationData[] data, Action<FrameEvent> onFrameEvent, CharacterPhysics physics)
    {
        animations = data;
        this.onFrameEvent = onFrameEvent;
        this.physics = physics;
    }

    public void Step()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Simulating) return; // If the player is still choosing, don't advance the frame.
        if (current == null) return; // If no animation is loaded, don't advance the frame. SAFE GUARD SINCE IF ANIMATIONDATA ISN'T LOADED PROPERLY UNITY WILL BREAK
        if (current.frames.Length == 0) return; // If an animationData exists but has no sprites in the frames array.

        if (current.continuousImpulse)
        {
            physics.ApplyImpulse(current.impulse);
        }

        Debug.Log($"Update — current: {current.moveId}, frame: {currentFrame}/{current.frames.Length}, frameCounter: {frameCounter}/{current.firstActionable}");
        AdvanceFrame();
    }

    public void PlayMove(string moveId, Action onComplete = null)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId)
            {
                current = anim;
                currentFrame = 0;
                Debug.Log($"Playing {moveId} — {current.frames.Length} frames");
                onAnimationComplete = onComplete;
                return;
            }
        }
        Debug.LogWarning($"No animation found for moveId: {moveId}");
    }

    private void AdvanceFrame()
    {

        if (currentFrame >= current.frames.Length) // If we've reached the end of the animation
        {
            if (current.loop)
            {
                // check firstActionable on loop completion
                if (frameCounter >= current.firstActionable)
                {
                    current = null;
                    frameCounter = 1;
                    onAnimationComplete?.Invoke();
                    onAnimationComplete = null;
                    return;
                }
                frameCounter++;
                currentFrame = 0;
            }   
            else if (frameCounter < current.firstActionable) // If we've reached the end of a non-looping animation but haven't hit the first actionable frame, dont advance
            {
                frameCounter++;
                return; // return without going to the next frame (for block)
            }
            else
            {
                frameCounter = 1;
                onAnimationComplete?.Invoke();
                onAnimationComplete = null;
                PlayMove("idle");
                return;
            }
        }

        // Should just be here right away in normal cases.
        spriteRenderer.sprite = current.frames[currentFrame];

        if (current.events != null)
        {
            foreach (var e in current.events)
            {
                if (e.frame == currentFrame)
                    this.onFrameEvent(e); // pass full event data so we can use hitboxData for SpawnHitbox events
            }
        }

        currentFrame++;
    }

    public void ResetToIdle()
    {
        current = null;
        currentFrame = 0;
    }
}