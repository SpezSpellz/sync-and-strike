using System;
using UnityEngine;

public class CharacterAnimation : MonoBehaviour
{
    private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;
    private SpriteRenderer previewRenderer;
    private Action<FrameEvent> onFrameEvent;
    private CharacterPhysics physics;

    private AnimationData current;
    private int currentFrame;   
    private int frameCounter = 1;
    private Action onAnimationComplete;

    private AnimationData previewMove;
    private int previewFrame;
    private float previewFrameTimer;
    private const float PREVIEW_FRAME_DURATION = 0.08f;

    private void Awake()
    {
        Debug.Log($"SpriteRenderer: {spriteRenderer}");
        EnsurePreviewRenderer();
    }

    private void EnsurePreviewRenderer()
    {
        if (previewRenderer != null)
            return;

        var previewObj = new GameObject("MovePreview");
        previewObj.transform.SetParent(spriteRenderer.transform, false);

        previewRenderer = previewObj.AddComponent<SpriteRenderer>();
        previewRenderer.sortingLayerID = spriteRenderer.sortingLayerID;
        previewRenderer.sortingOrder = spriteRenderer.sortingOrder - 1;
        previewRenderer.color = new Color(0f, 0f, 0f, 0.35f);
        previewRenderer.gameObject.SetActive(false);
    }

    public void Initialize(AnimationData[] data, Action<FrameEvent> onFrameEvent, CharacterPhysics physics)
    {
        animations = data;
        this.onFrameEvent = onFrameEvent;
        this.physics = physics;
    }

    public void UpdatePreview()
    {
        if (previewMove == null) return;
        if (TurnManager.Instance != null && TurnManager.Instance.Phase != TurnPhase.Planning)
            return;

        if (previewMove.frames == null || previewMove.frames.Length == 0)
            return;

        previewFrameTimer += Time.deltaTime;
        while (previewFrameTimer >= PREVIEW_FRAME_DURATION)
        {
            previewFrameTimer -= PREVIEW_FRAME_DURATION;
            previewFrame = (previewFrame + 1) % previewMove.frames.Length;
            previewRenderer.sprite = previewMove.frames[previewFrame];
        }
    }

    public void ShowPreview(AnimationData moveData)
    {
        previewMove = moveData;
        previewFrame = 0;
        previewFrameTimer = 0f;
        EnsurePreviewRenderer();
        previewRenderer.sprite = previewMove.frames.Length > 0 ? previewMove.frames[0] : null;
        previewRenderer.gameObject.SetActive(true);
    }

    public void HidePreview()
    {
        previewMove = null;
        if (previewRenderer != null)
            previewRenderer.gameObject.SetActive(false);
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
}