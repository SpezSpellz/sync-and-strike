using UnityEngine;

public class CharacterAnimation : MonoBehaviour
{
    private CharacterData characterData;

    private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;
    [SerializeField] private HitboxController hitbox;

    private AnimationData current;
    private int currentFrame;
    private System.Action onAnimationComplete;

    private void Awake()
    {
        Debug.Log($"SpriteRenderer: {spriteRenderer}");
        characterData = GetComponent<CharacterData>();
    }

    public void Initialize(AnimationData[] data)
    {
        animations = data;
    }

    private void Update()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Simulating) return; // If the player is still choosing, don't advance the frame.
        if (current == null) return; // If no animation is loaded, don't advance the frame. SAFE GUARD SINCE IF ANIMATIONDATA ISN'T LOADED PROPERLY UNITY WILL BREAK
        if (current.frames.Length == 0) return; // If an animationData exists but has no sprites in the frames array.

        Debug.Log($"Update — current: {current.moveId}, frame: {currentFrame}/{current.frames.Length}");
        AdvanceFrame();
    }

    public void PlayMove(string moveId, System.Action onComplete = null)
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
        if (currentFrame >= current.frames.Length)
        {
            if (current.loop)
                currentFrame = 0;
            else
            {
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
                    HandleEvent(e.type);
            }
        }

        // only advance if not on last frame of a looping animation with 1 frame
        if (current.loop && currentFrame == current.frames.Length - 1)
            return;

        currentFrame++;
    }

    private void HandleEvent(FrameEventType type)
    {
        switch (type)
        {
            case FrameEventType.ActivateHitbox:
                hitbox?.Activate();
                break;
            case FrameEventType.DeactivateHitbox:
                hitbox?.Deactivate();
                break;
        }
    }
}