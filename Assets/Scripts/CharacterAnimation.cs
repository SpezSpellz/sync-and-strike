using UnityEngine;

public class CharacterAnimation : MonoBehaviour
{
    private CharacterData characterData;

    private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;
    [SerializeField] private HitboxController hitbox;

    private AnimationData current;
    private int currentFrame;

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
        if (current == null || current.frames.Length == 0) return;
        Debug.Log($"Update — current: {current.moveId}, frame: {currentFrame}/{current.frames.Length}");
        AdvanceFrame();
    }

    public void PlayMove(string moveId)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId)
            {
                current = anim;
                currentFrame = 0;
                Debug.Log($"Playing {moveId} — {current.frames.Length} frames");
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