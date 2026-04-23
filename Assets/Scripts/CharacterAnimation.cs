using UnityEngine;

public class CharacterAnimation : MonoBehaviour
{
    [SerializeField] private AnimationData[] animations;
    [SerializeField] private SpriteRenderer spriteRenderer;
    [SerializeField] private HitboxController hitbox;

    private AnimationData current;
    private int currentFrame;
    private float timer;

    private void Update()
    {
        if (current == null || current.frames.Length == 0) return;

        timer += Time.deltaTime;

        if (timer >= current.frameTime)
        {
            timer -= current.frameTime;
            AdvanceFrame();
        }
    }

    public void PlayMove(Move move)
    {
        foreach (var anim in animations)
        {
            if (anim.move == move)
            {
                current = anim;
                currentFrame = 0;
                timer = 0f;
                return;
            }
        }
    }

    private void AdvanceFrame()
    {
        if (current.events != null)
        {
            foreach (var e in current.events)
            {
                if (e.frame == currentFrame)
                    HandleEvent(e.type);
            }
        }

        spriteRenderer.sprite = current.frames[currentFrame];
        currentFrame++;

        if (currentFrame >= current.frames.Length)
        {
            if (current.loop)
                currentFrame = 0;
            else
                current = null; // animation done, wait for next round
        }
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