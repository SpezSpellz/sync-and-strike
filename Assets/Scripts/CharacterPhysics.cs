using UnityEngine;

public class CharacterPhysics : MonoBehaviour
{
    private Rigidbody2D rb;

    [SerializeField] private AnimationData[] animations;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    public void ApplyMoveImpulse(string moveId)
    {
        AnimationData data = FindMove(moveId);
        if (data == null) return;

        Vector2 direction = FacingDirection();
        Vector2 force = new Vector2(data.impulse.x * direction.x, data.impulse.y);
        rb.AddForce(force, ForceMode2D.Impulse);
    }

    public void ApplyKnockback(Vector2 knockback)
    {
        rb.AddForce(knockback, ForceMode2D.Impulse);
    }

    private AnimationData FindMove(string moveId)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId) return anim;
        }
        Debug.LogWarning($"No AnimationData found for moveId: {moveId}");
        return null;
    }

    private Vector2 FacingDirection()
    {
        return transform.localScale.x > 0 ? Vector2.right : Vector2.left;
    }
}