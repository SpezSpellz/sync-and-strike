using UnityEngine;

public class CharacterPhysics : MonoBehaviour
{
    private Rigidbody2D rb;

    [SerializeField] private float punchForce = 6f;
    [SerializeField] private float kickForce = 9f;
    [SerializeField] private float dodgeForce = 5f;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    public void ApplyMoveImpulse(Move move)
    {
        Vector2 direction = FacingDirection();

        switch (move)
        {
            case Move.Punch:
                rb.AddForce(direction * punchForce, ForceMode2D.Impulse);
                break;
            case Move.Kick:
                rb.AddForce(direction * kickForce, ForceMode2D.Impulse);
                break;
            case Move.Dodge:
                rb.AddForce(-direction * dodgeForce, ForceMode2D.Impulse); // dodge away
                break;
        }
    }

    public void ApplyKnockback(Vector2 knockback)
    {
        rb.AddForce(knockback, ForceMode2D.Impulse);
    }

    private Vector2 FacingDirection()
    {
        // Returns +1 or -1 on X based on sprite facing
        return transform.localScale.x > 0 ? Vector2.right : Vector2.left;
    }
}