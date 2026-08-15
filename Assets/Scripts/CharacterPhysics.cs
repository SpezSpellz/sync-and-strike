using UnityEngine;

public class CharacterPhysics : PhysicsCollider
{
    [SerializeField]
    private float rayLength = 0.5f;
    [SerializeField]
    private LayerMask ground;
    private CharacterData characterData;
    private float veloX = 0.0f;
    private float veloY = 0.0f;
    private bool isPreview = false;
    public bool IsGrounded { get; private set; } = true;

    public void Initialize(CharacterData characterData, bool isPreview = false)
    {
        this.characterData = characterData;
        this.isPreview = isPreview;
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Vector2 direction = FacingDirection();
        veloX += impulse.x * direction.x;
        veloY += impulse.y;
    }

    public void AddVelocity(Vector2 velocity)
    {
        veloX += velocity.x;
        veloY += velocity.y;
    }

    public override AABB getBoundingBox()
    {
        return new AABB(
            transform.position.x - this.characterData.width / 2,
            transform.position.y - this.characterData.height / 2,
            transform.position.x + this.characterData.width / 2,
            transform.position.y + this.characterData.height / 2
        );
    }

    public override Vector2 getPosition()
    {
        return new Vector2(transform.position.x, transform.position.y);
    }

    public override Vector2 getVelocity()
    {
        return new Vector2(veloX, veloY);
    }

    public override bool hasGravity()
    {
        return true;
    }

    public override void setPosition(float x, float y)
    {
        transform.position = new Vector3(x, y, transform.position.z);
    }

    public override void setVelocity(float x, float y)
    {
        veloX = x;
        veloY = y;
    }

    public override void Step()
    {
        if (isPreview)
        {
            PreviewPhysicsManager.Instance.StepFor(this);
        }
        else
        {
            PhysicsManager.Instance.StepFor(this);
        }
        this.veloX *= 0.8f;
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawLine(transform.position, new Vector2(transform.position.x, transform.position.y - rayLength));
    }

    public void DetectGround()
    {
        IsGrounded = Physics2D.Raycast(transform.position, Vector2.down, rayLength, ground);
    }

    public Vector2 FacingDirection()
    {
        return transform.localScale.x > 0 ? Vector2.right : Vector2.left;
    }
}