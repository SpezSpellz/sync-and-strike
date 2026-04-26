using UnityEngine;

public class CharacterPhysics : PhysicsCollider
{

    private CharacterData characterData;
    private float veloX = 0.0f;
    private float veloY = 0.0f;

    public void Initialize(CharacterData characterData)
    {
        this.characterData = characterData;
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Vector2 direction = FacingDirection();
        veloX += impulse.x * direction.x;
        veloY += impulse.y;
    }

    public void ApplyKnockback(Vector2 knockback)
    {
        veloX += knockback.x;
        veloY += knockback.y;
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
        PhysicsManager.Instance.StepFor(this);
    }

    private Vector2 FacingDirection()
    {
        return transform.localScale.x > 0 ? Vector2.right : Vector2.left;
    }
}