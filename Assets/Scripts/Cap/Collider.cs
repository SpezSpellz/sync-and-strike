using UnityEngine;

public class Collider : Entity // "Collider" has the same name as built-in Unity component. Not recommended.
{
    public override AABB getBoundingBox()
    {
        return new AABB(
           this.transform.position.x - this.transform.localScale.x / 2,
           this.transform.position.y - this.transform.localScale.y / 2,
           this.transform.position.x + this.transform.localScale.x / 2,
           this.transform.position.y + this.transform.localScale.y / 2
        );
    }

    public override Vector2 getPosition()
    {
        return new Vector2(this.transform.position.x, this.transform.position.y);
    }

    public override Vector2 getVelocity()
    {
        return Vector2.zero;
    }

    public override bool hasGravity()
    {
        return false;
    }

    public override void setPosition(float x, float y)
    {
        
    }

    public override void setVelocity(float x, float y)
    {
        
    }

    public override void Step()
    {

    }
}