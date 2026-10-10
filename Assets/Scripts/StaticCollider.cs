using UnityEngine;

public class StaticCollider : PhysicsCollider
{
    /// <summary>
    /// The stage box, built ONCE and cached.
    ///
    /// A static collider cannot move (setPosition is deliberately a no-op), so its box is constant
    /// after the transform is authored. It was rebuilt on every query - once per collider pair per
    /// sweep pass, per fighter, per frame - even though nothing about it can change. Caching it is
    /// exact, not an approximation.
    /// </summary>
    private AABB _box;

    public override AABB getBoundingBox()
    {
        if (_box == null)
        {
            _box = new AABB(
               this.transform.position.x - this.transform.localScale.x / 2,
               this.transform.position.y - this.transform.localScale.y / 2,
               this.transform.position.x + this.transform.localScale.x / 2,
               this.transform.position.y + this.transform.localScale.y / 2
            );
        }
        return _box;
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