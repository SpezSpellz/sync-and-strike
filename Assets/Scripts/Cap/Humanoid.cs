using UnityEngine;
using UnityEngine.LowLevelPhysics2D;

public class Humanoid : Entity
{

    protected BaseMove current_move = null;
    protected float veloX = 0;
    protected float veloY = 0;

    public override AABB getBoundingBox()
    {
        return new AABB(
            transform.position.x - 0.5f,
            transform.position.y - 0.7f,
            transform.position.x + 0.5f,
            transform.position.y + 0.7f
       );
    }

    public override Vector2 getVelocity()
    {
        return new Vector2(this.veloX, this.veloY);
    }

    public override bool hasGravity()
    {
        return true;
    }

    public override void setVelocity(float x, float y)
    {
        this.veloX = x;
        this.veloY = y;
    }

    public override void Start()
    {
        base.Start();
        current_move = Moves.Instance.getWalkMove();
    }

    public bool isOnGround()
    {
        /*
        var self_aabb = this.getBoundingBox();
        foreach (Entity entity in World.Instance.getEntities())
        {
            AABB aabb = entity.getBoundingBox();
            if (aabb == null || entity == this)
                continue;
            var dist = self_aabb.sweep(aabb, 0, -0.01f);
            if (dist < 1)
                return true;
        }
        */
        return false;
    }

    public override void Step()
    {
        if (current_move != null)
        {
            current_move.update(this);
            if(current_move.isEnded())
                current_move = Moves.Instance.getWalkMove();
        }
    }

    public override Vector2 getPosition()
    {
        return new Vector2(this.transform.position.x, this.transform.position.y);
    }

    public override void setPosition(float x, float y)
    {
        this.transform.position = new Vector3(x, y, this.transform.position.z);
    }
}
