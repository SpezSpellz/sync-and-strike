using UnityEngine;
using UnityEngine.LowLevelPhysics2D;

public abstract class Entity : MonoBehaviour
{
    int entityId = -1;
    public virtual void Start()
    {
        this.entityId = World.Instance.addEntity(this);
    }

    public int getId()
    {
        return this.entityId;
    }

    public abstract AABB getBoundingBox();

    public abstract bool hasGravity();
    public abstract Vector2 getVelocity();

    public abstract void setVelocity(float x, float y);

    public abstract Vector2 getPosition();

    public abstract void setPosition(float x, float y);

    public abstract void Step();
}
