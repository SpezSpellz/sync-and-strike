
using UnityEngine;

public abstract class PhysicsCollider : MonoBehaviour
{
    int colliderId = -1;
    public virtual void Start()
    {
        this.colliderId = PhysicsManager.Instance.RegisterCollider(this);
    }

    public int getId()
    {
        return this.colliderId;
    }

    // Do NOT call this manually
    // Use PhysicsManager.Instance. RegisterCollider / UnRegisterCollider
    // to automatically assign id.
    public void setId(int colliderId)
    {
        this.colliderId = colliderId;
    }

    public abstract AABB getBoundingBox();

    public abstract bool hasGravity();
    public abstract Vector2 getVelocity();

    public abstract void setVelocity(float x, float y);

    public abstract Vector2 getPosition();

    public abstract void setPosition(float x, float y);

    public abstract void Step();
}