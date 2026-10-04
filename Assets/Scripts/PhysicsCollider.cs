using UnityEngine;

public abstract class PhysicsCollider : MonoBehaviour
{
    int colliderId = -1;
    public bool skipPhysicsManagerRegistration = false;
    public virtual void Start()
    {
        if (!skipPhysicsManagerRegistration)
        {
            var arena = GetComponentInParent<Arena>();
            var pm = arena != null ? arena.physicsManager : PhysicsManager.Instance;
            this.colliderId = pm.RegisterCollider(this);
        }
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

    /// <summary>
    /// Whether this collider should physically push <paramref name="other"/> around.
    /// Allies pass through each other (the companion must never shove the player).
    /// </summary>
    public virtual bool CollidesWith(PhysicsCollider other)
    {
        return true;
    }

    public abstract void Step();
}