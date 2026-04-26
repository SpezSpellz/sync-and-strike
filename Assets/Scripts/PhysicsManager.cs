using UnityEngine;

public class PhysicsManager : MonoBehaviour
{
    const float GRAVITY = .2f * 0.016f;
    public static PhysicsManager Instance { get; private set; }
    private IndexSet<PhysicsCollider> physicsObjects = new();
    private void Awake()
    {
        Instance = this;
    }

    public int RegisterCollider(PhysicsCollider collider)
    {
        if (collider.getId() != -1)
            return collider.getId();
        return physicsObjects.add(collider);
    }

    public void UnRegisterCollider(PhysicsCollider collider)
    {
        if (collider.getId() == -1)
            return;
        physicsObjects.remove(collider.getId());
        collider.setId(-1);
    }

    public void StepFor(PhysicsCollider collider)
    {
        Vector2 velo = collider.getVelocity();
        var beg = collider.getPosition();
        var tar = this.getClampedPosition(collider, velo);
        beg += tar;
        collider.setPosition(beg.x, beg.y);
        velo -= tar;
        tar = this.getClampedPosition(collider, new Vector2(velo.x, 0));
        beg += tar;
        collider.setPosition(beg.x, beg.y);
        velo -= tar;
        tar = this.getClampedPosition(collider, new Vector2(0, velo.y));
        collider.setPosition(beg.x + tar.x, beg.y + tar.y);
        velo = collider.getVelocity();
        if (collider.hasGravity())
        {
            velo.y -= GRAVITY;
        }
        collider.setVelocity(velo.x, velo.y);
    }

    private Vector2 getClampedPosition(PhysicsCollider collider, Vector2 velo)
    {
        AABB aabb = collider.getBoundingBox();
        if (aabb == null || velo.sqrMagnitude == 0)
            return Vector2.zero;
        float dist = float.PositiveInfinity;
        foreach (PhysicsCollider otherCollider in this.physicsObjects.getList())
        {
            if (collider == otherCollider)
                continue;
            AABB aabb2 = otherCollider.getBoundingBox();
            if (aabb2 == null)
                continue;
            float cur_dist = aabb.sweep(aabb2, velo.x, velo.y);
            if (cur_dist < dist)
                dist = cur_dist;
        }
        return velo.normalized * Mathf.Min(velo.magnitude, Mathf.Max(0, dist - 0.01f));
    }
}