using System;
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
        var velo_copy = velo;
        var beg = collider.getPosition();
        int tries = 0;
        while(tries < 10 && velo.sqrMagnitude > 0.00001)
        {
            var (tar, vert) = this.getClampedPosition(collider, velo);
            beg += tar;
            var collided = (velo - tar).sqrMagnitude > 0.00001;
            velo -= tar;
            if (collided)
            {
                if (vert)
                {
                    velo_copy.x *= -0.5f;
                    velo.x *= -0.5f;
                }
                else
                {
                    velo_copy.y *= -0.5f;
                    velo.y *= -0.5f;
                }
            }
            collider.setPosition(beg.x, beg.y);
            ++tries;
        }
        velo = velo_copy;
        if (collider.hasGravity())
        {
            velo.y -= GRAVITY;
        }
        collider.setVelocity(velo.x, velo.y);
    }

    private (Vector2, bool) getClampedPosition(PhysicsCollider collider, Vector2 velo)
    {
        bool vert = false;
        AABB aabb = collider.getBoundingBox();
        if (aabb == null || velo.sqrMagnitude == 0)
            return (Vector2.zero, false);
        float dist = float.PositiveInfinity;
        foreach (PhysicsCollider otherCollider in this.physicsObjects.getList())
        {
            if (collider == otherCollider)
                continue;
            AABB aabb2 = otherCollider.getBoundingBox();
            if (aabb2 == null)
                continue;
            var res = aabb.sweep(aabb2, velo.x, velo.y);
            if (res.Distance < dist)
            {
                dist = res.Distance;
                vert = res.HitVertical;
            }
        }
        return (velo.normalized * Mathf.Min(velo.magnitude, Mathf.Max(0, dist - 0.001f)), vert);
    }
}