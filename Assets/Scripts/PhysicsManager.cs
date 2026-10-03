using System;
using UnityEngine;

public class PhysicsManager : MonoBehaviour
{
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

    public IndexSet<PhysicsCollider> GetRegisteredObjects()
    {
        return this.physicsObjects;
    }

    public void StepFor(PhysicsCollider collider, IndexSet<PhysicsCollider> objects = null)
    {
        Vector2 velo = collider.getVelocity();
        var velo_copy = velo;
        // Captured so hurt states can still see how fast the fighter was travelling when it hit
        // the wall; the collision response below zeroes that axis before they get a chance to
        // read it, which would make the wall rebound impossible.
        var preBlockVelocity = velo_copy;
        var beg = collider.getPosition();
        int tries = 0;
        // touching_wall and the wall rebound are about the stage wall specifically, never about
        // being blocked by the floor or by the other fighter.
        bool blockedByStageWall = false;
        while(tries < 10 && velo.sqrMagnitude > 0.00001)
        {
            var res = this.getClampedPosition(collider, velo, objects);
            beg += res.travel;
            var collided = (velo - res.travel).sqrMagnitude > 0.00001;
            velo -= res.travel;
            if (collided)
            {
                // Zero the blocked axis instead of bouncing the fighter off the wall,
                // so knockback into a corner pins the character against it.
                if (res.vertical)
                {
                    velo_copy.x *= PhysicsConstants.WALL_RESTITUTION;
                    velo.x *= PhysicsConstants.WALL_RESTITUTION;
                }
                else
                {
                    velo_copy.y *= PhysicsConstants.WALL_RESTITUTION;
                    velo.y *= PhysicsConstants.WALL_RESTITUTION;
                }
                blockedByStageWall |= res.blockerIsStageWall;
            }
            collider.setPosition(beg.x, beg.y);
            ++tries;
        }
        velo = velo_copy;
        collider.setVelocity(velo.x, velo.y);

        // HurtGrounded checks touching_wall to suppress further horizontal push, and
        // that flag is a stage-boundary test only, so the floor and the opponent are excluded.
        if (collider is CharacterPhysics characterPhysics)
        {
            characterPhysics.SetWallContact(blockedByStageWall,
                                            blockedByStageWall ? preBlockVelocity : Vector2.zero);
        }
    }

    /// <summary>Outcome of one collision-resolution pass.</summary>
    private struct ClampResult
    {
        public Vector2 travel;      // how far this fighter may actually move this pass
        public bool vertical;       // the blocking face was vertical (a left/right wall)
        public bool blockerIsStageWall;
    }

    private ClampResult getClampedPosition(PhysicsCollider collider, Vector2 velo, IndexSet<PhysicsCollider> objects)
    {
        var result = new ClampResult { travel = Vector2.zero };
        AABB aabb = collider.getBoundingBox();
        if (aabb == null || velo.sqrMagnitude == 0)
            return result;

        float dist = float.PositiveInfinity;
        foreach (PhysicsCollider otherCollider in objects == null ? this.physicsObjects.getList() : objects.getList())
        {
            if (collider == otherCollider)
                continue;
            if (!collider.CollidesWith(otherCollider))
                continue;
            AABB aabb2 = otherCollider.getBoundingBox();
            if (aabb2 == null)
                continue;
            var res = aabb.sweep(aabb2, velo.x, velo.y);
            if (res.Distance < dist)
            {
                dist = res.Distance;
                result.vertical = res.HitVertical;
                // A vertical static collider is the stage wall. A horizontal one is the floor or
                // a platform, which touching_wall never reports.
                bool stageWall = otherCollider is StaticCollider
                    && result.vertical
                    && aabb2.getWidth() < aabb2.getHeight();
                result.blockerIsStageWall = stageWall;
            }
        }

        result.travel = velo.normalized * Mathf.Min(velo.magnitude, Mathf.Max(0, dist - PhysicsConstants.COLLIDER_SKIN));
        return result;
    }
}
