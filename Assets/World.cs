using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class World : MonoBehaviour
{
    private const float SECONDS_PER_FRAME = 0.016f;
    private IndexSet<Entity> entities = new IndexSet<Entity>();
    private float ticksAwaiting = 0;
    public static World Instance { get; private set; }

    public int addEntity(Entity entity)
    {
        if(entity.getId() != -1)
            return entity.getId();
        return entities.add(entity);
    }

    public List<Entity> getEntities()
    {
        return this.entities.getList();
    }

    void Awake()
    {
        if (Instance != null)
        {
            Debug.LogError("There is more than one instance!");
            return;
        }
        Instance = this;
    }

    private void Update()
    {
        ticksAwaiting += Time.deltaTime;
        while(ticksAwaiting > 0)
        {
            ticksAwaiting -= SECONDS_PER_FRAME;
            Step();
        }
    }

    public Player getPlayer()
    {
        return (Player)this.entities.getList().First((x) => x is Player);
    }

    private void Step()
    {
        foreach (Entity entity in this.entities.getList())
        {
            entity.Step();
        }
        this.StepPhysics();
    }

    const float GRAVITY = .2f * 0.016f;

    public void StepPhysics()
    {
        foreach (Entity entity in this.getEntities())
        {
            Vector2 velo = entity.getVelocity();
            var beg = entity.getPosition();
            var tar = this.getClampedPosition(entity, velo);
            beg += tar;
            entity.setPosition(beg.x, beg.y);
            velo -= tar;
            tar = this.getClampedPosition(entity, new Vector2(velo.x, 0));
            beg += tar;
            entity.setPosition(beg.x, beg.y);
            velo -= tar;
            tar = this.getClampedPosition(entity, new Vector2(0, velo.y));
            entity.setPosition(beg.x + tar.x, beg.y + tar.y);
            velo = entity.getVelocity();
            if (entity.hasGravity())
            {
                velo.y -= GRAVITY;
            }
            entity.setVelocity(velo.x, velo.y);
        }
    }

    private Vector2 getClampedPosition(Entity entity, Vector2 velo)
    {
        AABB aabb = entity.getBoundingBox();
        if (aabb == null || velo.sqrMagnitude == 0)
            return Vector2.zero;
        float dist = float.PositiveInfinity;
        foreach (Entity entity2 in this.entities.getList())
        {
            if (entity == entity2)
                continue;
            AABB aabb2 = entity2.getBoundingBox();
            if (aabb2 == null)
                continue;
            float cur_dist = aabb.sweep(aabb2, velo.x, velo.y);
            if (cur_dist < dist)
                dist = cur_dist;
        }
        return velo.normalized * Mathf.Min(velo.magnitude, Mathf.Max(0, dist - 0.01f));
    }
}
