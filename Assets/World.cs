using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    private const float SECONDS_PER_FRAME = 0.016f;
    private IndexSet<Entity> entities = new IndexSet<Entity>();
    private float ticksAwaiting = 0;
    public static World Instance { get; private set; }

    public int addEntity(Entity entity)
    {
        if (entity.getId() != -1)
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
        while (ticksAwaiting > 0)
        {
            ticksAwaiting -= SECONDS_PER_FRAME;
            Step();
        }
    }

    private void Step()
    {
        foreach (Entity entity in this.entities.getList())
        {
            entity.Step();
        }
    }

    const float GRAVITY = .2f * 0.016f;

}
