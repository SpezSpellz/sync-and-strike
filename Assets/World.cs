using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    private const float SECONDS_PER_FRAME = 0.016f;
    private List<Entity> entities = new List<Entity>();
    private HashSet<Entity> entitiesSet = new HashSet<Entity>();
    private float ticksAwaiting = 0;
    public static World Instance { get; private set; }

    public void addEntity(Entity entity)
    {
        if(entitiesSet.Contains(entity)) return;
            entities.Add(entity);
    }

    public List<Entity> getEntities()
    {
        return entities;
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

    private void Step()
    {
        foreach (Entity entity in entities)
        {
            entity.Step();
        }
    }
}
