using System.Collections.Generic;
using UnityEngine;

public class World : MonoBehaviour
{
    private List<Entity> entities = new List<Entity>();
    private HashSet<Entity> entitiesSet = new HashSet<Entity>();
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
}
