using UnityEngine;

public class PreviewPhysicsManager : MonoBehaviour
{
    public static PreviewPhysicsManager Instance { get; private set; }
    [HideInInspector] public Arena Arena;
    private IndexSet<PhysicsCollider> objects = new();

    private void Awake()
    {
        Instance = this;
        Arena = GetComponentInParent<Arena>();
    }

    public int Register(PhysicsCollider physics)
    {
        if (physics.getId() != -1)
            return physics.getId();
        return objects.add(physics);
    }

    public void Unregister(PhysicsCollider physics)
    {
        if (physics.getId() == -1)
            return;
        objects.remove(physics.getId());
        physics.setId(-1);
    }

    public IndexSet<PhysicsCollider> GetRegisteredObjects()
    {
        return this.objects;
    }

    public void StepFor(PhysicsCollider physics)
    {
        PhysicsManager pm = Arena != null ? Arena.physicsManager : PhysicsManager.Instance;
        pm.StepFor(physics, objects); // reuse the real physics step
    }
}