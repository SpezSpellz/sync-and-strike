using UnityEngine;

public class PreviewPhysicsManager : MonoBehaviour
{
    public static PreviewPhysicsManager Instance { get; private set; }
    private IndexSet<PhysicsCollider> objects = new();

    private void Awake() => Instance = this;

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

    public void StepFor(PhysicsCollider physics)
    {
        PhysicsManager.Instance.StepFor(physics, objects); // reuse existing physics step
    }
}