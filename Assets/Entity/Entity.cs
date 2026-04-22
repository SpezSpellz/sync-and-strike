using UnityEngine;

public abstract class Entity : MonoBehaviour
{
    public virtual void Start()
    {
        World.Instance.addEntity(this);
    }

    public abstract void Step();
}
