using UnityEngine;

public class Hurtbox : MonoBehaviour
{
    public CharacterPhysics owner;

    private void Awake()
    {
        if (owner == null)
            owner = GetComponentInParent<CharacterPhysics>();
    }
}