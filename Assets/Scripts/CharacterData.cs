using UnityEngine;

public class CharacterData : MonoBehaviour
{
    public  AnimationData[] animations { get; private set; }

    private void Awake()
    {
        animations = Resources.LoadAll<AnimationData>("Characters/Swordsman/AnimationData"); // CHANGE HERE
        Debug.Log($"Loaded {animations.Length} animations");

        GetComponent<CharacterAnimation>().Initialize(animations);
        GetComponent<CharacterPhysics>().Initialize(animations);
    }

    public AnimationData GetMove(string moveId)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId) return anim;
        }
        Debug.LogWarning($"No AnimationData found for moveId: {moveId}");
        return null;
    }
}