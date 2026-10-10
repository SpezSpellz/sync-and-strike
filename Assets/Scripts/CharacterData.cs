using UnityEngine;

public class CharacterData : MonoBehaviour
{
    public AnimationData[] animations { get; private set; }

    public float width;

    public float height;

    public float health;

    public float maxHealth;

    [Header("Combat")]
    [Tooltip("Allies pass through each other and cannot damage each other.")]
    public CombatTeam team = CombatTeam.Neutral;

    [Tooltip("Characters recover from a knockdown with less health than a combo end.")]
    public bool isPlayerSide;

    /// <summary>
    /// Multiplier applied to every hit this fighter lands. 1 is parity. Set at spawn from
    /// <see cref="CombatBalance"/> for the enemy rather than authored, so the code-built training
    /// arena and the shipped scene cannot disagree about how hard the enemy hits.
    /// </summary>
    public float damageMultiplier = 1f;

    private void Awake()
    {
        animations = Resources.LoadAll<AnimationData>(AnimationDataResourcePath);
        if (animations == null || animations.Length == 0)
        {
            Debug.LogError($"No AnimationData loaded from Resources/{AnimationDataResourcePath} for {name}");
            animations = new AnimationData[0];
            return;
        }
        Debug.Log($"Loaded {animations.Length} animations for {name}");
    }

    /// <summary>Folder under Resources/ that holds this character's move ScriptableObjects.</summary>
    public const string AnimationDataResourcePath = "Characters/Swordsman/AnimationData";

    public AnimationData GetMove(string moveId)
    {
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId) return anim;
        }
        Debug.LogWarning($"No AnimationData found for moveId: {moveId}");
        return null;
    }

    public bool HasMove(string moveId)
    {
        if (string.IsNullOrEmpty(moveId)) return false;
        foreach (var anim in animations)
        {
            if (anim.moveId == moveId) return true;
        }
        return false;
    }
}