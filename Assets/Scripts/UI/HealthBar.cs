using UnityEngine;

/// <summary>
/// Health bar fill. Shrinks from the anchored side so the bar drains towards the
/// character's edge rather than always draining to the left.
/// </summary>
public class HealthBar : MonoBehaviour
{
    [SerializeField]
    private CharacterData characterData;

    [Tooltip("The fighter this bar tracks. When set, the data is resolved from it at runtime " +
             "so bars can be added without rewiring the asset reference by hand.")]
    [SerializeField]
    private CharacterController owner;

    [Tooltip("Drain from the right instead of the left (used by the enemy bar).")]
    [SerializeField]
    private bool flip;

    [Tooltip("Hide the whole bar when the tracked fighter dies.")]
    [SerializeField]
    private bool hideWhenDead;

    /// <summary>The fighter this bar tracks, if one was assigned.</summary>
    public CharacterController Owner => owner;

    private RectTransform rectTransform;
    private RectTransform root;
    private float maxSize;
    private bool resolved;

    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        this.maxSize = this.rectTransform.rect.width;
        root = transform.parent as RectTransform;
        ResolveData();
    }

    /// <summary>
    /// Points this bar at a fighter. This overwrites any previously assigned data, which matters
    /// for cloned bars: they inherit the source bar's serialized reference and would otherwise
    /// keep tracking the original fighter.
    /// </summary>
    public void Bind(CharacterController fighter)
    {
        owner = fighter;
        characterData = fighter != null ? fighter.Data : null;
        resolved = characterData != null;
        if (rectTransform != null && maxSize <= 0f)
        {
            maxSize = rectTransform.rect.width;
        }
    }

    private void ResolveData()
    {
        if (resolved) return;
        if (characterData == null && owner != null)
        {
            characterData = owner.Data;
        }
        resolved = true;
    }

    private void Update()
    {
        ResolveData();
        if (characterData == null) return;

        if (hideWhenDead && owner != null && owner.IsDead())
        {
            if (root != null) root.gameObject.SetActive(false);
            return;
        }
        if (root != null && !root.gameObject.activeSelf) root.gameObject.SetActive(true);

        float t = characterData.maxHealth > 0f ? characterData.health / characterData.maxHealth : 0f;
        t = Mathf.Clamp01(t);
        this.rectTransform.sizeDelta = new Vector2(this.maxSize * t, this.rectTransform.sizeDelta.y);
        this.rectTransform.localPosition = new Vector3((flip ? -((1 - t) / 2) : ((1 - t) / 2)) * this.maxSize, 0, 0);
    }
}