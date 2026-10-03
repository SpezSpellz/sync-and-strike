using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Adds the companion's health bar to the player side of the HUD.
///
/// The player bar occupies the top-left strip. This clones that bar, shrinks it and stacks it
/// directly underneath with a "Companion" caption, so the layout stays consistent with the
/// existing bars without hand-authoring scene objects. The bar auto-binds to whichever fighter
/// is on <see cref="CombatTeam.Companion"/>, so it works even if the companion is added later.
/// </summary>
public class CompanionHealthBarUI : MonoBehaviour
{
    [Header("Source")]
    [Tooltip("The player bar to clone. Defaults to the left-most bar under the canvas.")]
    [SerializeField] private RectTransform sourceBar;

    [Header("Layout")]
    [Tooltip("Fraction of the player bar's width. 1 = same size as the player's bar.")]
    [SerializeField] private float widthScale = 1f;

    [Tooltip("Fraction of the player bar's height. 1 = same size as the player's bar.")]
    [SerializeField] private float heightScale = 1f;

    [Tooltip("Vertical gap between the player bar and the companion bar, in pixels.")]
    [SerializeField] private float gap = 6f;

    [Tooltip("Pixels to shift the companion bar horizontally, lining it up with the player bar.")]
    [SerializeField] private float extraLeftOffset = 0f;

    [Header("Colours")]
    [SerializeField] private Color fillColor = new Color(0.4f, 0.9f, 0.6f, 1f);
    [SerializeField] private Color backColor = new Color(0f, 0f, 0f, 1f);

    [Header("Caption")]
    [Tooltip("Off by default: a full-width bar leaves no room beside it, and the companion's " +
             "nametag plus its green colour already identify the bar.")]
    [SerializeField]
    private bool showCaption = false;

    private HealthBar healthBar;
    private Image fillImage;
    private GameObject captionObject;
    private bool bound;

    private void Start()
    {
        if (sourceBar == null) sourceBar = FindSourceBar();
        if (sourceBar == null)
        {
            Debug.LogWarning("CompanionHealthBarUI: no source health bar found; companion bar not created.");
            enabled = false;
            return;
        }
        Build();
    }

    /// <summary>
    /// The player's bar: the one anchored to the left edge of the canvas. The enemy bar is
    /// right-anchored, and any bar we already created is skipped.
    /// </summary>
    private RectTransform FindSourceBar()
    {
        var bars = GetComponentsInChildren<HealthBar>(true);
        HealthBar best = null;
        foreach (var bar in bars)
        {
            if (bar.Owner != null && bar.Owner.Team == CombatTeam.Companion) continue;
            var rect = bar.transform as RectTransform;
            if (rect == null) continue;
            if (rect.anchorMin.x > 0.5f) continue;          // right-anchored: enemy side
            if (best == null) { best = bar; continue; }
            var bestRect = best.transform as RectTransform;
            if (rect.anchoredPosition.x < bestRect.anchoredPosition.x) best = bar;
        }
        return best != null ? best.transform as RectTransform : null;
    }

    private void Build()
    {
        // sourceBar is the HealthBar's own rect (the coloured fill). The bar we want to copy
        // is its parent, which is the background Image containing both the fill and the bar.
        var sourceRoot = sourceBar.parent as RectTransform;
        if (sourceRoot == null) sourceRoot = sourceBar;

        var clone = Instantiate(sourceRoot.gameObject, sourceRoot.parent);
        clone.name = "Companion Health";

        var cloneRect = clone.GetComponent<RectTransform>();
        float w = sourceRoot.sizeDelta.x * widthScale;
        float h = sourceRoot.sizeDelta.y * heightScale;

        // Keep the source bar's anchoring, then shrink and stack underneath it.
        cloneRect.anchorMin = sourceRoot.anchorMin;
        cloneRect.anchorMax = sourceRoot.anchorMax;
        cloneRect.pivot = sourceRoot.pivot;
        cloneRect.sizeDelta = new Vector2(w, h);
        cloneRect.anchoredPosition = sourceRoot.anchoredPosition
            + new Vector2(extraLeftOffset, -(h * 0.5f + gap + sourceRoot.sizeDelta.y * 0.5f));

        // Resize the fill to match the new background. It must keep explicit anchors and
        // sizeDelta, because HealthBar drives sizeDelta directly to animate the bar; stretching
        // it to the parent would make it ignore those writes and never shrink.
        healthBar = clone.GetComponentInChildren<HealthBar>(true);
        var fillRect = healthBar != null ? healthBar.transform as RectTransform : null;
        if (fillRect != null)
        {
            fillRect.anchorMin = new Vector2(0.5f, 0.5f);
            fillRect.anchorMax = new Vector2(0.5f, 0.5f);
            fillRect.pivot = new Vector2(0.5f, 0.5f);
            fillRect.sizeDelta = new Vector2(w, h);
            fillRect.anchoredPosition = Vector2.zero;
        }

        var background = clone.GetComponent<Image>();
        if (background != null) background.color = backColor;

        var fillImageRef = fillRect != null ? fillRect.GetComponent<Image>() : null;
        if (fillImageRef != null && fillImageRef != background)
        {
            fillImageRef.color = fillColor;
            fillImage = fillImageRef;
        }

        if (healthBar != null)
        {
            healthBar.Bind(FindCompanion());
        }
        else
        {
            Debug.LogWarning("CompanionHealthBarUI: cloned bar has no HealthBar component.");
        }

        if (showCaption) BuildCaption(cloneRect);
    }

    private void BuildCaption(RectTransform barRect)
    {
        // "Companion" label to the left of the bar, so the player can tell whose bar it is.
        var go = new GameObject("Caption", typeof(RectTransform));
        go.transform.SetParent(barRect.parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = barRect.anchorMin;
        rect.anchorMax = barRect.anchorMax;
        rect.pivot = new Vector2(1f, 0.5f);
        rect.sizeDelta = new Vector2(180f, barRect.sizeDelta.y);
        rect.anchoredPosition = barRect.anchoredPosition + new Vector2(-barRect.sizeDelta.x * 0.5f - 8f, 0f);

        var text = go.AddComponent<TMPro.TextMeshProUGUI>();
        text.text = "Companion";
        text.fontSize = 22f;
        text.alignment = TMPro.TextAlignmentOptions.Right;
        text.color = fillColor;
        text.raycastTarget = false;
        captionObject = go;
    }

    /// <summary>
    /// Re-bind once the fighters have registered. CharacterController.Start registers with
    /// TurnManager, and script execution order is not guaranteed, so the binding is retried
    /// for a few frames in case the companion registered after this component ran.
    /// </summary>
    private void Update()
    {
        if (healthBar == null) return;
        if (bound) return;
        var companion = FindCompanion();
        if (companion != null)
        {
            healthBar.Bind(companion);
            bound = true;
        }
    }

    private CharacterController FindCompanion()
    {
        // Prefer the registered fighter from TurnManager, then fall back to a scene scan.
        if (TurnManager.Instance != null)
        {
            var registered = TurnManager.Instance.FindCompanion();
            if (registered != null) return registered;
        }
        var all = FindObjectsByType<CharacterController>(FindObjectsSortMode.None);
        CharacterController best = null;
        foreach (var c in all)
        {
            if (c == null) continue;
            if (c.Data == null || c.Data.team != CombatTeam.Companion) continue;
            if (best == null) best = c;
        }
        return best;
    }
}