using TMPro;
using UnityEngine;

/// <summary>
/// Small name label that floats above a fighter.
///
/// Built entirely in code so it works for any character without scene setup, and so it
/// picks its text and colour from the character's <see cref="CombatTeam"/> automatically.
/// The label tracks the fighter's world position every frame and is hidden when the
/// fighter is dead or falls off screen.
/// </summary>
[RequireComponent(typeof(CharacterController))]
public class NameTag : MonoBehaviour
{
    [Header("Text")]
    [Tooltip("Override the automatic label. Leave empty to derive it from the team.")]
    [SerializeField] private string overrideText = "";

    [Tooltip("Hide the tag for this fighter even though a team-based label exists.")]
    [SerializeField] private bool hidden = false;

    [Header("Appearance")]
    [Tooltip("Canvas-local font size. The canvas is scaled so this stays constant on screen.")]
    [SerializeField] private float fontSize = 8f;
    [Tooltip("How far above the fighter's centre the label sits, in world units.")]
    [SerializeField] private float worldHeight = 0.95f;
    [Tooltip("Width of the label rect in canvas units. Must comfortably fit the longest name.")]
    [SerializeField] private float rectWidth = 140f;

    [Tooltip("Fade the tag out when the fighter is knocked into the corner or off camera.")]
    [SerializeField] private bool hideWhenDead = true;

    private CharacterController owner;
    private Transform canvasTransform;
    private TextMeshProUGUI label;
    private Camera worldCamera;

    private void Awake()
    {
        // Resolve the owner first: BuildLabel reads Team from it, and CharacterController
        // reads CharacterData in its own Awake, so both must exist before we build.
        owner = GetComponent<CharacterController>();
        worldCamera = Camera.main;
        BuildLabel();
    }

    private void Start()
    {
        // Re-apply in case the team's CharacterData was not ready during Awake.
        label.text = ResolveText();
        label.color = ResolveColor();
    }

    private string ResolveText()
    {
        if (!string.IsNullOrEmpty(overrideText)) return overrideText;
        switch (ResolveTeam())
        {
            case CombatTeam.Player: return "You";
            case CombatTeam.Companion: return "Companion";
            case CombatTeam.Enemy: return "Enemy";
            default: return owner != null ? owner.name : "";
        }
    }

    /// <summary>
    /// Reads the team defensively. CharacterController.characterData is populated in its own
    /// Awake, and script execution order is not guaranteed, so the tag can run first.
    /// </summary>
    private CombatTeam ResolveTeam()
    {
        if (owner == null || owner.Data == null) return CombatTeam.Neutral;
        return owner.Data.team;
    }

    private Color ResolveColor()
    {
        // Match the debug colours already defined for teams, but bright enough to read.
        Color c = CombatTeamUtility.DebugColor(ResolveTeam());
        c.a = 1f;
        return c;
    }

    private void BuildLabel()
    {
        var canvasGo = new GameObject("NameTagCanvas");
        canvasGo.transform.SetParent(transform, false);
        canvasGo.transform.localPosition = new Vector3(0f, worldHeight, 0f);
        canvasGo.transform.localRotation = Quaternion.identity;
        canvasGo.transform.localScale = Vector3.one;

        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        // A world-space canvas is sized in canvas units and then scaled down. At a scale of
        // 0.01 * orthographicSize, one canvas unit is a constant ~0.5% of the screen height,
        // so the label keeps the same apparent size while the camera zooms.
        canvasGo.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        canvasGo.GetComponent<RectTransform>().sizeDelta = new Vector2(rectWidth, rectWidth * 0.25f);
        canvasTransform = canvasGo.transform;

        var textGo = new GameObject("Label", typeof(RectTransform));
        textGo.transform.SetParent(canvasGo.transform, false);
        var textRect = textGo.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0.5f, 0.5f);
        textRect.anchorMax = new Vector2(0.5f, 0.5f);
        textRect.pivot = new Vector2(0.5f, 0.5f);
        textRect.sizeDelta = new Vector2(rectWidth, rectWidth * 0.25f);
        textRect.localPosition = Vector3.zero;

        label = textGo.AddComponent<TextMeshProUGUI>();
        label.text = ResolveText();
        label.fontSize = fontSize;
        label.alignment = TextAlignmentOptions.Center;
        label.color = ResolveColor();
        label.raycastTarget = false;
        // Without this a narrow rect wraps the name one letter per line, which reads as a
        // vertical column instead of a horizontal label.
        label.enableWordWrapping = false;
        label.overflowMode = TextOverflowModes.Overflow;
    }

    private void LateUpdate()
    {
        if (label == null) return;

        bool shouldShow = !hidden && !(hideWhenDead && owner != null && owner.IsDead());
        canvasTransform.gameObject.SetActive(shouldShow);
        if (!shouldShow) return;

        // Self-correct if the label was built before the character's data was ready.
        string wanted = ResolveText();
        if (label.text != wanted)
        {
            label.text = wanted;
            label.color = ResolveColor();
        }

        // Follow the fighter's body, staying upright regardless of how the sprite flips.
        Vector3 world = owner.transform.position;
        world.y += worldHeight;
        canvasTransform.position = world;
        canvasTransform.rotation = Quaternion.identity;

        if (worldCamera == null) worldCamera = Camera.main;
        // Compute the scale absolutely (never relative to the previous frame) so the label
        // keeps a constant apparent size as the camera zooms.
        float dist = worldCamera != null && worldCamera.orthographic
            ? worldCamera.orthographicSize
            : 3.6f;
        float baseScale = 0.01f * dist;
        // Fighters are mirrored with a negative X scale to face left, which would mirror any
        // child too. Cancel it so the label always reads left-to-right.
        float flipX = owner.transform.localScale.x < 0f ? -1f : 1f;
        canvasTransform.localScale = new Vector3(baseScale * flipX, baseScale, baseScale);
    }
}