using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// After every companion turn the player is asked whether that move was a good call.
/// The answer is written to a JSONL training log next to the state snapshot and outcome,
/// which is exactly the supervision a behaviour-cloning / preference model needs later.
///
/// The whole panel is built in code so it works without any scene or prefab setup.
/// Keyboard shortcuts: G = good, B = bad.
/// </summary>
public class CompanionVoteUI : MonoBehaviour
{
    private static CompanionVoteUI instance;

    private Canvas canvas;
    private GameObject panel;
    private TextMeshProUGUI promptLabel;
    private Button goodButton;
    private Button badButton;

    private string pendingMoveId = "";
    private System.Action<bool, bool> onResolved; // (isGood, timedOut)
    private bool waiting;

    public bool IsWaiting => waiting;

    public static CompanionVoteUI Instance
    {
        get
        {
            if (instance == null)
            {
                var go = new GameObject("CompanionVoteUI");
                instance = go.AddComponent<CompanionVoteUI>();
                DontDestroyOnLoad(go);
            }
            return instance;
        }
    }

    private void Awake()
    {
        BuildUI();
        Hide();
    }

    private void BuildUI()
    {
        canvas = gameObject.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 1000;

        // Both of these are required for the GOOD/BAD buttons to actually be clickable.
        // A GraphicRaycaster is what turns pointer input into button presses, and the
        // CanvasScaler keeps the panel a sensible size on any resolution. Without the
        // raycaster the prompt could only ever be answered with the G/B keys, which made the
        // game look frozen with no controls for the whole vote window.
        var scaler = gameObject.AddComponent<UnityEngine.UI.CanvasScaler>();
        scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        gameObject.AddComponent<UnityEngine.UI.GraphicRaycaster>();

        if (FindAnyObjectByType<EventSystem>() == null)
        {
            var es = new GameObject("EventSystem", typeof(EventSystem));
            es.transform.SetParent(transform, false);
        }

        panel = new GameObject("Panel", typeof(RectTransform));
        panel.transform.SetParent(canvas.transform, false);
        var panelRect = panel.GetComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0.5f, 1f);
        panelRect.anchorMax = new Vector2(0.5f, 1f);
        panelRect.pivot = new Vector2(0.5f, 1f);
        panelRect.anchoredPosition = new Vector2(0f, -24f);
        panelRect.sizeDelta = new Vector2(760f, 132f);

        var bg = panel.AddComponent<Image>();
        bg.color = new Color(0f, 0f, 0f, 0.82f);

        promptLabel = CreateText(panelRect, "Prompt", "Companion acted", 22, TextAlignmentOptions.Center);
        promptLabel.rectTransform.anchorMin = new Vector2(0f, 0.42f);
        promptLabel.rectTransform.anchorMax = new Vector2(1f, 1f);
        promptLabel.rectTransform.offsetMin = new Vector2(12f, 0f);
        promptLabel.rectTransform.offsetMax = new Vector2(-12f, -8f);

        CreateButton(panelRect, "GoodButton", "GOOD  (G)", new Vector2(0f, 0f), new Vector2(0.42f, 0.38f),
            new Color(0.25f, 0.62f, 0.32f), out goodButton, () => Resolve(true, false));
        CreateButton(panelRect, "BadButton", "BAD  (B)", new Vector2(0.58f, 0f), new Vector2(1f, 0.38f),
            new Color(0.68f, 0.25f, 0.25f), out badButton, () => Resolve(false, false));
    }

    private TextMeshProUGUI CreateText(RectTransform parent, string name, string content, float size, TextAlignmentOptions align)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var text = go.AddComponent<TextMeshProUGUI>();
        text.text = content;
        text.fontSize = size;
        text.alignment = align;
        text.color = Color.white;
        text.raycastTarget = false;
        return text;
    }

    private void CreateButton(RectTransform parent, string name, string label, Vector2 anchorMin, Vector2 anchorMax,
        Color color, out Button button, UnityEngine.Events.UnityAction onClick)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rect = go.GetComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.offsetMin = new Vector2(8f, 6f);
        rect.offsetMax = new Vector2(-8f, -6f);

        var image = go.AddComponent<Image>();
        image.color = color;

        button = go.AddComponent<Button>();
        button.targetGraphic = image;
        button.onClick.AddListener(onClick);

        var labelGo = new GameObject("Label", typeof(RectTransform));
        labelGo.transform.SetParent(go.transform, false);
        var labelRect = labelGo.GetComponent<RectTransform>();
        labelRect.anchorMin = Vector2.zero;
        labelRect.anchorMax = Vector2.one;
        labelRect.offsetMin = Vector2.zero;
        labelRect.offsetMax = Vector2.zero;

        var text = labelGo.AddComponent<TextMeshProUGUI>();
        text.text = label;
        text.fontSize = 24;
        text.alignment = TextAlignmentOptions.Center;
        text.color = Color.white;
        text.raycastTarget = false;
    }

    public void Show(string moveId, float timeoutSeconds, System.Action<bool, bool> onResolved)
    {
        pendingMoveId = moveId;
        this.onResolved = onResolved;
        waiting = true;
        promptLabel.text = string.IsNullOrEmpty(moveId) || moveId == "continue"
            ? "Companion: <no action>   —   was that a good call?"
            : $"Companion: <{moveId}>   —   was that a good call?";
        panel.SetActive(true);
        StopAllCoroutines();
        StartCoroutine(AutoResolve(timeoutSeconds));
    }

    public void Hide()
    {
        waiting = false;
        if (panel != null) panel.SetActive(false);
    }

    private IEnumerator AutoResolve(float timeout)
    {
        if (timeout <= 0f)
        {
            yield break;
        }
        yield return new WaitForSeconds(timeout);
        if (waiting)
        {
            Resolve(true, true);
        }
    }

    private void Update()
    {
        if (!waiting) return;
        // Keyboard shortcuts. Uses the legacy Input API, matching the rest of the project.
        if (Input.GetKeyDown(KeyCode.G)) Resolve(true, false);
        else if (Input.GetKeyDown(KeyCode.B)) Resolve(false, false);
    }

    /// <summary>External escape hatch (TurnManager backstop) so a prompt can never soft-lock.</summary>
    public void ForceResolve()
    {
        if (!waiting) return;
        Resolve(false, true);
    }

    private void Resolve(bool isGood, bool timedOut)
    {
        if (!waiting) return;
        waiting = false;
        Hide();
        var callback = onResolved;
        onResolved = null;
        callback?.Invoke(isGood, timedOut);
    }
}