using UnityEngine;
using UnityEngine.UI;

public class Flip : MonoBehaviour
{
    private Toggle toggle;
    private PlayerController player;
    private bool isControllable;

    private void Awake()
    {
        toggle = GetComponent<Toggle>();

        if (toggle == null)
        {
            Debug.LogError("Toggle is NULL on: " + gameObject.name);
            return;
        }

        player = FindFirstObjectByType<PlayerController>();
        toggle.onValueChanged.AddListener(OnFlipButtonPressed);
    }

    public void Initialize(bool isControllable)
    {
        this.isControllable = isControllable;
        ResetToggle();
    }

    private void OnEnable()
    {
        if (TurnManager.Instance != null && TurnManager.Instance.Phase == TurnPhase.Planning)
        {
            ResetToggle();
        }
    }

    private void OnFlipButtonPressed(bool isOn)
    {
        if (TurnManager.Instance == null || TurnManager.Instance.Phase != TurnPhase.Planning)
        {
            ResetToggle();
            return;
        }

        if (!isControllable || player == null)
        {
            ResetToggle();
            return;
        }

        bool currentlyFlipped = player.transform.localScale.x < 0f;
        bool shouldFlipTo = !currentlyFlipped;

        player.Flip(shouldFlipTo);
        PreviewManager.Instance.RestartAllPreviews();
        UpdateVisual(isOn);
    }

    private void ResetToggle()
    {
        if (toggle == null)
            return;

        toggle.SetIsOnWithoutNotify(false);
        UpdateVisual(false);
    }

    private void UpdateVisual(bool isOn)
    {
        if (toggle == null)
            return;

        var colors = UIManager.Colors;
        Color baseColor = isOn ? colors.toggleOn : colors.toggleOff;

        ColorBlock cb = toggle.colors;
        cb.normalColor = baseColor;
        cb.highlightedColor = baseColor;
        cb.pressedColor = colors.toggleOn;
        cb.selectedColor = baseColor;
        cb.disabledColor = colors.disabled;
        cb.colorMultiplier = 1f;

        toggle.colors = cb;
    }
}