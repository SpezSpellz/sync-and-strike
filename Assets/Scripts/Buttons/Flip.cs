using UnityEngine;
using UnityEngine.UI;

public class Flip : MonoBehaviour
{
    private Toggle toggle;
    private CharacterController owner;
    private bool isControllable;

    private void Awake()
    {
        toggle = GetComponent<Toggle>();

        if (toggle == null)
        {
            Debug.LogError("Toggle is NULL on: " + gameObject.name);
            return;
        }
        toggle.onValueChanged.AddListener(OnFlipButtonPressed);
    }

    public void Initialize(CharacterController owner, bool isControllable)
    {
        this.owner = owner;
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
        if (TurnManager.Instance == null || TurnManager.Instance.Phase != TurnPhase.Planning || owner == null)
        {
            ResetToggle();
            return;
        }

        bool currentlyFlipped = owner.PreviewScale.x < 0f;
        bool shouldFlipTo = !currentlyFlipped;

        owner.Flip(shouldFlipTo, !isControllable);
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