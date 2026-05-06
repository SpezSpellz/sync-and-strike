using UnityEngine;
using UnityEngine.UI;

public class Flip : MonoBehaviour
{
    private Toggle toggle;
    private PlayerController player;

    private void Start()
    {
        toggle = GetComponent<Toggle>();

        if (toggle == null)
        {
            Debug.LogError("Toggle is NULL on: " + gameObject.name);
            return;
        }

        player = FindFirstObjectByType<PlayerController>();

        toggle.onValueChanged.AddListener(OnFlipButtonPressed);

        // apply initial visual
        UpdateVisual(toggle.isOn);
    }

    private void OnFlipButtonPressed(bool isOn)
    {
        // revert if not in planning phase
        if (TurnManager.Instance.Phase != TurnPhase.Planning)
        {
            toggle.isOn = !isOn;
            return;
        }

        player.Flip(isOn);
        UpdateVisual(isOn);
    }

    private void UpdateVisual(bool isOn)
    {
        var colors = UIManager.Colors;

        Color baseColor = isOn ? colors.toggleOn : colors.toggleOff;

        Debug.Log("Color is" + baseColor);

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