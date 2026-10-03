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

        // The toggle now mirrors the fighter's real facing, so its value IS the target facing.
        // Previously this derived the direction from PreviewScale and inverted the toggle's
        // meaning, so clicking "on" could flip the fighter the same way it was already facing.
        if (isControllable) owner.FlipAndRememberFacing(isOn);
        else owner.Flip(isOn, true);   // mirror panel: preview ghost only

        PreviewManager.Instance.RestartAllPreviews();
        UpdateVisual(isOn);
    }

    /// <summary>
    /// Syncs the toggle to the fighter's actual facing. This used to force the toggle off every
    /// time, which only read correctly while the opponent happened to be on the right; once the
    /// fighters crossed over the toggle showed the opposite of reality.
    /// </summary>
    private void ResetToggle()
    {
        if (toggle == null || owner == null)
            return;

        bool flipped = owner.IsFlipped;
        toggle.SetIsOnWithoutNotify(flipped);
        UpdateVisual(flipped);
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