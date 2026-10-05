using UnityEngine;
using UnityEngine.UI;

public class Flip : MonoBehaviour
{
    private Toggle toggle;
    private CharacterController owner;
    
    private bool isControllable; // True if button is for the player, not the enemy panel!

    private void Awake()
    {
        toggle = GetComponent<Toggle>();

        if (toggle == null)
        {
            Debug.LogError("Toggle is NULL on: " + gameObject.name);
            return;
        }
        toggle.onValueChanged.AddListener(OnFlipToggleChanged);
    }

    public void Initialize(CharacterController owner, bool isControllable)
    {
        this.owner = owner;
        this.isControllable = isControllable;
        SyncToggleToPlayerChoice();
    }

    private void OnEnable()
    {
        if (TurnManager.Instance != null && TurnManager.Instance.Phase == TurnPhase.Planning)
        {
            SyncToggleToPlayerChoice();
        }
    }

    private void OnFlipToggleChanged(bool useOppositeFacing)
    {
        if (TurnManager.Instance == null || TurnManager.Instance.Phase != TurnPhase.Planning || owner == null)
        {
            SyncToggleToPlayerChoice();
            return;
        }

        if (isControllable)
        {
            // On reverses automatic facing; off returns to facing the target.
            if (useOppositeFacing)
                owner.ReverseFacingForThisTurn();
            else
                owner.ResumeAutomaticFacing();
        }
        else
        {
            // The mirror panel changes only its preview, relative to the fighter's real facing.
            bool previewFacesLeft = owner.IsFacingLeft;
            if (useOppositeFacing) previewFacesLeft = !previewFacesLeft;
            owner.SetFacingLeft(previewFacesLeft, previewOnly: true);
        }

        PreviewManager.Instance.RestartAllPreviews();
        UpdateToggleColors(useOppositeFacing);
    }

    /// <summary>
    /// Shows whether the player has chosen a facing override this turn. Automatic facing never
    /// activates the toggle, even when it turns the fighter left.
    /// </summary>
    private void SyncToggleToPlayerChoice()
    {
        if (toggle == null || owner == null)
            return;

        bool playerChoseOppositeFacing = isControllable && owner.HasPlayerFacingOverride;
        toggle.SetIsOnWithoutNotify(playerChoseOppositeFacing);
        if (!isControllable) owner.SetFacingLeft(owner.IsFacingLeft, previewOnly: true);
        UpdateToggleColors(playerChoseOppositeFacing);
    }

    private void UpdateToggleColors(bool useOppositeFacing)
    {
        if (toggle == null)
            return;

        var colors = UIManager.Colors;
        Color baseColor = useOppositeFacing ? colors.toggleOn : colors.toggleOff;

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