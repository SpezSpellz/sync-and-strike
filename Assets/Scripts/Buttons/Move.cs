using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MoveButton : MonoBehaviour
{
    [SerializeField] private Image icon;

    public string moveId;
    public string moveName;
    public TextMeshProUGUI nameText;
    private CharacterController owner;
    private AnimationData data;
    private Button button;
    private MoveSelectionUI parentUI;
    public bool IsSelected { get; private set; }
    private bool isControllable;

    private void Awake()
    {
        button = GetComponent<Button>();
        if (button != null)
            button.onClick.AddListener(OnMoveButtonPressed);
        icon = transform.Find("Icon").GetComponent<Image>();
        nameText = transform.parent.parent.Find("MoveName").GetComponent<TextMeshProUGUI>(); // get the text above
    }

    public void Initialize(AnimationData data, bool isControllable, CharacterController owner, MoveSelectionUI parentUI)
    {
        this.data = data;
        this.isControllable = isControllable;
        this.owner = owner;
        this.parentUI = parentUI;
        moveId = data.moveId;
        moveName = data.moveName;
        icon.sprite = data.icon;
        SetSelected(false);
    }

    private void OEnable()
    {
        SetSelected(false);
    }

    public bool IsUsable()
    {
        return owner.CanUseMove(data);
    }

    public void SetSelected(bool selected)
    {
        IsSelected = selected;
        if (button != null)
            button.image.color = selected ? UIManager.Colors.moveSelected : UIManager.Colors.primary;
    }

    private void OnMoveButtonPressed()
    {
        if (!isControllable) return;
        if (TurnManager.Instance == null) return;
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;
        if (!IsUsable()) return;
        if (parentUI == null) return;

        if (IsSelected)
        {
            parentUI.ClearSelection();
            owner.SelectMove("continue");
            JumpWheel.Instance.Hide();
            return;
        }

        parentUI.SelectMoveButton(this);
        nameText.text = moveName; // set the text above to the column

        Debug.Log($"data: {this.data}");
        Debug.Log($"nameText: {nameText}");
        Debug.Log($"TurnManager: {TurnManager.Instance}");

        switch (this.data.requiredInput)
        {
            case RequiredInput.None:
                if (owner != null) owner.SelectMove(moveId);
                JumpWheel.Instance.Hide();
                break;
            case RequiredInput.JumpWheel:
                JumpWheel.Instance.Show();
                break;
            /*
            case RequiredInput.DirectionWheel:
                FindFirstObjectByType<DirectionWheel>().Show(data.moveId); // NOT IMPLEMENTED YET
                break;
                */
        }
    }
}