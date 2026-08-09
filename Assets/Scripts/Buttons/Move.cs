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
    private bool isControllable;

    private void Awake()
    {
        button = GetComponent<Button>();
        if (button != null)
            button.onClick.AddListener(OnMoveButtonPressed);
        icon = transform.Find("Icon").GetComponent<Image>();
        nameText = transform.parent.parent.Find("MoveName").GetComponent<TextMeshProUGUI>(); // get the text above
    }

    public void Initialize(AnimationData data, bool isControllable, CharacterController owner)
    {
        this.data = data;
        this.isControllable = isControllable;
        this.owner = owner;
        moveId = data.moveId;
        moveName = data.moveName;
        icon.sprite = data.icon;
        RefreshState();
    }

    private void OnEnable()
    {
        RefreshState();
        Debug.Log($"Refresh {moveId} usable={button.interactable} owner={owner?.name}");
    }
    
    private void Update()
    {
        RefreshState();
    }

    private void RefreshState()
    {
        bool usable = isControllable
            && owner != null
            && TurnManager.Instance != null
            && TurnManager.Instance.Phase == TurnPhase.Planning
            && owner.CanUseMove(data);

        if (button != null)
            button.interactable = usable;

        if (icon != null)
            icon.color = usable ? Color.white : UIManager.Colors.disabled;
    }

    private void OnMoveButtonPressed()
    {
        if (!isControllable) return;
        if (TurnManager.Instance == null) return;
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;
        if (!owner.CanUseMove(data)) return;

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