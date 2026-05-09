using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MoveButton : MonoBehaviour
{
    [SerializeField] private Image icon;

    public string moveId;
    public string moveName;
    public TextMeshProUGUI nameText;
    private AnimationData data;

    private void Awake()
    {
        GetComponent<Button>().onClick.AddListener(OnMoveButtonPressed);
        icon = transform.Find("Icon").GetComponent<Image>();
        nameText = transform.parent.parent.Find("MoveName").GetComponent<TextMeshProUGUI>(); // get the text above
    }


    public void Initialize(AnimationData data)
    {
        this.data = data;
        moveId = data.moveId;
        moveName = data.moveName;

        icon.sprite = data.icon;
    }

    private void OnMoveButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        nameText.text = moveName; // set the text above to the column

        Debug.Log($"data: {this.data}");
        Debug.Log($"nameText: {nameText}");
        Debug.Log($"TurnManager: {TurnManager.Instance}");

        switch (this.data.requiredInput)
        {
            case RequiredInput.None:
                FindFirstObjectByType<PlayerController>().SelectMove(moveId);
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