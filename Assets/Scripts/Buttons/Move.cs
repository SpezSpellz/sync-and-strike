using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MoveButton : MonoBehaviour
{
    [SerializeField] private Image icon;
    public string moveId;
    public string moveName;
    public TextMeshProUGUI nameText;

    private void Awake()
    {
        GetComponent<Button>().onClick.AddListener(OnMoveButtonPressed);
        icon = transform.Find("Icon").GetComponent<Image>();
        nameText = transform.parent.parent.Find("MoveName").GetComponent<TextMeshProUGUI>(); // get the text above
    }


    public void Initialize(AnimationData data)
    {
        moveId = data.moveId;
        moveName = data.moveName;

        icon.sprite = data.icon;
    }

    private void OnMoveButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;
        FindFirstObjectByType<PlayerController>().SelectMove(moveId);
        nameText.text = moveName; // update the text above the column
    }
}