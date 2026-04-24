using UnityEngine;
using UnityEngine.UI;

public class MoveButton : MonoBehaviour
{
    [SerializeField] private Image icon;
    public string moveId;
    public string moveName;

    private void Awake()
    {
        GetComponent<Button>().onClick.AddListener(OnMoveButtonPressed);
        icon = transform.Find("Icon").GetComponent<Image>();
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
    }
}