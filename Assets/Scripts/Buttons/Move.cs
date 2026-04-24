using UnityEngine;
using UnityEngine.UI;

public class Move : MonoBehaviour
{
    [SerializeField] private string moveId;
    private Button button;

    private void Awake()
    {
        button = GetComponent<Button>();
        button.onClick.AddListener(OnMoveButtonPressed);
    }

    private void OnMoveButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        PlayerController player = FindObjectOfType<PlayerController>();
        player.SelectMove(moveId);
    }
}