using UnityEngine;
using UnityEngine.UI;

public class LockIn : MonoBehaviour
{
    private Button button;

    private void Awake()
    {
        button = GetComponent<Button>();
        button.onClick.AddListener(OnLockInButtonPressed);
    }

    private void OnLockInButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        PlayerController player = FindObjectOfType<PlayerController>();
        TurnManager.Instance.SubmitMove(player, player.SelectedMove);
    }
}