using UnityEngine;
using UnityEngine.UI;

public class LockIn : MonoBehaviour
{
    private Button button;

    private void Start()
    {
        button = GetComponent<Button>();

        Debug.Log("Button component: " + button);

        if (button == null)
        {
            Debug.LogError("Button is NULL on: " + gameObject.name);
            return;
        }

        button.onClick.AddListener(OnLockInButtonPressed);
    }

    private void OnLockInButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        PlayerController player = FindFirstObjectByType<PlayerController>();
        TurnManager.Instance.SubmitMove(player, player.SelectedMove);
    }
}