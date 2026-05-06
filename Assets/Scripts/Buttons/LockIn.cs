using UnityEngine;
using UnityEngine.UI;

public class LockIn : MonoBehaviour
{
    private Button button;

    private void Start()
    {
        button = GetComponent<Button>();

        if (button == null)
        {
            Debug.LogError("Button is NULL on: " + gameObject.name);
            return;
        }

        button.image.color = UIManager.Colors.toggleOff;
        button.onClick.AddListener(OnLockInButtonPressed);
    }

    private void OnLockInButtonPressed()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        PlayerController player = FindFirstObjectByType<PlayerController>();
        TurnManager.Instance.SubmitMove(player);
        button.image.color = UIManager.Colors.toggleOn;

        // assuming the turn is done
        button.image.color = UIManager.Colors.toggleOff;
    }
}