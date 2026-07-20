using UnityEngine;
using UnityEngine.UI;

public class LockIn : MonoBehaviour
{
    private Button button;
    
    public void Initialize(bool isControllable)
    {
        button = GetComponent<Button>();

        if (button == null)
        {
            Debug.LogError("Button is NULL on: " + gameObject.name);
            return;
        }

        // Button only function when it's for controllable character
        if (isControllable)
        {
            button.image.color = UIManager.Colors.toggleOff;
            button.onClick.AddListener(OnLockInButtonPressed);
        }
        
        else
        {
            button.image.color = UIManager.Colors.disabled;
        }
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