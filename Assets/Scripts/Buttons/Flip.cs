using UnityEngine;
using UnityEngine.UI;

public class Flip : MonoBehaviour
{
    private Toggle toggle;

    private void Start()
    {
        toggle = GetComponent<Toggle>();

        Debug.Log("Toggle component: " + toggle);

        if (toggle == null)
        {
            Debug.LogError("Toggle is NULL on: " + gameObject.name);
            return;
        }

        toggle.onValueChanged.AddListener(OnFlipButtonPressed);
    }

    private void OnFlipButtonPressed(bool active)
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning) return;

        PlayerController player = FindFirstObjectByType<PlayerController>();
        player.Flip(active);
    }
}