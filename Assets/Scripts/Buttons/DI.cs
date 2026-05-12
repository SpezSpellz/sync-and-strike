using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class DI : MonoBehaviour, IDragHandler, IPointerDownHandler, IPointerUpHandler
{
    [Header("References")]
    [SerializeField] private RectTransform indicator;
    [SerializeField] private RectTransform wheelRect;

    [Header("Settings")]
    [SerializeField] private float wheelRadius = 75f;

    public static DI Instance { get; private set; }

    private float currentPower = 0f;
    private float currentDirection = Mathf.PI / 2f;

    private bool isActive = false;
    private PlayerController player;
    private string pendingMoveId;

    private void Awake()
    {
        Instance = this;
        player = FindObjectOfType<PlayerController>();
    }

    public void Show(string moveId)
    {
        pendingMoveId = moveId;
        gameObject.SetActive(true);
        indicator.anchoredPosition = Vector2.zero;
        currentPower = 0f;
        currentDirection = 0f;
    }

    public void Hide()
    {
        gameObject.SetActive(false);
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        isActive = true;
        UpdateInput(eventData.position);
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (!isActive) return;
        UpdateInput(eventData.position);
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        if (!isActive) return;
        isActive = false;

        if (currentPower > 0)
        {
            player.setKnockbackInfo(currentPower, currentDirection);
        }
    }

    private void UpdateInput(Vector2 screenPosition)
    {
        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            wheelRect, screenPosition, null, out Vector2 localPoint
        );

        float distance = localPoint.magnitude;

        // clamp to wheel radius
        if (distance > wheelRadius)
            localPoint = localPoint.normalized * wheelRadius;

        // power = 0 to 1 based on distance from center
        currentPower = Mathf.Clamp01(distance / wheelRadius);

        // direction in radians, full 360
        currentDirection = Mathf.Atan2(localPoint.y, localPoint.x);

        // move indicator
        indicator.anchoredPosition = localPoint;
    }
}