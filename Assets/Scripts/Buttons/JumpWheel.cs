using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class JumpWheel : MonoBehaviour, IDragHandler, IPointerDownHandler, IPointerUpHandler
{
    [Header("References")]
    [SerializeField] private RectTransform indicator;
    [SerializeField] private RectTransform wheelRect;

    [Header("Settings")]
    [SerializeField] private float wheelRadius = 100f;
    [SerializeField] private float deadZoneRadius = 50f;
    [SerializeField] private float floorRestriction = 15f;

    private bool isActive = false;
    private float currentPower = 0f;
    private float currentDirection = Mathf.PI / 2f; // default straight up
    private PlayerController player;

    // clamp constants matching your friend's values
    private const float MIN_POWER = 0.5f;
    private const float MAX_POWER = 1.0f;
    private const float MIN_DIRECTION = 0.5235987755982988f;  // 30 degrees
    private const float MAX_DIRECTION = 2.6179938779914944f;  // 150 degrees
    public static JumpWheel Instance { get; private set; }

    private void Awake()
    {
        Instance = this;
        player = FindObjectOfType<PlayerController>();
        gameObject.SetActive(false);
    }

    public void Show()
    {
        gameObject.SetActive(true);
        // offset origin to bottom of wheel, wheel darius divide by 2 because the wheel is not a full circle, only upper half, so going from middle (of the upper half) to bottom is only half of the full circle radius
        indicator.anchoredPosition = new Vector2(0, 0);
        currentPower = 0f;
        currentDirection = Mathf.PI / 2f;
    }

    public void Hide() // unused
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
            player.setJumpInfo(currentPower, currentDirection);
            player.SelectMove("jump");
        }
    }

    private void UpdateInput(Vector2 screenPosition)
    {
        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            wheelRect, screenPosition, null, out Vector2 localPoint
        );      

        // ignore input from below the wheel entirely
        if (localPoint.y < 0)
        {
            currentPower = 0f;
            float clampedX = Mathf.Clamp(localPoint.x, -wheelRadius, wheelRadius);
            float indicatorX = Mathf.Clamp(Mathf.Abs(clampedX), deadZoneRadius, wheelRadius) * Mathf.Sign(clampedX);
            indicator.anchoredPosition = new Vector2(indicatorX, floorRestriction - wheelRadius / 2f);
            return;
        }

        float distance = localPoint.magnitude;

        // clamp to wheel radius
        float clampedDistance = Mathf.Min(distance, wheelRadius);
        Vector2 clampedPoint = localPoint.normalized * clampedDistance;

        // floor restriction
        clampedPoint.y = Mathf.Max(floorRestriction, clampedPoint.y);

        // move indicator — sticks to dead zone edge if inside
        float indicatorDistance = Mathf.Clamp(distance, deadZoneRadius, wheelRadius);
        Vector2 indicatorPoint = localPoint.normalized * indicatorDistance;
        indicatorPoint.y = Mathf.Max(floorRestriction, indicatorPoint.y);
        indicator.anchoredPosition = new Vector2(indicatorPoint.x, indicatorPoint.y - wheelRadius / 2f);

        // dead zone
        if (distance < deadZoneRadius)
        {
            currentPower = 0f;
            return;
        }

        // power
        float normalizedDistance = (clampedDistance - deadZoneRadius) / (wheelRadius - deadZoneRadius);
        currentPower = Mathf.Lerp(MIN_POWER, MAX_POWER, normalizedDistance);

        // direction
        float angle = Mathf.Atan2(clampedPoint.y, clampedPoint.x);
        currentDirection = Mathf.Clamp(angle, MIN_DIRECTION, MAX_DIRECTION);
    }
}