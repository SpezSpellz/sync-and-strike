using UnityEngine;

/// <summary>
/// Camera that frames the living combatants.
///
/// Framing behaviour:
///  - Target is the centre of the living combatants' bounds.
///  - Horizontal position lerps toward the target at 0.28 per frame for a soft follow.
///  - The camera zooms out when horizontal or vertical separation exceeds the
///    corresponding threshold, then returns to its base size as they regroup.
///  - The floor stays at a fixed screen height while the camera zooms.
///  - Horizontal position is clamped to the stage walls.
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraFollow : MonoBehaviour
{
    [Header("Follow")]
    [Tooltip("Camera position and zoom lerp toward their targets at 0.28 per frame.")]
    [Range(0.01f, 1f)]
    private float followLerp = 0.28f;

    [Tooltip("Ignore the companion when framing; only the two main fighters are considered.")]
    private bool ignoreCompanion = false;

    [Header("Zoom")]
    [Tooltip("Max vertical separation of 210px ~= 4.2 units. The horizontal threshold scales with camera aspect.")]
    private float maxVerticalSeparation = 4.2f;

    [Tooltip("Default viewport height is 360px ~= 7.2 units (half = orthographic size).")]
    private float baseOrthographicSize = 3.6f;

    [Tooltip("Upper bound on how far the camera may zoom out, in orthographic size units.")]
    private float maxOrthographicSize = 7.5f;

    [Tooltip("20px ~= 0.4 units of slack past the walls.")]
    private float stagePadding = 0.4f;

    [Tooltip("Fixed floor position measured down from the top of the screen. 0.82 leaves room for the bottom UI.")]
    [Range(0f, 1f)]
    private float groundScreenFraction = 0.55f;

    [Header("Stage bounds (0 = auto-detect from the wall colliders)")]
    private float minX = 0f;
    private float maxX = 0f;
    private float floorY = 0f;
    private float ceilingY = 0f;

    private Camera cam;
    private float currentSize;
    private bool hasFloor;

    private void Awake()
    {
        cam = GetComponent<Camera>();
        currentSize = baseOrthographicSize;
        AutoDetectBounds();
    }

    /// <summary>
    /// Derives the playable area from the StaticCollider walls/floor/ceiling so the camera
    /// clamps to the real geometry instead of hand-typed numbers.
    /// </summary>
    private void AutoDetectBounds()
    {
        hasFloor = floorY != 0f;
        if (minX != 0f && maxX != 0f && floorY != 0f && ceilingY != 0f) return;

        var walls = FindObjectsByType<StaticCollider>(FindObjectsSortMode.None);
        foreach (var wall in walls)
        {
            var box = wall.getBoundingBox();
            if (box == null) continue;
            bool tallAndThin = box.getWidth() < box.getHeight();
            if (tallAndThin)
            {
                // Wall: inner face is whichever side is closer to the stage centre.
                if (box.getCenter().x < 0f) minX = box.maxX;
                else maxX = box.minX;
            }
            else
            {
                // Floor or ceiling: flat and wide.
                if (box.getCenter().y < 0f)
                {
                    floorY = box.maxY;
                    hasFloor = true;
                }
                else ceilingY = box.minY;
            }
        }
    }

    private void Update()
    {
        Vector2 target;
        Vector2 spread;
        if (!FindFraming(out target, out spread)) return;

        // Follow the fighters horizontally. When the floor is known, its screen
        // position determines camera height rather than the fighters' midpoint.
        Vector3 pos = transform.position;
        pos.x = Mathf.Lerp(pos.x, target.x, followLerp);

        // Orthographic size is half the viewport height; horizontal world-space
        // coverage also depends on aspect, so either axis can set the zoom.
        float wantedSize = baseOrthographicSize;
        if (maxVerticalSeparation > 0.01f)
        {
            float verticalRatio = spread.y / maxVerticalSeparation;
            float horizontalRatio = spread.x / (maxVerticalSeparation * Mathf.Max(cam.aspect, 0.01f));
            wantedSize *= Mathf.Max(1f, Mathf.Max(verticalRatio, horizontalRatio));
        }
        currentSize = Mathf.Lerp(currentSize, Mathf.Min(wantedSize, maxOrthographicSize), followLerp);
        if (cam.orthographic) cam.orthographicSize = currentSize;

        // At screen fraction g from the top, the floor is (2g - 1) half-heights
        // below the camera centre. Use the smoothed size to keep it fixed while zooming.
        pos.y = hasFloor && cam.orthographic
            ? floorY + (2f * groundScreenFraction - 1f) * cam.orthographicSize
            : Mathf.Lerp(pos.y, target.y, followLerp);

        pos = ClampToStage(pos);
        transform.position = pos;
    }

    private Vector3 ClampToStage(Vector3 pos)
    {
        if (cam == null || !cam.orthographic) return pos;
        float halfH = cam.orthographicSize;
        float halfW = halfH * cam.aspect;

        if (maxX > minX)
        {
            float left = minX - stagePadding + halfW;
            float right = maxX + stagePadding - halfW;
            pos.x = right > left
                ? Mathf.Clamp(pos.x, left, right)   // room to pan
                : (minX + maxX) * 0.5f;             // stage narrower than the view, centre it
        }
        if (!hasFloor && ceilingY > floorY)
        {
            // Preserve the old vertical clamp when a floor cannot be detected.
            float minY = floorY - groundScreenFraction * 2f * halfH;
            float maxY = ceilingY + stagePadding - halfH;
            pos.y = maxY > minY ? Mathf.Clamp(pos.y, minY, maxY) : (floorY + ceilingY) * 0.5f;
        }
        pos.z = -10f;
        return pos;
    }

    private bool FindFraming(out Vector2 target, out Vector2 spread)
    {
        Vector2 min = new Vector2(float.MaxValue, float.MaxValue);
        Vector2 max = new Vector2(float.MinValue, float.MinValue);
        bool found = false;
        ForEachCombatant(c =>
        {
            Vector2 position = c.GetPosition();
            min = Vector2.Min(min, position);
            max = Vector2.Max(max, position);
            found = true;
        });
        target = found ? (min + max) * 0.5f : Vector2.zero;
        spread = found ? max - min : Vector2.zero;
        return found;
    }

    private void ForEachCombatant(System.Action<CharacterController> action)
    {
        if (TurnManager.Instance == null) return;
        TurnManager.Instance.ForEachPlayer(c =>
        {
            if (c == null || c.IsDead()) return;
            if (ignoreCompanion && c is CompanionController) return;
            action(c);
        });
    }
}
