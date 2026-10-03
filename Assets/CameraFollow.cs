using UnityEngine;

/// <summary>
/// Camera that frames both fighters.
///
/// Framing behaviour:
///  - Target is the midpoint of the two fighters (here: all living combatants).
///  - Position lerps toward the target at 0.28 per frame for a soft follow.
///  - The camera only zooms out when the fighters separate vertically by more than
///    CAMERA_MAX_Y_DIST (210px); it never zooms in.
///  - The view is clamped to the stage so the camera never shows past the walls.
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraFollow : MonoBehaviour
{
    [Header("Follow")]
    [Tooltip("Camera lerps toward the midpoint at 0.28 per frame.")]
    [Range(0.01f, 1f)]
    private float followLerp = 0.28f;

    [Tooltip("Ignore the companion when framing; only the two main fighters are considered.")]
    private bool ignoreCompanion = false;

    [Header("Zoom")]
    [Tooltip("Max vertical separation of 210px ~= 4.2 units. Beyond this the camera zooms out.")]
    private float maxVerticalSeparation = 4.2f;

    [Tooltip("Default viewport height is 360px ~= 7.2 units (half = orthographic size).")]
    private float baseOrthographicSize = 3.6f;

    [Tooltip("Upper bound on how far the camera may zoom out, in orthographic size units.")]
    private float maxOrthographicSize = 7.5f;

    [Tooltip("20px ~= 0.4 units of slack past the walls.")]
    private float stagePadding = 0.4f;

    [Tooltip("How far down the screen the ground line may sit. 0.8 keeps fighters clear of the bottom UI.")]
    [Range(0.5f, 1.5f)]
    private float groundScreenFraction = 0.82f;

    [Header("Stage bounds (0 = auto-detect from the wall colliders)")]
    private float minX = 0f;
    private float maxX = 0f;
    private float floorY = 0f;
    private float ceilingY = 0f;

    private Camera cam;
    private float currentSize;

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
                if (box.getCenter().y < 0f) floorY = box.maxY;
                else ceilingY = box.minY;
            }
        }
    }

    private void Update()
    {
        Vector2 target;
        int count;
        FindTarget(out target, out count);
        if (count <= 0) return;

        // Soft follow toward the midpoint.
        Vector3 pos = transform.position;
        pos.x = Mathf.Lerp(pos.x, target.x, followLerp);
        pos.y = Mathf.Lerp(pos.y, target.y, followLerp);

        // Zoom out only when the fighters are far apart vertically.
        float wantedSize = baseOrthographicSize;
        if (maxVerticalSeparation > 0.01f)
        {
            float spread = VerticalSpread();
            if (spread > maxVerticalSeparation)
            {
                float ratio = spread / maxVerticalSeparation;
                wantedSize = baseOrthographicSize * ratio;
            }
        }
        currentSize = Mathf.Lerp(currentSize, Mathf.Min(wantedSize, maxOrthographicSize), followLerp);
        if (cam.orthographic) cam.orthographicSize = currentSize;

        pos = ClampToStage(pos);
        transform.position = pos;
    }

    private float VerticalSpread()
    {
        // Only zoom based on vertical separation between the fighters.
        float min = float.MaxValue, max = float.MinValue;
        bool any = false;
        ForEachCombatant(c =>
        {
            float y = c.GetPosition().y;
            if (y < min) min = y;
            if (y > max) max = y;
            any = true;
        });
        return any ? max - min : 0f;
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
        if (ceilingY > floorY)
        {
            // The stage is only slightly taller than the view, so requiring the whole viewport
            // to fit inside the stage would shove the ground line off the bottom of the screen
            // and hide the fighters behind the move-selection UI. Instead, keep the ground line
            // inside a band on screen: never higher than `groundScreenFraction` down the view,
            // and never so low that the ceiling leaves the top.
            float minY = floorY - groundScreenFraction * 2f * halfH;
            float maxY = ceilingY + stagePadding - halfH;
            pos.y = maxY > minY ? Mathf.Clamp(pos.y, minY, maxY) : (floorY + ceilingY) * 0.5f;
        }
        pos.z = -10f;
        return pos;
    }

    private void FindTarget(out Vector2 target, out int count)
    {
        Vector2 sum = Vector2.zero;
        int found = 0;
        ForEachCombatant(c =>
        {
            sum += c.GetPosition();
            found++;
        });
        count = found;
        target = count > 0 ? sum / count : Vector2.zero;
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