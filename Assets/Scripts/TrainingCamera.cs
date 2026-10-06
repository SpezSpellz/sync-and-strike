using UnityEngine;

/// <summary>Shows one complete training arena without following its fighters.</summary>
[RequireComponent(typeof(Camera))]
public class TrainingCamera : MonoBehaviour
{
    [Tooltip("Extra world-space room around the arena edges.")]
    public float framingPadding = 0.5f;

    [Tooltip("Background colour matching the scene camera.")]
    public Color backgroundColor = new Color(0.19215687f, 0.3019608f, 0.4745098f);

    private Camera cam;
    private float halfStageWidth;
    private float stageBottom;
    private float stageTop;
    private float floorY;
    private float lastAspect;

    private void Awake()
    {
        cam = GetComponent<Camera>();
        cam.orthographic = true;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = backgroundColor;
        cam.depth = 0f; // Draw after the scene's camera, which has depth -1.
    }

    /// <summary>Fit the arena while keeping its floor at the normal camera's screen height.</summary>
    public void FrameStage(float halfWidth, float bottom, float floor, float top)
    {
        halfStageWidth = halfWidth;
        stageBottom = bottom;
        stageTop = top;
        floorY = floor;
        UpdateZoom();
    }

    private void LateUpdate()
    {
        // Fighter motion never changes the view. On resize, adjust height with zoom to pin the floor.
        if (halfStageWidth > 0f && !Mathf.Approximately(cam.aspect, lastAspect))
            UpdateZoom();
    }

    private void UpdateZoom()
    {
        lastAspect = cam.aspect;
        float padding = Mathf.Max(0f, framingPadding);
        float floorFraction = CameraFollow.GroundScreenFraction;
        float sizeForTop = (stageTop - floorY + padding) / (2f * floorFraction);
        float sizeForBottom = (floorY - stageBottom + padding) / (2f * (1f - floorFraction));
        float sizeForWidth = (halfStageWidth + padding) / Mathf.Max(lastAspect, 0.01f);
        cam.orthographicSize = Mathf.Max(sizeForTop, Mathf.Max(sizeForBottom, sizeForWidth));
        transform.localPosition = new Vector3(0f,
            floorY + (2f * floorFraction - 1f) * cam.orthographicSize, -10f);
    }
}
