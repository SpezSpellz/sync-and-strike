using UnityEngine;

/// <summary>
/// Frames one training arena on screen.
///
/// The scene's existing <see cref="CameraFollow"/> cannot be reused: it reads
/// <see cref="TurnManager.Instance"/>, which with several arenas alive is whichever arena's Awake
/// ran last, not the one you want to look at. This follows the fighters of a specific arena instead.
///
/// Purely cosmetic. It does nothing under -nographics.
/// </summary>
[RequireComponent(typeof(Camera))]
public class TrainingCamera : MonoBehaviour
{
    [Tooltip("Fighters to keep framed. Set by TrainingArenaBuilder for the rendered arena.")]
    public CharacterController[] targets;

    /// <summary>
    /// Half the visible height, in world units. Matches the scene CameraFollow's baseOrthographicSize
    /// (3.6), not the Main Camera's 5. At 5 a 0.7-unit fighter renders less than a third of the size
    /// it does in normal play.
    /// </summary>
    public float orthographicSize = 3.6f;

    [Tooltip("Background colour. Matches the scene camera's blue; anything darker reads as the " +
             "training view breaking the game's look.")]
    public Color backgroundColor = new Color(0.19215687f, 0.3019608f, 0.4745098f);

    [Tooltip("Where the camera sits vertically relative to the fighters.")]
    public float verticalOffset = 0.5f;

    private Camera cam;

    private void Awake()
    {
        // Guaranteed by [RequireComponent]; adding it here would collide with the builder's own add.
        cam = GetComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = orthographicSize;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = backgroundColor;
    }

    /// <summary>
    /// Snap straight to the midpoint. Lamping is unnecessary here: training arenas are placed at
    /// fixed offsets, so a smooth follow would only make the view swim.
    /// </summary>
    public void SnapToTargets()
    {
        if (targets == null) return;
        Vector2 sum = Vector2.zero;
        int count = 0;
        foreach (var t in targets)
        {
            if (t == null) continue;
            sum += t.GetPosition();
            count++;
        }
        if (count == 0) return;

        Vector2 mid = sum / count;
        transform.position = new Vector3(mid.x, mid.y + verticalOffset, -10f);
    }

    private void LateUpdate()
    {
        SnapToTargets();
    }
}