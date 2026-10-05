using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PreviewManager : MonoBehaviour
{
    public static PreviewManager Instance { get; private set; }
    [HideInInspector] public Arena Arena;
    private readonly List<PreviewController> previews = new();
    private const float SIMULATION_STEP = 1f / 60f;
    private const int REPLAY_PAUSE_STEPS = 30; // 0.5 seconds at 60 fps
    private float accumulator;
    private int pauseStepsRemaining;

    private void Awake()
    {
        Instance = this;
        Arena = GetComponentInParent<Arena>();
    }

    public void RegisterPreview(PreviewController preview)
    {
        if (preview == null) return;
        if (!previews.Contains(preview)) previews.Add(preview);
        // A new move selection starts one shared cycle for every active preview.
        RestartAllPreviews();
    }
    public void UnregisterPreview(PreviewController preview)
    {
        if (preview == null) return;
        previews.Remove(preview);
        if (previews.Count == 0) pauseStepsRemaining = 0;
    }

    public void RestartAllPreviews()
    {
        pauseStepsRemaining = 0;
        foreach (var p in previews.ToList())
        {
            if (p != null)
                p.Restart();
        }
    }

    private void Update()
    {
        previews.RemoveAll(p => p == null);
        accumulator += Time.deltaTime;
        var hbm = Arena != null ? Arena.PreviewHitboxManager : PreviewHitboxManager.Instance;

        while (accumulator >= SIMULATION_STEP)
        {
            if (previews.Count > 0)
            {
                if (pauseStepsRemaining > 0)
                {
                    pauseStepsRemaining--;
                    if (pauseStepsRemaining == 0) RestartAllPreviews();
                }
                else
                {
                    foreach (var p in previews.ToList())
                        p.Step(SIMULATION_STEP);

                    // A hit can make a settled preview active again. Resolve it before
                    // checking whether the group is ready for the shared pause.
                    hbm?.Step();
                    if (previews.All(p => p.IsCycleComplete))
                        pauseStepsRemaining = REPLAY_PAUSE_STEPS;
                }
            }

            accumulator -= SIMULATION_STEP;
        }
    }

    public PreviewController GetPreviewByOwner(CharacterController owner)
    {
        return previews.FirstOrDefault(p => p.Owner == owner);
    }
}
