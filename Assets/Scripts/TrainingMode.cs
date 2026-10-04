using System;
using UnityEngine;

/// <summary>
/// Central place for training-mode toggles that used to be scattered around (the commented-out
/// SECONDS_PER_FRAME line, ad-hoc debug toggles). A training run is selected with the command
/// line switch <c>-training</c>:
/// <list type="bullet">
/// <item>the turn loop integrates with a much smaller timestep, so simulation work is not
/// capped by wall-clock frame rate;</item>
/// <item>the companion's vote prompt is disabled so training never blocks on UI;</item>
/// <item>UnityEngine.Random is seeded so runs are reproducible.</item>
/// </list>
/// </summary>
public static class TrainingMode
{
    public static bool enabled;

    /// <summary>Seconds per simulation frame while training (vs 1/60 in normal play).</summary>
    public static float secondsPerFrame = 1f / 60f;

    /// <summary>Value used to seed UnityEngine.Random in training mode.</summary>
    public static int seed = 12345;

    [RuntimeInitializeOnLoadMethod]
    private static void Initialize()
    {
        enabled = Array.Exists(Environment.GetCommandLineArgs(), a => a == "-training");
        if (enabled)
        {
            secondsPerFrame = 0.001f;
            UnityEngine.Random.InitState(seed);
        }
    }
}