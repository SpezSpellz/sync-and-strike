using UnityEditor;
using UnityEngine;

/// <summary>
/// Editor-only entry point for a training run.
///
/// The normal way to train is <c>Unity.exe -batchmode -nographics -training</c>, which launches a
/// separate process. That is not always possible (the editor may already hold the project open, or
/// you may just want to watch a short run happen). These menu items let the same training code run
/// inside the already-open editor by forcing <see cref="TrainingMode"/> on and entering play mode.
///
/// Note that this is NOT as fast as a batch-mode build: the editor still renders, and the frame
/// rate is the editor's. Use it to verify the training loop works, not to run a real training job.
/// </summary>
public static class TrainingMenu
{
    private const string RigName = "TrainingRig";

    [MenuItem("Tools/Training/Start Training In Editor")]
    public static void StartInEditor()
    {
        EnsureRig();
        // Stored in EditorPrefs, not a static: entering play mode reloads the domain and would
        // reset a static flag back to false before training mode had a chance to read it.
        TrainingMode.ForcedByEditor = true;
        if (!EditorApplication.isPlaying) EditorApplication.isPlaying = true;
        Debug.Log("[Training] Entering play mode as a training run. Stop play mode to end it.");
    }

    [MenuItem("Tools/Training/Stop Training In Editor")]
    public static void StopInEditor()
    {
        TrainingMode.ForcedByEditor = false;
        if (EditorApplication.isPlaying) EditorApplication.isPlaying = false;
    }

    [MenuItem("Tools/Training/Create Training Rig")]
    public static void EnsureRig()
    {
        var go = GameObject.Find(RigName);
        bool created = false;
        if (go == null)
        {
            go = new GameObject(RigName);
            go.AddComponent<TrainingArenaBuilder>();
            go.AddComponent<TrainingMatchRunner>();
            created = true;
        }

        var runner = go.GetComponent<TrainingMatchRunner>();
        if (runner == null)
        {
            Debug.LogError($"[Training] '{RigName}' exists but has no TrainingMatchRunner.");
            return;
        }

        // Editor defaults, applied EVERY time rather than only at creation. A rig is a scene object
        // that keeps whatever values it was saved with, so setting these only when it is first made
        // means an older rig silently keeps stale settings forever - which is exactly how reporting
        // ended up quiet until match 25.
        //
        // One arena: far easier to read the log with a single match in flight, and this is for
        // verifying the loop rather than throughput. Report every match so progress is visible.
        runner.arenaCount = 1;
        runner.reportEvery = 1;

        if (created)
        {
            Undo.RegisterCreatedObjectUndo(go, "Create Training Rig");
            Debug.Log("[Training] Created TrainingRig. Use Tools > Training > Start Training In Editor.");
        }
    }
}