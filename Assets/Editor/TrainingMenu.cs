using System.IO;
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
        // Eight arenas, not one. TurnManager now spends a whole Unity frame's simulation budget rather
        // than stopping after a single resolved turn, so a frame can carry many turns; arenas are the
        // cheapest way to use the rest of that budget, and self-play only improves with more concurrent
        // opponents. One arena left the frame mostly idle and threw away throughput. Override from the
        // command line with -arenas=N (up to 64).
        runner.arenaCount = 8;
        runner.reportEvery = 1;

        if (created)
        {
            Undo.RegisterCreatedObjectUndo(go, "Create Training Rig");
            Debug.Log("[Training] Created TrainingRig. Use Tools > Training > Start Training In Editor.");
        }
    }

    private const string ShippedDir = "Assets/StreamingAssets/AI";

    /// <summary>
    /// Copy a trained role's runtime weights into StreamingAssets so a build ships them.
    ///
    /// This copies the FILE rather than calling PolicyLearner.ExportRole because that method reads a
    /// live session, and there is no session in edit mode. Copying the file the trainer last wrote is also
    /// more honest: it is exactly the weights the run finished on.
    ///
    /// Exporting the enemy here is what makes the shipped enemy the trained one: a frozen role reads this
    /// copy FIRST (see PolicyLearner.LoadWeights), ahead of any stray file in a player's persistent data.
    /// </summary>
    [MenuItem("Tools/Training/Export Frozen Enemy")]
    public static void ExportFrozenEnemy() => ExportWeights("enemy");

    /// <summary>Ship a companion starting point. A player who has already trained one keeps their own.</summary>
    [MenuItem("Tools/Training/Export Companion Baseline")]
    public static void ExportCompanionBaseline() => ExportWeights("companion");

    private static void ExportWeights(string role)
    {
        var src = Path.Combine(Application.persistentDataPath, role + "_policy.json");
        if (!File.Exists(src))
        {
            Debug.LogError($"[Training] No trained {role} weights at {src}. Run a training run first.");
            return;
        }
        Directory.CreateDirectory(ShippedDir);
        var dst = Path.Combine(ShippedDir, role + "_policy.json");
        File.Copy(src, dst, true);
        AssetDatabase.ImportAsset(dst.Replace('\\', '/'));
        Debug.Log($"[Training] Exported {role} policy ({new FileInfo(dst).Length / 1024} KB) to {dst}. "
            + "Commit the file so a build ships it.");
    }
}