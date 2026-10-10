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

    #if UNITY_EDITOR
    /// <summary>
    /// EditorPrefs key holding the editor-only training override.
    ///
    /// This MUST live in EditorPrefs rather than a static field: entering play mode triggers a domain
    /// reload, which wipes every static. An earlier version stored the flag in a static bool, so it
    /// was reset to false before the [RuntimeInitializeOnLoadMethod] hook below could read it and
    /// training silently never started. EditorPrefs survives the reload.
    /// </summary>
    private const string EditorPrefKey = "SyncedStrike.ForceTrainingInEditor";
#endif

    /// <summary>
    /// Continue from whatever weights are already on disk instead of re-warming from the rule-based
    /// expert. Off by default on purpose: a stale or barely-trained file silently disabled the expert
    /// warm start, leaving the policy to explore from near-random weights with no reward signal, and
    /// it collapsed onto a single do-nothing move.
    /// </summary>
    public static bool resume;

    /// <summary>
    /// Stop a batch-mode run after this many matches, then flush and quit. 0 means no limit.
    ///
    /// Exists so a long run can be driven as a sequence of bounded segments. Each segment appends to
    /// its own CSV, so quality can be judged from the data before committing to a longer run, and a
    /// run that has visibly diverged is stopped rather than left burning the machine.
    /// </summary>
    public static int maxMatches;

    /// <summary>Stop a run after this many seconds, for the same reason. 0 means no limit.</summary>
    public static int maxSeconds;

    /// <summary>
    /// One in every N training matches is an EVALUATION match: neither side records a transition, and
    /// the opponents are fixed (rule-based or random) rather than drawn from the curriculum.
    ///
    /// These matches exist because once a curriculum is in play the ordinary win rate stops meaning
    /// anything - it mixes several opponents with different difficulties, and the role being trained is
    /// only ever on one side of it. The eval slice is the only uncontaminated measurement available, and
    /// it is what the promotion gate reads. Without it the gate would advance on in-sample performance.
    ///
    /// Costs one match in N. 0 disables the slice, which also disables promotion, since the gate has
    /// nothing to read.
    ///
    /// N sets how long the curriculum takes to promote, and the coupling is easy to get wrong. The
    /// slice rotates over 4 (role x opponent) pairs, so each pair collects one graded sample every
    /// 4 x evalEvery matches. At 25 that was one sample per 100 matches, which against a 50-sample gate
    /// needed 5,000 matches to promote - so a real 748-match run advanced zero times and the opponent
    /// mix never left 100% random. At 8 a role reaches the 16-sample gate in roughly 512 matches.
    /// Keep evalEvery and OpponentCurriculum's advanceWindow in step when either is changed.
    /// </summary>
    public static int evalEvery = 8;

    /// <summary>
    /// Turn the opponent curriculum off, so every training match is plain self-play.
    ///
    /// Exists so the curriculum can be A/B measured against the previous behaviour without a recompile,
    /// which is the only way to tell whether it is actually earning the throughput it costs.
    /// </summary>
    public static bool curriculum = true;

    /// <summary>
    /// Simulation steps one arena may run per Unity frame while training.
    ///
    /// This is the knob that actually controls throughput. TurnManager drains its simulation backlog
    /// against this budget each frame and discards any surplus, so total simulated frames per second is
    /// roughly (arena count x this) x rendered frame rate. Raising it past what the machine can sustain
    /// does not speed anything up - it just means frames overrun and Unity drops them - which is why the
    /// telemetry reports frames per second rather than assuming this value was achieved.
    ///
    /// Also the reason N arenas in one process scale: arenas share the process but each has its own
    /// TurnManager, so this budget is effectively per arena.
    /// </summary>
    public static int maxSimFramesPerUnityFrame = 64;

    /// <summary>
    /// Editor-only override for the <c>-training</c> switch, so a training run can be exercised inside
    /// the already-open editor (which cannot receive a new command line). Always false in a build.
    ///
    /// Stored in EditorPrefs, not a static field: entering play mode triggers a domain reload, which
    /// wipes every static. An earlier version used a static bool and it was always reset to false
    /// before the [RuntimeInitializeOnLoadMethod] hook below could read it, so training silently never
    /// started. EditorPrefs survives the reload.
    /// </summary>
    public static bool ForcedByEditor
    {
        get
        {
#if UNITY_EDITOR
            return UnityEditor.EditorPrefs.GetInt(EditorPrefKey, 0) == 1;
#else
            return false;
#endif
        }
        set
        {
#if UNITY_EDITOR
            UnityEditor.EditorPrefs.SetInt(EditorPrefKey, value ? 1 : 0);
#endif
        }
    }

    [RuntimeInitializeOnLoadMethod]
    private static void Initialize()
    {
        var args = Environment.GetCommandLineArgs();
        bool requested = Array.Exists(args, a => a == "-training");
#if UNITY_EDITOR
        requested = requested || ForcedByEditor;
#endif
        enabled = requested;
        resume = Array.Exists(args, a => a == "-resume");

        // Bounded-run limits, e.g. -maxMatches=200 -maxSeconds=600.
        //
        // A training run is normally left to finish on its own, but that makes "is this working?"
        // unanswerable until the very end. Bounding a run lets it be driven as a sequence of segments,
        // each writing its own CSV, so quality can be judged from real data before committing to a
        // longer run - and a run that has visibly diverged gets stopped instead of left burning CPU.
        maxMatches = ReadIntArg(args, "-maxMatches");
        maxSeconds = ReadIntArg(args, "-maxSeconds");

        // -evalEvery=N, the evaluation slice size. Read as an int rather than through the float override
        // path because it is a COUNT, and a fractional match count has no meaning - silently rounding one
        // down here would quietly disable promotion and leave the curriculum frozen in phase 1.
        int evalEveryArg = ReadIntArg(args, "-evalEvery");
        if (evalEveryArg > 0) evalEvery = evalEveryArg;

        curriculum = !Array.Exists(args, a => a == "-noCurriculum");
        if (!curriculum) evalEvery = 0;

        // -simFrames=N, the throughput knob. Clamped because an unbounded value would let one frame
        // simulate indefinitely, which looks like a hang rather than a slow run.
        int simFrames = ReadIntArg(args, "-simFrames");
        if (simFrames > 0)
            maxSimFramesPerUnityFrame = Mathf.Clamp(simFrames, 1, 100000);

        // -simSpeed=X scales the simulated timestep. 1 is the default 1ms step; 0.5 halves it, which makes
        // the same number of turns cover half as much game time and roughly doubles turn throughput.
        // Only sensible while TRAINING, since it changes the physics the policies are learning.
        foreach (var a in args)
        {
            if (string.IsNullOrEmpty(a) || !a.StartsWith("-simSpeed=")) continue;
            if (float.TryParse(a.Substring(10), System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture, out float s)
                && s > 0.0001f && s <= 1f)
            {
                secondsPerFrame = 0.001f * s;
                Debug.Log($"[Training] simulation timestep set to {secondsPerFrame}s/frame "
                    + $"(simSpeed {s}).");
            }
            else
            {
                Debug.LogWarning($"[Training] ignoring -simSpeed='{a}' (expected 0 < value <= 1).");
            }
        }

        // Combat balance overrides, e.g. -enemyDamage=1.9 -enemyHealth=1.9. Parsed unconditionally
        // (not only when -training is set) so the same switch can pin the numbers for a balance test
        // in a normal build. They only change the ENEMY, and only at fighter spawn.
        foreach (var a in args)
        {
            if (string.IsNullOrEmpty(a)) continue;
            const string damageKey = "-enemyDamage=";
            const string healthKey = "-enemyHealth=";
            string field = null;
            string valueText = null;
            if (a.StartsWith(damageKey)) { field = "enemyDamage"; valueText = a.Substring(damageKey.Length); }
            else if (a.StartsWith(healthKey)) { field = "enemyHealth"; valueText = a.Substring(healthKey.Length); }
            if (field == null) continue;

            if (float.TryParse(valueText, System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture, out float v)
                && CombatBalance.TrySet(field, v))
            {
                Debug.Log($"[Training] enemy balance override {field}={v:0.###} ({CombatBalance.Describe()}).");
            }
            else
            {
                Debug.LogWarning($"[Training] ignoring '{a}' (expected a positive value).");
            }
        }

        if (enabled)
        {
            if (secondsPerFrame <= 0f) secondsPerFrame = 0.001f;
            UnityEngine.Random.InitState(seed);
        }
    }

    /// <summary>
    /// Read an integer switch such as <c>-maxMatches=200</c>. Returns 0 for absent, malformed or
    /// non-positive values, which all mean "no limit".
    /// </summary>
    private static int ReadIntArg(string[] args, string name)
    {
        foreach (var a in args)
        {
            if (string.IsNullOrEmpty(a) || !a.StartsWith(name + "=")) continue;
            if (int.TryParse(a.Substring(name.Length + 1), out int v) && v > 0) return v;
            Debug.LogWarning($"[Training] ignoring malformed limit '{a}'");
            return 0;
        }
        return 0;
    }
}