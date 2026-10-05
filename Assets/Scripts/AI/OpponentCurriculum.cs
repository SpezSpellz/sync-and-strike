using System;
using UnityEngine;

/// <summary>
/// Where a fighter's actions for one match come from.
///
/// The distinction that matters is <see cref="BehaviourSourceUtil.Records"/>: a transition is only a
/// valid PPO sample if the network actually chose the action that produced the reward. Every source
/// except <see cref="BehaviourSource.Live"/> is played by something other than the network - or by the
/// network with recording switched off - so recording is gated on that one value rather than on a list
/// of exclusions.
/// </summary>
public enum BehaviourSource
{
    /// <summary>Act with this fighter's own PPO network, and record the turn as a training sample.</summary>
    Live,

    /// <summary>
    /// Act with this fighter's own network but record nothing. Used by the evaluation slice, where the
    /// point is to measure the policy against an opponent it is NOT simultaneously training on.
    /// </summary>
    LiveNoLearn,

    /// <summary>Act with the rule-based <see cref="FighterAI"/> expert. Fixed, competent, non-drifting.</summary>
    Rule,

    /// <summary>Uniform over the legal actions. Incoherent, but cheap exploration pressure.</summary>
    Random,
}

public static class BehaviourSourceUtil
{
    /// <summary>True when the action comes from the fighter's own network, whether or not it records.</summary>
    public static bool UsesNetwork(this BehaviourSource s) =>
        s == BehaviourSource.Live || s == BehaviourSource.LiveNoLearn;

    /// <summary>
    /// True when the turn is a valid PPO sample.
    ///
    /// Single source of truth for "may this turn enter the rollout buffer". Gating here rather than at
    /// the call site is deliberate: a scripted fighter that recorded anyway would put a log-prob
    /// against a distribution it did not sample from, so every advantage computed from that transition
    /// would be wrong - and the buffer only discards its batch every 2048 turns, so the poison would
    /// survive a long time before it showed up.
    /// </summary>
    public static bool Records(this BehaviourSource s) => s == BehaviourSource.Live;

    /// <summary>Short token for the telemetry CSV, so win rates can be grouped by opponent.</summary>
    public static string Token(this BehaviourSource s)
    {
        switch (s)
        {
            case BehaviourSource.Live: return "live";
            case BehaviourSource.LiveNoLearn: return "live_eval";
            case BehaviourSource.Rule: return "rule";
            case BehaviourSource.Random: return "random";
            default: return "unknown";
        }
    }

    /// <summary>Parse a telemetry token back, for reading a run's CSV.</summary>
    public static BehaviourSource FromToken(string token)
    {
        switch (token)
        {
            case "live": return BehaviourSource.Live;
            case "live_eval": return BehaviourSource.LiveNoLearn;
            case "rule": return BehaviourSource.Rule;
            case "random": return BehaviourSource.Random;
            default: return BehaviourSource.Live;
        }
    }
}

/// <summary>
/// A fixed-capacity ring of recent win/loss outcomes, so the curriculum can gate on a real win rate
/// rather than on match count.
///
/// Capacity is larger than the window it is usually read over, because the same buffer serves two
/// windows of different lengths: a short one to decide whether to promote, and a long one to detect
/// that the policy has started forgetting.
/// </summary>
public sealed class WinWindow
{
    private readonly bool[] buf;
    private int head;
    private int total;

    public WinWindow(int capacity) { buf = new bool[Mathf.Max(8, capacity)]; }

    /// <summary>Total outcomes ever pushed, including any that have since rolled out of the ring.</summary>
    public int Count => total;

    public void Push(bool won)
    {
        buf[head] = won;
        head = (head + 1) % buf.Length;
        total++;
    }

    public void Clear() { Array.Clear(buf, 0, buf.Length); head = 0; total = 0; }

    /// <summary>Win rate over the most recent <paramref name="lastK"/> outcomes, oldest excluded.</summary>
    public float Rate(int lastK)
    {
        int k = Mathf.Min(lastK, total);
        if (k <= 0) return 0f;
        int wins = 0;
        for (int i = 1; i <= k; i++)
        {
            int idx = ((head - i) % buf.Length + buf.Length) % buf.Length;
            if (buf[idx]) wins++;
        }
        return (float)wins / k;
    }
}

/// <summary>
/// Per-role opponent curriculum: which opponents a role faces, and when it is promoted to harder ones.
///
/// The progression is deliberately easy-first:
///
///   phase 0 - 100% random. Teaches approach, spacing and hit confirmation against an opponent that
///             cannot punish anything. The per-turn shaping terms (approach, damage dealt) still vary
///             turn to turn here, so there IS a gradient even though the terminal bonus is nearly
///             constant because a random opponent loses almost every match.
///   phase 1 - 50% random, 50% rule-based. The rule brain is the first opponent that actually defends,
///             so blocking and spacing have to be learned for real to keep winning.
///   phase 2 - 25% random, 25% rule, 50% live opponent. Cross-play.
///
/// Why this rather than a fixed mixture: a win rate against a live opponent that improves in lockstep
/// is not evidence of anything. A curriculum measures against FIXED opponents first, where the number
/// means what it says, and only introduces the drifting one once there is something worth drifting
/// with.
///
/// Phase is per role, not global. Two roles advancing independently is what stops the match alternation
/// from producing a sawtooth where one side races ahead, the other spends its turns catching up, and
/// the first races ahead again.
///
/// Known gap, recorded deliberately: phase 2's live share is the only learning signal and it MOVES, so
/// a policy can still overfit to it. Random and rule-based at 25% each hold a competence band, but they
/// are fixed opponents and exert no pressure against drift. If the training win rate climbs while both
/// evaluation win rates fall, that live share is the knob to reduce.
/// </summary>
public sealed class OpponentCurriculum
{
    public const int PhaseCount = 3;

    /// <summary>Fraction of matches fought against the random source, per phase. Live is the remainder.</summary>
    private readonly float[] randomShare = { 1f, 0.5f, 0.25f };

    /// <summary>Fraction fought against the rule-based expert, per phase.</summary>
    private readonly float[] ruleShare = { 0f, 0.5f, 0.25f };

    // Promotion gates. Advancing needs a high win rate over a SHORT window; regressing needs a low one
    // over a LONG window. The asymmetry is the point - promoting early is cheap to undo, but dropping a
    // phase the policy has genuinely outgrown would cost it real progress, so that takes more evidence.
    private float advanceRandomRate = 0.85f;
    private float advanceRuleRate = 0.70f;
    private float regressRuleRate = 0.50f;
    private int advanceWindow = 50;
    private int regressWindow = 200;

    private readonly WinWindow vsRandom;
    private readonly WinWindow vsRule;
    private readonly System.Random rng;

    private int phase;

    public OpponentCurriculum(string role, int seed)
    {
        Role = role;
        vsRandom = new WinWindow(regressWindow);
        vsRule = new WinWindow(regressWindow);
        rng = new System.Random(seed);
    }

    public string Role { get; }
    public int Phase => phase;
    public float RandomShare => randomShare[phase];
    public float RuleShare => ruleShare[phase];

    /// <summary>Everything not given to random or rule-based. Derived, so the three can never disagree.</summary>
    public float LiveShare => Mathf.Max(0f, 1f - randomShare[phase] - ruleShare[phase]);

    /// <summary>Draw this role's opponent for one match.</summary>
    public BehaviourSource Draw()
    {
        float r = randomShare[phase];
        float ru = ruleShare[phase];
        float x = (float)rng.NextDouble();
        if (x < r) return BehaviourSource.Random;
        if (x < r + ru) return BehaviourSource.Rule;
        return BehaviourSource.Live;
    }

    /// <summary>
    /// Record an evaluation match's outcome.
    ///
    /// Only <see cref="BehaviourSource.Rule"/> and <see cref="BehaviourSource.Random"/> count. Training
    /// matches are deliberately excluded: in phases 0 and 1 the policy is learning against these very
    /// opponents, so their training win rate is in-sample performance and would promote the curriculum
    /// on the strength of memorisation rather than skill.
    /// </summary>
    public void RecordEval(BehaviourSource opponent, bool won)
    {
        if (opponent == BehaviourSource.Random) vsRandom.Push(won);
        else if (opponent == BehaviourSource.Rule) vsRule.Push(won);
    }

    /// <summary>
    /// Consider a promotion. Returns true on the match that crossed the gate, so the caller can log it.
    ///
    /// Windows are cleared on promotion so the next phase is judged on fresh evidence rather than on the
    /// easy opponents that were just beaten.
    /// </summary>
    public bool TickAdvance()
    {
        if (phase >= PhaseCount - 1) return false;
        if (phase == 0)
        {
            if (vsRandom.Count < advanceWindow) return false;
            if (vsRandom.Rate(advanceWindow) < advanceRandomRate) return false;
        }
        else
        {
            if (vsRule.Count < advanceWindow) return false;
            if (vsRule.Rate(advanceWindow) < advanceRuleRate) return false;
        }
        phase++;
        vsRandom.Clear();
        vsRule.Clear();
        return true;
    }

    /// <summary>
    /// Consider a demotion, as a forgetting guard.
    ///
    /// Regressing on the rule-based rate is what stops phase 2 from quietly dropping the policy back to
    /// where it was: once the live opponent is strong enough that the fixed opponents stop being won,
    /// that shows up here as a sustained fall rather than being absorbed indefinitely.
    /// </summary>
    public bool TickRegress()
    {
        if (phase <= 0) return false;
        if (vsRule.Count < regressWindow) return false;
        if (vsRule.Rate(regressWindow) >= regressRuleRate) return false;
        phase--;
        vsRandom.Clear();
        vsRule.Clear();
        return true;
    }

    public float EvalWinRate(BehaviourSource opponent, int lastK) =>
        opponent == BehaviourSource.Random ? vsRandom.Rate(lastK) : vsRule.Rate(lastK);

    public int EvalSampleCount(BehaviourSource opponent) =>
        opponent == BehaviourSource.Random ? vsRandom.Count : vsRule.Count;

    /// <summary>Window the promotion gate reads, so the console line and the gate agree.</summary>
    public int AdvanceWindow => advanceWindow;

    /// <summary>One line for the console, so the curriculum's state is visible without the CSV.</summary>
    public override string ToString()
    {
        return $"{Role} phase {phase + 1}/{PhaseCount} "
               + $"(random {RandomShare:0.##} rule {RuleShare:0.##} live {LiveShare:0.##}) "
               + $"evalWR random {EvalWinRate(BehaviourSource.Random, advanceWindow):0.##}"
               + $"/{EvalSampleCount(BehaviourSource.Random)} "
               + $"rule {EvalWinRate(BehaviourSource.Rule, advanceWindow):0.##}"
               + $"/{EvalSampleCount(BehaviourSource.Rule)}";
    }

    /// <summary>
    /// Apply a <c>-mix.name=value</c> override. Returns false for an unknown field so the caller warns
    /// instead of silently reporting a sweep that tested nothing.
    /// </summary>
    public bool TrySetMix(string field, float value)
    {
        switch (field)
        {
            case "p1random": randomShare[0] = Mathf.Clamp01(value); return true;
            case "p1rule": ruleShare[0] = Mathf.Clamp01(value); return true;
            case "p2random": randomShare[1] = Mathf.Clamp01(value); return true;
            case "p2rule": ruleShare[1] = Mathf.Clamp01(value); return true;
            case "p3random": randomShare[2] = Mathf.Clamp01(value); return true;
            case "p3rule": ruleShare[2] = Mathf.Clamp01(value); return true;
            case "advanceRandom": advanceRandomRate = Mathf.Clamp01(value); return true;
            case "advanceRule": advanceRuleRate = Mathf.Clamp01(value); return true;
            case "regressRule": regressRuleRate = Mathf.Clamp01(value); return true;
            case "advanceWindow": advanceWindow = Mathf.Max(8, Mathf.RoundToInt(value)); return true;
            case "regressWindow": regressWindow = Mathf.Max(advanceWindow, Mathf.RoundToInt(value)); return true;
            default: return false;
        }
    }

    /// <summary>Current phase weights, for the run banner.</summary>
    public string DescribeShares() =>
        $"phase {phase + 1}/{PhaseCount}: random {RandomShare:0.##}, rule {RuleShare:0.##}, live {LiveShare:0.##}";
}