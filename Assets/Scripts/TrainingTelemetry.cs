using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Writes a training run's progress to a CSV so it can be graphed and tuned.
///
/// Exists because the console told us a policy had collapsed but not WHY. Everything this records is
/// a signal that distinguishes the failure modes that look identical in the log:
///
///   * frames_per_turn near maxTurnFrames  -> the turn is stalling, a completion path is missing.
///   * entropy_mean near 0                 -> the policy has collapsed onto one action.
///   * action_share_top1 near 1            -> confirmed collapse; that one action is the tell.
///   * explained_variance below 0          -> the critic is worse than predicting the mean, so every
///                                            advantage estimate is noise and learning is random.
///   * clip_fraction far from ~0.1         -> updates are either too timid or diverging.
///   * whiff_rate high with damage_dealt 0 -> swinging at everything, which the current reward never
///                                            penalises.
///
/// One row per match per fighter, long format (one row per fighter rather than a wide row per match),
/// so every column is directly plottable against match number without unpivoting.
///
/// Rows are buffered and flushed periodically rather than written per match: a training run does
/// thousands of matches, and an unbuffered File.AppendAllText per match becomes a measurable share
/// of wall-clock on its own.
/// </summary>
public static class TrainingTelemetry
{
    private static readonly StringBuilder Buffer = new StringBuilder();
    private static string path;
    private static bool headerWritten;
    private static bool enabled = true;
    private static int bufferedRows;

    /// <summary>Header row. Kept in one place so the column order and the writer cannot drift.</summary>
    private const string Header =
        "match,arena,fighter,role,team,result,opponent,is_eval,is_learner,turns,frames_turn," +
        "damage_dealt_to_enemy,damage_taken,hits_landed,attacks_thrown,whiff_rate," +
        "health_left,reward_last,entropy,entropy_unmasked,legal_action_count," +
        "action_share_top1,updates,turns_recorded," +
        "explained_variance,approx_kl,clip_fraction,value_loss,advantage_magnitude," +
        "rejected_transitions,nonfinite_grad_steps,clipped_grad_steps,network_poisoned,max_abs_weight," +
        "logprob_categorical,logprob_scalar,jump_power,jump_angle,di_power,di_angle," +
        "sd_jump_power,sd_jump_angle,sd_di_power,sd_di_angle," +
        "epochs_run,kl_all_epochs";

    /// <summary>
    /// Open the run's CSV. The file is timestamped so successive runs do not overwrite each other's
    /// history, and so a run can be compared against the one before it.
    /// </summary>
    public static void Begin()
    {
        if (!enabled || path != null) return;
        try
        {
            string dir = Path.Combine(Application.persistentDataPath, "training_logs");
            Directory.CreateDirectory(dir);
            string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            path = Path.Combine(dir, $"training-{stamp}.csv");
            headerWritten = false;
            Debug.Log($"[Training] Telemetry writing to {path}");
        }
        catch (Exception e)
        {
            // Telemetry must never be the reason a training run dies.
            enabled = false;
            Debug.LogWarning($"[Training] Telemetry disabled: {e.Message}");
        }
    }

    /// <summary>Stop recording and flush. Safe to call more than once.</summary>
    public static void End()
    {
        Flush();
        path = null;
    }

    /// <summary>
    /// Append one row per fighter for a finished match.
    /// </summary>
    /// <param name="damageToEnemy">
    /// Per-fighter credit for the enemy's missing health. Only meaningful for the allies; the enemy
    /// row gets its own total, which is the sum of what the allies did to it.
    /// </param>
    /// <param name="allySource">
    /// What the ally team was playing this match. Each row records the source of the side OPPOSING it,
    /// so an ally row gets <paramref name="enemySource"/> and the enemy row gets this. Two parameters
    /// rather than one because the two sides can be driven by different sources in the same match -
    /// that is the whole point of the curriculum.
    /// </param>
    /// <param name="isEval">
    /// True for the evaluation slice, where neither side recorded a transition. These are the only rows
    /// whose win rate measures generalisation rather than training performance, and they are what the
    /// curriculum's promotion gate reads.
    /// </param>
    public static void RecordMatch(
        ArenaRunHandle run,
        string result,
        int matchNumber,
        Dictionary<CharacterController, float> damageToEnemy,
        BehaviourSource allySource,
        BehaviourSource enemySource,
        bool isEval)
    {
        if (!enabled) return;
        if (string.IsNullOrEmpty(path)) Begin();
        if (string.IsNullOrEmpty(path)) return;

        try
        {
            if (!headerWritten)
            {
                Buffer.Append(Header).Append('\n');
                headerWritten = true;
            }

            foreach (var f in run.Everyone())
            {
                if (f == null) continue;
                var ai = f as AIController;
                var stats = f.CombatStats;
                float cred = damageToEnemy != null && damageToEnemy.TryGetValue(f, out var d) ? d : 0f;
                var policy = ai != null ? ai.PolicyForTelemetry : null;

                Buffer.Append(matchNumber).Append(',')
                      .Append(run.Index).Append(',')
                      .Append(f.name).Append(',')
                      .Append(policy != null ? policy.Role : "rules").Append(',')
                      .Append(f.Team).Append(',')
                      .Append(result).Append(',')
                      // The source of the side OPPOSING this fighter, so win rates can be grouped by
                      // opponent without the reader having to know which team a row belongs to.
                      //
                      // The team test selects which of the two sources belongs to the OTHER side: an
                      // ally's opponent is the enemy, so an ally row reads enemySource, and the enemy
                      // row reads allySource. This was originally inverted, recording each fighter's
                      // OWN source - which made every learner row read "live" (its own behaviour) and
                      // made the column useless for measuring win rate against a given opponent.
                      .Append((f.Team == CombatTeam.Enemy ? allySource : enemySource).Token()).Append(',')
                      .Append(isEval ? 1 : 0).Append(',')
                      // Whether THIS fighter was producing training samples this match.
                      //
                      // Reads straight off the behaviour source rather than being passed in, because
                      // BehaviourSource.Live is already the exact definition of "recording". Deriving it
                      // keeps the two from ever disagreeing.
                      //
                      // This column exists because the alternative is genuinely misleading: every fighter
                      // gets a row in every match, including the ones acting from a scripted source, so
                      // `role` + `opponent` alone cannot distinguish a policy that played from one that
                      // was a sparring partner. Reading "companion vs live" as a companion result
                      // attributes scripted-random wins to the learned companion. Without this column the
                      // only way to recover the learner rows is to infer it from arena parity.
                      .Append(ai != null && ai.BehaviourSource.Records() ? 1 : 0).Append(',')
                      .Append(run.Turns).Append(',')
                      .Append(run.FramesThisTurnLastMatch).Append(',')
                      .Append(Num(cred)).Append(',')
                      .Append(Num(stats.damageTaken)).Append(',')
                      .Append(stats.hitsLanded).Append(',')
                      .Append(stats.attacksThrown).Append(',')
                      .Append(Num(stats.WhiffRate)).Append(',')
                      .Append(Num(f.GetHealth())).Append(',')
                      .Append(Num(policy != null ? policy.LastReward : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.Entropy : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.EntropyUnmasked : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LegalActionCount : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.TopActionShare : 0f)).Append(',')
                      .Append(policy != null ? policy.Updates : 0).Append(',')
                      .Append(policy != null ? policy.Turns : 0).Append(',')
                      .Append(Num(policy != null ? policy.Trainer.ExplainedVariance : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.Trainer.ApproxKl : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.Trainer.ClipFraction : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.Trainer.ValueLoss : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.Trainer.AdvantageMagnitude : 0f)).Append(',')
                      .Append(policy != null && policy.Trainer != null ? policy.Trainer.RejectedTransitionCount : 0).Append(',')
                      .Append(policy != null && policy.Trainer != null ? policy.Trainer.NonFiniteSteps : 0).Append(',')
                      .Append(policy != null && policy.Trainer != null ? policy.Trainer.ClippedSteps : 0).Append(',')
                      .Append(policy != null && policy.Trainer != null && policy.Trainer.NetworkIsPoisoned ? 1 : 0)
                      .Append(',')
                      .Append(Num(policy != null && policy.Trainer != null ? policy.Trainer.MaxAbsWeight : 0f))
                      .Append(',')
                      .Append(Num(policy != null ? policy.LastCategoricalLogProb : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LastScalarLogProb : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LastScalarValue[ActionScalars.JumpPower] : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LastScalarValue[ActionScalars.JumpAngle] : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LastScalarValue[ActionScalars.DiPower] : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.LastScalarValue[ActionScalars.DiAngle] : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.ScalarSd(ActionScalars.JumpPower) : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.ScalarSd(ActionScalars.JumpAngle) : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.ScalarSd(ActionScalars.DiPower) : 0f)).Append(',')
                      .Append(Num(policy != null ? policy.ScalarSd(ActionScalars.DiAngle) : 0f))
                      // Epochs actually run and the KL the early stop watches. Both are needed to tell
                      // "the policy is learning" from "the update overshot and was cut short", which
                      // approx_kl (epoch 0 only) cannot show on its own.
                      .Append(',').Append(policy != null && policy.Trainer != null ? policy.Trainer.EpochsRun : 0)
                      .Append(',').Append(Num(policy != null && policy.Trainer != null ? policy.Trainer.MeanKlAcrossEpochs : 0f))
                      .Append('\n');
            }

            if (++bufferedRows >= FlushEveryRows) Flush();
        }
        catch (Exception e)
        {
            enabled = false;
            Debug.LogWarning($"[Training] Telemetry disabled after a write error: {e.Message}");
        }
    }

    /// <summary>
    /// Rows buffered before a disk write.
    ///
    /// Kept small deliberately. An earlier value of 200 meant a short or failing run left an EMPTY
    /// file with no header, which is indistinguishable from "telemetry is broken" - the worst
    /// possible failure mode for a diagnostic tool. At 3 rows per match this is one write per match,
    /// which is negligible next to the cost of simulating a match.
    /// </summary>
    private const int FlushEveryRows = 3;

    /// <summary>
    /// A compact per-run summary line for the console, so a run's health is visible without opening
    /// the CSV. Deliberately small: this prints every report interval and a long line would drown
    /// the match and turn messages it sits between.
    /// </summary>
    public static void LogSnapshot<T>(IReadOnlyList<T> runs, int matches)
    {
        if (!enabled) return;
        double frames = 0, turns = 0;
        int counted = 0;
        // Summed across arenas via the generic handle so this does not depend on the runner's private
        // ArenaRun type.
        foreach (var r in runs)
        {
            var h = r as ArenaRunHandle;
            if (h == null) continue;
            frames += h.FramesThisTurnLastMatch;
            turns += h.Turns;
            counted++;
        }
        if (counted == 0) return;
        Debug.Log($"[Training] telemetry: {matches} matches, mean {frames / turns:0.0} frames/turn "
            + $"(a value near 180 means a turn is stalling).");
        Flush();
    }

    private static void Flush()
    {
        if (!headerWritten || Buffer.Length == 0) return;
        try
        {
            File.AppendAllText(path, Buffer.ToString());
            Buffer.Clear();
            bufferedRows = 0;
        }
        catch (Exception e)
        {
            enabled = false;
            Debug.LogWarning($"[Training] Telemetry flush failed, disabling: {e.Message}");
        }
    }

    /// <summary>Invariant-culture float, so a decimal comma cannot corrupt the column layout.</summary>
    private static string Num(float v) =>
        float.IsNaN(v) || float.IsInfinity(v) ? "0" : v.ToString("0.####", CultureInfo.InvariantCulture);

    /// <summary>
    /// Read-only view of a runner's ArenaRun, so telemetry can be written without depending on the
    /// runner's private nested type.
    /// </summary>
    public sealed class ArenaRunHandle
    {
        public int Index;
        public int Turns;
        public int FramesThisTurnLastMatch;
        private readonly Func<IEnumerable<CharacterController>> everyone;

        public ArenaRunHandle(int index, int turns, int frames, Func<IEnumerable<CharacterController>> everyone)
        {
            Index = index;
            Turns = turns;
            FramesThisTurnLastMatch = frames;
            this.everyone = everyone;
        }

        public IEnumerable<CharacterController> Everyone() => everyone();
    }
}
