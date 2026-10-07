using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A PPO policy over the companion's move + facing choice, backed by <see cref="NeuralNetwork"/>.
///
/// Action space is a flat categorical over (move x facing), e.g. 8 moves * 2 = 16 actions. The
/// continuous parts of a turn (jump power/angle and knockback indirection power/angle) are left to
/// the rule-based helper, because they are direct functions of relative geometry and learning them
/// would add a continuous-action head for very little benefit. This keeps the learned part focused
/// on the decision that actually matters: what to do this turn.
///
/// The enemy does NOT use this class. It is frozen; only the companion adapts, on the player's own
/// machine.
/// </summary>
public class NeuralPolicy : FighterPolicy
{
    /// <summary>Stable action space, indexed as moveIndex * 2 + (facing left ? 1 : 0).</summary>
    public readonly string[] MoveIds;

    private readonly NeuralNetwork net;
    private readonly FighterAI helper;
    private readonly System.Random rng;

    /// <summary>Observation captured at the last decision, replayed when the reward arrives.</summary>
    public float[] LastObs { get; private set; }
    public int LastAction { get; private set; }
    public float LastLogProb { get; private set; }
    public float LastValue { get; private set; }
    public float LastSelfHealth { get; private set; }
    public float LastTargetHealth { get; private set; }

    // --- Hybrid policy: the scalar (Gaussian) half ---

    /// <summary>How many scalar heads this policy drives. Zero disables the whole mechanism.</summary>
    public int ScalarCount { get; }

    /// <summary>Per-head range descriptors and per-sample scratch, reused every decision.</summary>
    private readonly ContinuousHead[] heads;

    /// <summary>
    /// Raw pre-squash samples from the last decision. These are what the transition carries, because the
    /// Gaussian log-prob is a density over the raw value rather than over the gameplay value.
    /// </summary>
    public readonly float[] LastRawZ;

    /// <summary>Denormalised scalar values from the last decision, in gameplay units.</summary>
    public readonly float[] LastScalarValue;

    /// <summary>Categorical term of the joint log-prob from the last decision.</summary>
    public float LastCategoricalLogProb { get; private set; }

    /// <summary>Sum of the Gaussian terms of the joint log-prob from the last decision.</summary>
    public float LastScalarLogProb { get; private set; }

    // Current standard deviation of a scalar head, for telemetry.
    //
    // Logged because two or three of the four heads are INERT on a typical turn - jump geometry only
    // matters for jumps, DI only during hitstun - so their sigma is expected to drift upward while mu is
    // ignored. Harmless while the game clamps, but worth watching rather than inferring, and it is the
    // signal that would justify gating these heads or giving them a lower entropy weight.
    public float ScalarSd(int index)
    {
        if (index < 0 || index >= ScalarCount) return 0f;
        return Mathf.Exp(net.ScalarLogSd[index]);
    }

    // --- Telemetry -----------------------------------------------------------
    //
    // These exist to make policy collapse measurable instead of inferred. A policy that has
    // collapsed onto a single action looks exactly like a healthy one in the console; the difference
    // is that Entropy falls toward 0 and TopActionShare rises toward 1 while the reward plateaus.
    // Both are cheap to accumulate at sampling time and impossible to reconstruct later.

    /// <summary>Which role this policy trains as. Read by the telemetry writer.</summary>
    public string Role { get; set; } = "unknown";

    /// <summary>
    /// Shannon entropy of the last sampled distribution, in nats. Natural log of 1 is 0, so this
    /// falls toward 0 as the policy becomes confident. A value persistently near 0 next to a
    /// TopActionShare near 1 is the signature of collapse.
    /// </summary>
    public float Entropy { get; private set; }

    /// <summary>
    /// Fraction of the last batch of sampled actions that were the single most common one. Cheaper
    /// and more direct than entropy for spotting collapse: 1.0 means the policy picked one action
    /// every single turn, which is never healthy.
    /// </summary>
    public float TopActionShare { get; private set; }

    /// <summary>Reward from the most recently resolved turn, for spotting a reward that never moves.</summary>
    public float LastReward { get; set; }

    /// <summary>PPO updates applied. Mirrors the session counter for the CSV.</summary>
    public int Updates { get; set; }

    /// <summary>
    /// The trainer sharing this policy's network, for telemetry.
    ///
    /// Null for a policy that was not created by PolicyLearner. Two fighters on one network share
    /// ONE trainer, so these diagnostics describe the shared learner rather than a single body - which
    /// is the correct scope for them, unlike entropy and action share.
    /// </summary>
    public PPOTrainer Trainer { get; set; }

    /// <summary>Turns recorded into the rollout buffer.</summary>
    public int Turns { get; set; }

    /// <summary>
    /// How many (move, facing) pairs were legal on the last sampled turn.
    ///
    /// This exists to disambiguate a zero reading of <see cref="Entropy"/>. Entropy is measured over
    /// the MASKED distribution, and when the mask is down to a single legal action the softmax is
    /// exactly 1 for that action and 0 for everything else, so entropy is arithmetically 0 - the
    /// same number a collapsed network produces. A policy that can only legally do one thing has not
    /// collapsed at all.
    ///
    /// Read it alongside <see cref="EntropyUnmasked"/>:
    ///   entropy 0 + unmasked healthy + legal count low  -> restrictive mask, not collapse
    ///   entropy 0 + unmasked 0     + legal count high -> genuine collapse
    /// </summary>
    public float LegalActionCount { get; private set; }

    /// <summary>
    /// Entropy over the FULL action distribution with no masking applied.
    ///
    /// This is the policy's real confidence, uncontaminated by which moves happened to be legal this
    /// turn. It is the number that actually distinguishes "the network has saturated its logits" from
    /// "there was only one option".
    /// </summary>
    public float EntropyUnmasked { get; private set; }

    private const int ActionShareWindow = 128;
    private readonly int[] actionTally = new int[256];
    private int actionTallyCount;
    private int legalCountTally;

    /// <summary>
    /// Record one sampled action and refresh the diagnostics.
    ///
    /// Entropy is computed over the masked distribution because that is what the policy actually
    /// chose from: illegal moves were not available, so counting them would understate its real
    /// confidence. The unmasked entropy is computed alongside it precisely because that masking can
    /// drive the masked reading to zero on its own.
    /// </summary>
    private void TallyAction(int action, float[] maskedProbs, float[] unmaskedProbs, int legalCount)
    {
        if (actionTally.Length > 0 && action >= 0 && action < actionTally.Length) actionTally[action]++;
        actionTallyCount++;
        legalCountTally += legalCount;

        Entropy = Shannon(maskedProbs);
        EntropyUnmasked = Shannon(unmaskedProbs);

        int top = 0;
        for (int i = 1; i < actionTally.Length; i++)
            if (actionTally[i] > actionTally[top]) top = i;

        TopActionShare = actionTallyCount > 0 ? (float)actionTally[top] / actionTallyCount : 0f;
        // Averaged over the SAME window as TopActionShare, so the two are directly comparable.
        LegalActionCount = actionTallyCount > 0 ? (float)legalCountTally / actionTallyCount : 0f;

        // Roll the window once it is full so these track recent behaviour rather than the whole run;
        // a policy that collapsed late should read as collapsed, not diluted by its own healthy past.
        if (actionTallyCount >= ActionShareWindow)
        {
            Array.Clear(actionTally, 0, actionTally.Length);
            actionTallyCount = 0;
            legalCountTally = 0;
        }
    }

    private static float Shannon(float[] probs)
    {
        float entropy = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];
            if (p > 0f) entropy -= p * Mathf.Log(p);
        }
        return entropy;
    }

    public NeuralPolicy(CharacterController owner, FighterAI helper, NeuralNetwork net, System.Random rng)
        : this(owner, helper, net, rng, ActionScalars.Count)
    {
    }

    // scalarCount Gaussian heads to drive. Pass 0 for the move-only policy, which is the pre-hybrid
    // behaviour, still supported so this change can be A/B measured against it.
    public NeuralPolicy(CharacterController owner, FighterAI helper, NeuralNetwork net, System.Random rng,
                        int scalarCount)
    {
        this.helper = helper;
        this.net = net;
        this.rng = rng;
        MoveIds = BuildMoveIds(owner);
        ScalarCount = Mathf.Clamp(scalarCount, 0, ActionScalars.Count);
        heads = ActionScalars.CreateAll();
        LastRawZ = new float[ScalarCount];
        LastScalarValue = new float[ScalarCount];
    }

    /// <summary>
    /// The moves this fighter can choose between, in policy-index order.
    ///
    /// Public and static because the random behaviour source on <see cref="AIController"/> needs the
    /// SAME list. It samples over exactly the action space the network was trained on, so a random
    /// opponent is drawing from the same (move x facing) grid rather than from some looser notion of
    /// "a random move" - which matters, because a curriculum that graduates into phase 2 is only
    /// meaningful if the policy has already seen the opponent's action distribution.
    /// </summary>
    public static string[] BuildMoveIds(CharacterController owner)
    {
        var list = new List<string>();
        var seen = new HashSet<string>();
        var animations = owner.Data.animations;
        if (animations != null)
        {
            foreach (var anim in animations)
            {
                if (anim == null || string.IsNullOrEmpty(anim.moveId)) continue;
                if (!seen.Add(anim.moveId)) continue;
                list.Add(anim.moveId);
            }
        }
        if (!seen.Contains("idle")) list.Insert(0, "idle");
        return list.ToArray();
    }

    public int ActionCount => MoveIds.Length * 2;

    /// <summary>1 where the (move, facing) pair is currently performable, 0 otherwise.</summary>
    public int[] BuildLegalMask(CharacterController self) => BuildLegalMask(self, MoveIds);

    /// <summary>
    /// Legality mask for an arbitrary move list. Shared with <see cref="AIController"/>'s random
    /// behaviour source so both index the action space identically.
    /// </summary>
    public static int[] BuildLegalMask(CharacterController self, string[] moveIds)
    {
        var mask = new int[moveIds.Length * 2];
        for (int i = 0; i < moveIds.Length; i++)
        {
            int legal = 0;
            var data = self.Data.HasMove(moveIds[i]) ? self.Data.GetMove(moveIds[i]) : null;
            if (data != null && self.CanUseMove(data)) legal = 1;
            if (legal == 0 && moveIds[i] == "idle") legal = 1; // idle is always performable
            mask[i * 2] = legal;
            mask[i * 2 + 1] = legal;
        }
        if (NeuralNetwork.FirstLegal(mask) == 0 && mask[0] == 0)
        {
            // Defensive: never leave the policy with no legal action at all.
            mask[0] = 1;
        }
        return mask;
    }

    /// <summary>
    /// Uniform draw over the legal entries of a mask.
    ///
    /// Not the same as <see cref="SampleFromLogits"/> with flat logits on purpose: this one ignores the
    /// network entirely, which is the entire point of a random opponent. It is seeded from the
    /// controller's own RNG so a run stays reproducible under the training seed.
    /// </summary>
    public static int SampleUniform(int[] mask, System.Random rng)
    {
        int legal = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i] != 0) legal++;
        if (legal <= 0) return NeuralNetwork.FirstLegal(mask);
        int pick = rng.Next(legal);
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] == 0) continue;
            if (pick == 0) return i;
            pick--;
        }
        return NeuralNetwork.FirstLegal(mask);
    }

    public override AIDecision Decide(CharacterController self, CharacterController target, CharacterController ally)
    {
        // RAW observation: the capture-time-scaled but otherwise un-normalised vector. This is what gets
        // stored on LastObs and handed to the trainer, so the rollout buffer holds exactly what the
        // network saw and PPO's stored-vs-recomputed log-prob comparison stays valid.
        var rawObs = AIDecisionContext.Capture(self, target, ally, default).ToFeatureVector();

        // Statistics are updated from the raw vector only, so they describe the distribution the policy
        // itself is visiting.
        PolicyLearner.Normalizer.Observe(Role, rawObs);

        // The vector actually fed to the network. Equal to rawObs when normalisation is off or not yet
        // trustworthy, so this is a no-op in the default configuration.
        var obs = PolicyLearner.Normalizer.Prepare(Role, rawObs);
        var mask = BuildLegalMask(self);

        LastObs = obs;
        LastSelfHealth = self.GetHealth();
        LastTargetHealth = target != null ? target.GetHealth() : 0f;

        net.Forward(obs);
        LastValue = net.Value;
        var rawLogits = net.Logits;
        // Mask illegal actions before sampling so the policy can never pick an unperformable move.
        var logits = new float[ActionCount];
        for (int a = 0; a < ActionCount; a++) logits[a] = mask[a] != 0 ? rawLogits[a] : float.NegativeInfinity;

        int action = ClampAction(SampleFromLogits(logits, mask));
        LastAction = action;

        // --- Sample the scalars ---
        //
        // Always drawn so LastRawZ is valid for the transition; the raw z is what the stored log-prob
        // describes and it must belong to the action that was actually played.
        SampleScalars();

        // Store the log-prob over the FULL (unmasked) categorical PLUS the Gaussian terms, because
        // that joint sum is what PPOTrainer recomputes during the update. Using the masked log-prob here
        // would make the importance ratio exp(newLogProb - storedLogProb) compare two different
        // distributions and silently corrupt every gradient.
        float catLogProb = LogProbFromLogits(rawLogits, action);
        float scalarLogProb = 0f;
        for (int s = 0; s < ScalarCount; s++) scalarLogProb += heads[s].LogProbRaw(LastRawZ[s]);
        LastLogProb = catLogProb + scalarLogProb;

        // The split is recorded so the balance between the two halves of the joint log-prob stays
        // visible. If the Gaussian terms dominate, the importance ratio becomes mostly about aim and
        // the effective clipping on the MOVE head changes - which would look like the policy learning to
        // aim well while quietly getting worse at choosing moves.
        LastCategoricalLogProb = catLogProb;
        LastScalarLogProb = scalarLogProb;

        // Diagnostics are measured on the MASKED distribution: those are the moves the policy could
        // actually pick, so including the illegal ones would understate its real confidence. The
        // UNMASKED entropy and the legal-action count are recorded alongside it because a tight mask
        // can drive the masked entropy to zero on its own, which is indistinguishable from collapse.
        int legalCount = 0;
        for (int i = 0; i < mask.Length; i++) legalCount += mask[i] != 0 ? 1 : 0;
        TallyAction(action, SoftmaxMasked(logits, mask), Softmax(rawLogits), legalCount);

        // The policy owns both halves of the action: the (move, facing) pair and all four scalars.
        // Without a scalar head (ScalarCount == 0) the rule brain still supplies jump/DI geometry,
        // which is the only sane default when the network cannot choose it.
        var decision = ScalarCount > 0
            ? new AIDecision
            {
                jumpPower = LastScalarValue[ActionScalars.JumpPower],
                jumpAngle = LastScalarValue[ActionScalars.JumpAngle],
                diPower = LastScalarValue[ActionScalars.DiPower],
                diAngle = LastScalarValue[ActionScalars.DiAngle],
            }
            : helper.Decide(self, target, ally);

        decision.moveId = MoveIds[action / 2];
        decision.flipped = (action % 2) == 1;
        decision.rationale = "ppo";
        return decision;
    }

    /// <summary>
    /// Draw one raw sample per scalar head, recording both the raw z and the denormalised value.
    ///
    /// The RAW z is what goes on the transition, not the gameplay value, because the Gaussian log-prob
    /// is a density over z. Storing the squashed value would force the trainer to invert the squash to
    /// recover z, and the clamping in that round trip would move the action the log-prob describes.
    /// </summary>
    private void SampleScalars()
    {
        if (ScalarCount == 0) return;
        var means = net.ScalarMeans;
        var logSds = net.ScalarLogSd;
        for (int s = 0; s < ScalarCount; s++)
        {
            heads[s].mu = means[s];
            heads[s].logSd = logSds[s];
            LastRawZ[s] = heads[s].SampleZ(rng);
            LastScalarValue[s] = heads[s].Squash(LastRawZ[s]);
        }
    }

    /// <summary>
    /// Observation for the current instant, used as the next-state when a turn resolves.
    ///
    /// Goes through the normaliser exactly as Decide does. That matters more than it looks: GAE
    /// bootstraps the next state's value, and the value head was trained on normalised inputs, so
    /// handing it a differently-scaled vector would make the bootstrap inconsistent with the value it
    /// is bootstrapping from - and the resulting advantage estimate would be wrong even with a perfect
    /// critic.
    ///
    /// Statistics are deliberately NOT updated here. This is the same instant already observed at
    /// Decide time, so counting it again would double-weight that observation.
    /// </summary>
    public float[] CaptureObservation(CharacterController self, CharacterController target, CharacterController ally)
    {
        var raw = AIDecisionContext.Capture(self, target, ally, default).ToFeatureVector();
        return PolicyLearner.Normalizer.Prepare(Role, raw);
    }

    private int ClampAction(int a)
    {
        if (a < 0) return 0;
        if (a >= ActionCount) return ActionCount - 1;
        return a;
    }

    private int SampleFromLogits(float[] logits, int[] mask)
    {
        float max = float.NegativeInfinity;
        for (int a = 0; a < ActionCount; a++) if (logits[a] > max) max = logits[a];
        float sum = 0f;
        for (int a = 0; a < ActionCount; a++) if (logits[a] > float.NegativeInfinity) sum += Mathf.Exp(logits[a] - max);
        if (sum <= 0f) return NeuralNetwork.FirstLegal(mask);
        float r = (float)rng.NextDouble() * sum;
        for (int a = 0; a < ActionCount; a++)
        {
            if (logits[a] == float.NegativeInfinity) continue;
            r -= Mathf.Exp(logits[a] - max);
            if (r <= 0f) return a;
        }
        return NeuralNetwork.FirstLegal(mask);
    }

    private float LogProbFromLogits(float[] logits, int action)
    {
        float max = float.NegativeInfinity;
        for (int a = 0; a < ActionCount; a++) if (logits[a] > max) max = logits[a];
        float sum = 0f;
        for (int a = 0; a < ActionCount; a++) if (logits[a] > float.NegativeInfinity) sum += Mathf.Exp(logits[a] - max);
        return (logits[action] - max) - Mathf.Log(Mathf.Max(sum, 1e-8f));
    }

    /// <summary>
    /// Softmax over the legal actions only, for diagnostics. Numerically stabilised by subtracting
    /// the max, and illegal actions get exactly 0 rather than a tiny residue, so a policy that can
    /// only legally do one thing reports zero entropy instead of a misleading small value.
    /// </summary>
    private float[] SoftmaxMasked(float[] maskedLogits, int[] mask)
    {
        var probs = new float[ActionCount];
        float max = float.NegativeInfinity;
        for (int a = 0; a < ActionCount; a++)
            if (mask[a] != 0 && maskedLogits[a] > max) max = maskedLogits[a];

        float sum = 0f;
        for (int a = 0; a < ActionCount; a++)
        {
            if (mask[a] == 0 || maskedLogits[a] == float.NegativeInfinity) { probs[a] = 0f; continue; }
            probs[a] = Mathf.Exp(maskedLogits[a] - max);
            sum += probs[a];
        }
        if (sum <= 0f) return probs;
        for (int a = 0; a < ActionCount; a++) probs[a] /= sum;
        return probs;
    }

    /// <summary>
    /// Softmax over every action, ignoring legality. This is the policy's unconditional confidence,
    /// which is the only way to tell a saturated network from a fighter that simply had one legal
    /// move available.
    /// </summary>
    private float[] Softmax(float[] rawLogits)
    {
        var probs = new float[ActionCount];
        float max = float.NegativeInfinity;
        for (int a = 0; a < ActionCount; a++) if (rawLogits[a] > max) max = rawLogits[a];

        float sum = 0f;
        for (int a = 0; a < ActionCount; a++)
        {
            probs[a] = Mathf.Exp(rawLogits[a] - max);
            sum += probs[a];
        }
        if (sum <= 0f) return probs;
        for (int a = 0; a < ActionCount; a++) probs[a] /= sum;
        return probs;
    }
}