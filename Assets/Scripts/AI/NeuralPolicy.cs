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
    public int LastExpertAction { get; private set; }
    public float LastSelfHealth { get; private set; }
    public float LastTargetHealth { get; private set; }

    /// <summary>When true the rule-based move is used instead of the sampled one (warm start).</summary>
    public bool ForceExpert { get; set; }

    public NeuralPolicy(CharacterController owner, FighterAI helper, NeuralNetwork net, System.Random rng)
    {
        this.helper = helper;
        this.net = net;
        this.rng = rng;
        MoveIds = BuildMoveIds(owner);
    }

    private static string[] BuildMoveIds(CharacterController owner)
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
    public int[] BuildLegalMask(CharacterController self)
    {
        var mask = new int[ActionCount];
        for (int i = 0; i < MoveIds.Length; i++)
        {
            int legal = 0;
            var data = self.Data.HasMove(MoveIds[i]) ? self.Data.GetMove(MoveIds[i]) : null;
            if (data != null && self.CanUseMove(data)) legal = 1;
            if (legal == 0 && MoveIds[i] == "idle") legal = 1; // idle is always performable
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

    public override AIDecision Decide(CharacterController self, CharacterController target, CharacterController ally)
    {
        // The rule brain supplies the jump/DI geometry and, during warm start, the expert move.
        var expert = helper.Decide(self, target, ally);

        var obs = AIDecisionContext.Capture(self, target, ally, default).ToFeatureVector();
        var mask = BuildLegalMask(self);

        LastObs = obs;
        LastSelfHealth = self.GetHealth();
        LastTargetHealth = target != null ? target.GetHealth() : 0f;

        int expertMove = IndexOfMove(expert.moveId);
        LastExpertAction = ClampAction(expertMove * 2 + (expert.flipped ? 1 : 0));

        LastValue = net.Forward(obs);
        var rawLogits = net.Logits;
        // Mask illegal actions before sampling so the policy can never pick an unperformable move.
        var logits = new float[ActionCount];
        for (int a = 0; a < ActionCount; a++) logits[a] = mask[a] != 0 ? rawLogits[a] : float.NegativeInfinity;

        int action;
        if (ForceExpert)
        {
            action = LastExpertAction;
        }
        else
        {
            action = SampleFromLogits(logits, mask);
        }
        action = ClampAction(action);
        LastAction = action;
        // Store the log-prob over the FULL (unmasked) distribution, because that is what
        // PPOTrainer recomputes during the update. Using the masked log-prob here would make the
        // importance ratio exp(newLogProb - storedLogProb) compare two different distributions and
        // silently corrupt every gradient.
        LastLogProb = LogProbFromLogits(rawLogits, action);

        var decision = expert;
        decision.moveId = MoveIds[action / 2];
        decision.flipped = (action % 2) == 1;
        decision.rationale = ForceExpert ? "ppo (warm start expert)" : "ppo";
        return decision;
    }

    /// <summary>Observation for the current instant, used as the next-state when a turn resolves.</summary>
    public float[] CaptureObservation(CharacterController self, CharacterController target, CharacterController ally)
    {
        return AIDecisionContext.Capture(self, target, ally, default).ToFeatureVector();
    }

    private int IndexOfMove(string moveId)
    {
        for (int i = 0; i < MoveIds.Length; i++) if (MoveIds[i] == moveId) return i;
        return 0;
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
}