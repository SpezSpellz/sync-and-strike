using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// On-device learner for the companion. Records one transition per turn (observation, the action
/// the companion played, and a reward derived from what happened), runs PPO updates on the
/// player's machine, and persists the policy to disk so the companion keeps its learned behaviour
/// across sessions.
///
/// The enemy is deliberately NOT part of this system; it stays a frozen rule-based / (later) frozen
/// ONNX policy. Only the companion adapts, and only to this player's own preferences.
///
/// Reward is shaped from the turn outcome (damage dealt/taken, hit landed, kill) and the player's
/// good/bad vote. The vote arrives one step later than the reward, so it is applied as a late
/// adjustment to the previous transition.
/// </summary>
public class CompanionLearner
{
    /// <summary>Reward weights for a resolved turn.</summary>
    [Serializable]
    public class RewardConfig
    {
        public float hitLanded = 0.5f;
        public float damageDealt = 0.02f;   // per health point
        public float damageTaken = 0.03f;   // per health point
        public float killBonus = 5f;
        public float voteGood = 1.5f;
        public float voteBad = 1.5f;
    }

    private class Session
    {
        public CompanionController owner;
        public NeuralNetwork net;
        public PPOTrainer trainer;
        public NeuralPolicy policy;
        public bool warmStart = true;
        public int turns = 0;
        public int updates = 0;
        public bool hasPending;
        public float pendingSelfHealth;
        public float pendingTargetHealth;
        public float lastReward;
    }

    private static readonly Dictionary<int, Session> sessions = new Dictionary<int, Session>();

    // Warm start: for the first N turns the companion plays the rule-based expert and clones it,
    // so it begins competent rather than random, then PPO takes over.
    private const int WarmStartTurns = 200;
    private const int Hidden = 32;
    private const float CloneLearningRate = 1e-3f;

    private static readonly List<float[]> cloneObs = new List<float[]>();
    private static readonly List<int> cloneActions = new List<int>();

    private static RewardConfig rewardConfig = new RewardConfig();

    /// <summary>Reward weights, exposed so they can be tuned from the inspector in play.</summary>
    public static RewardConfig Config => rewardConfig;

    /// <summary>Create (or fetch) the PPO policy for a companion and wire it to the turn loop.</summary>
    public static NeuralPolicy CreatePolicy(CompanionController companion, FighterAI ruleBrain)
    {
        var key = companion.GetInstanceID();
        if (sessions.TryGetValue(key, out var existing))
            return existing.policy;

        int obsSize = AIDecisionContext.FeatureNames().Length;
        int actionCount = Mathf.Max(2, companion.Data.animations != null ? companion.Data.animations.Length * 2 + 2 : 4);
        // Exact action count depends on the unique move ids; compute it from a throwaway policy.
        var probe = new NeuralPolicy(companion, ruleBrain, new NeuralNetwork(obsSize, Hidden, 2, 1), new System.Random(1));
        actionCount = probe.ActionCount;

        var net = new NeuralNetwork(obsSize, Hidden, actionCount, companion.GetInstanceID());
        LoadWeights(net);

        var trainer = new PPOTrainer(net, new PPOTrainer.Hyper(), new System.Random(companion.GetInstanceID()));
        var policy = new NeuralPolicy(companion, ruleBrain, net, new System.Random(companion.GetInstanceID()));
        policy.ForceExpert = true; // warm start: play the expert until enough samples exist

        var session = new Session { owner = companion, net = net, trainer = trainer, policy = policy };
        sessions[key] = session;

        var turnMgr = Arena.Of(companion)?.turnManager ?? TurnManager.Instance;
        turnMgr.TurnResolved += () => OnTurnResolved(key);
        CompanionVoteManager.VoteResolved += (isGood, timedOut) => OnVoteResolved(key, isGood, timedOut);

        return policy;
    }

    /// <summary>Called right after the companion commits a move for the turn.</summary>
    public static void NotifyDecision(CompanionController companion)
    {
        if (!sessions.TryGetValue(companion.GetInstanceID(), out var s)) return;

        // Apply any pending PPO update now, after the previous turn's vote has been folded in.
        if (!s.warmStart && s.trainer.ReadyToUpdate())
        {
            s.trainer.Update();
            s.updates++;
            if (s.updates % 25 == 0) SaveWeights(s.net);
        }

        var p = s.policy;
        s.hasPending = true;
        s.pendingSelfHealth = p.LastSelfHealth;
        s.pendingTargetHealth = p.LastTargetHealth;

        if (s.warmStart)
        {
            cloneObs.Add(p.LastObs);
            cloneActions.Add(p.LastExpertAction);
            if (cloneObs.Count >= 32)
            {
                CloneBatch(s);
                cloneObs.Clear();
                cloneActions.Clear();
            }
        }
    }

    private static void OnTurnResolved(int key)
    {
        if (!sessions.TryGetValue(key, out var s) || !s.hasPending) return;
        s.hasPending = false;

        var companion = s.owner;
        if (companion == null) return;

        var target = companion.ResolveTargetForTraining();
        var ally = companion.ResolveAllyForTraining();
        float selfNow = companion.GetHealth();
        float targetNow = target != null ? target.GetHealth() : 0f;

        float dmgDealt = Mathf.Max(0f, s.pendingTargetHealth - targetNow);
        float dmgTaken = Mathf.Max(0f, s.pendingSelfHealth - selfNow);

        float reward = 0f;
        if (dmgDealt > 0f) reward += rewardConfig.hitLanded;
        reward += rewardConfig.damageDealt * dmgDealt;
        reward -= rewardConfig.damageTaken * dmgTaken;
        if (target != null && target.IsDead()) reward += rewardConfig.killBonus;

        bool done = companion.IsDead() || (target != null && target.IsDead());

        var nextObs = s.policy.CaptureObservation(companion, target, ally);
        s.trainer.Add(s.policy.LastObs, nextObs, s.policy.LastAction,
                      s.policy.LastLogProb, s.policy.LastValue, reward, done);
        s.lastReward = reward;
        s.turns++;

        if (s.warmStart && s.turns >= WarmStartTurns)
        {
            // Enough expert data collected; hand control to the learned policy.
            s.warmStart = false;
            s.policy.ForceExpert = false;
        }
    }

    private static void OnVoteResolved(int key, bool isGood, bool timedOut)
    {
        if (!sessions.TryGetValue(key, out var s)) return;
        // A timeout is not a signal: the player did not actually judge the move, so it must not
        // push the policy either way. Only a real good/bad vote counts.
        if (timedOut) return;
        float delta = isGood ? rewardConfig.voteGood : -rewardConfig.voteBad;
        s.trainer.AdjustLastReward(delta);
    }

    /// <summary>Supervised clone of the rule-based expert's moves onto the network.</summary>
    private static void CloneBatch(Session s)
    {
        if (cloneObs.Count == 0) return;
        s.net.ClearGradients();
        for (int i = 0; i < cloneObs.Count; i++)
        {
            s.net.Forward(cloneObs[i]);
            // Maximising log pi(expert) is exactly dLogProb = 1 on the expert action.
            s.net.Backprop(cloneObs[i], cloneActions[i], 1f, 0f, 0f);
        }
        s.net.ApplyGradients(CloneLearningRate / cloneObs.Count);
    }

    // --- Persistence ---------------------------------------------------------

    private static string SavePath => Path.Combine(Application.persistentDataPath, "companion_policy.json");

    private static void SaveWeights(NeuralNetwork net)
    {
        try
        {
            var json = JsonUtility.ToJson(net.Export());
            File.WriteAllText(SavePath, json);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not save companion policy: {e.Message}");
        }
    }

    private static void LoadWeights(NeuralNetwork net)
    {
        try
        {
            if (!File.Exists(SavePath)) return;
            var weights = JsonUtility.FromJson<NeuralNetwork.Weights>(File.ReadAllText(SavePath));
            net.Import(weights);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not load companion policy: {e.Message}");
        }
    }

    /// <summary>Force-save the current companion policy (e.g. on quit).</summary>
    public static void SaveAll()
    {
        foreach (var s in sessions.Values) SaveWeights(s.net);
    }
}