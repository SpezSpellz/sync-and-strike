using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

/// <summary>
/// Owns the "rate your companion's move" loop.
///
/// Flow: the companion commits a move -> the turn resolves -> the player votes good or bad ->
/// the vote plus the state snapshot and the turn outcome are appended to a JSONL training
/// log -> the next planning phase starts.
///
/// The log format is one JSON object per line so it can be streamed straight into a
/// Python/pandas training script later without any conversion step.
/// </summary>
public class CompanionVoteManager : MonoBehaviour
{
    [Serializable]
    public class VoteRecord
    {
        public int turn;
        public float timestamp;
        public string companion;
        public string moveId;
        public bool flipped;
        public float jumpPower;
        public float jumpAngle;
        public float diPower;
        public float diAngle;
        public string rationale;
        public bool voteGood;
        public bool timedOut;
        public float damageDealt;
        public float damageTaken;
        public bool hitLanded;
        public float[] features;
    }

    private static CompanionVoteManager instance;

    /// <summary>
    /// Self-bootstrapping on purpose: the turn loop must never deadlock waiting on a rating
    /// prompt that has no manager, so the manager is created on demand if it was not added
    /// to the scene as a component.
    /// </summary>
    public static CompanionVoteManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindAnyObjectByType<CompanionVoteManager>();
                if (instance == null)
                {
                    var go = new GameObject("CompanionVoteManager");
                    instance = go.AddComponent<CompanionVoteManager>();
                }
            }
            return instance;
        }
    }

    [Header("Behaviour")]
    [Tooltip("Prompt the player after each companion turn.")]
    private bool votingEnabled = true;

    [Tooltip("Seconds before the prompt auto-resolves. 0 disables the timer.")]
    private float voteTimeout = 12f;

    [Tooltip("Also log the label with the turn outcome appended, for offline reward shaping.")]
    private bool logOutcomes = true;

    private bool available = true;
    private CompanionController companion;
    private AIDecision pendingDecision;
    private AIDecisionContext pendingContext;
    private int turnCounter = 0;
    private float pendingCompanionHealth;
    private float pendingTargetHealth;

    private string logPath;

    private readonly List<VoteRecord> history = new List<VoteRecord>();

    private void Awake()
    {
        if (instance != null && instance != this)
        {
            available = false;
            enabled = false;
            return;
        }
        instance = this;
        logPath = Path.Combine(Application.persistentDataPath, "companion_votes.jsonl");
    }

    /// <summary>Whether the rating prompt should be shown at all.</summary>
    public bool VotingEnabled => votingEnabled;

    /// <summary>
    /// Called by TurnManager's backstop when the prompt has not resolved in time. Records the
    /// turn as skipped so the next planning phase can start without blocking the game.
    /// </summary>
    public void ForceResolveAsSkipped()
    {
        if (companion == null) return;
        CompanionVoteUI.Instance.ForceResolve();
    }

    public string LogPath => logPath;
    public IReadOnlyList<VoteRecord> History => history;
    public bool IsAwaitingVote => companion != null && CompanionVoteUI.Instance.IsWaiting;

    /// <summary>Called by TurnManager right before a new planning phase begins.</summary>
    public void PrepareForTurn(CompanionController companionController)
    {
        if (!available || !votingEnabled) return;
        this.companion = companionController;
    }

    /// <summary>
    /// Shows the rating prompt for the companion's move from the turn that just finished.
    /// <paramref name="turnResult"/> is filled in once the player votes.
    /// </summary>
    public void BeginVoteFor(CompanionController companionController, AIDecision decision,
        AIDecisionContext context, float timeout)
    {
        if (!available || !votingEnabled) return;
        if (companionController == null || companionController.IsDead()) return;

        this.companion = companionController;
        this.pendingDecision = decision;
        this.pendingContext = context;
        this.pendingCompanionHealth = companionController.GetHealth();

        var target = companionController.ResolveTargetForUI();
        this.pendingTargetHealth = target != null ? target.GetHealth() : 0f;

        CompanionVoteUI.Instance.Show(decision.moveId, timeout > 0f ? timeout : voteTimeout, OnVoteResolved);
    }

    private void OnVoteResolved(bool isGood, bool timedOut)
    {
        var record = new VoteRecord
        {
            turn = ++turnCounter,
            timestamp = Time.time,
            companion = companion != null ? companion.name : "companion",
            moveId = pendingDecision.moveId,
            flipped = pendingDecision.flipped,
            jumpPower = pendingDecision.jumpPower,
            jumpAngle = pendingDecision.jumpAngle,
            diPower = pendingDecision.diPower,
            diAngle = pendingDecision.diAngle,
            rationale = pendingDecision.rationale,
            voteGood = isGood,
            timedOut = timedOut,
            features = pendingContext.ToFeatureVector(),
        };

        if (logOutcomes)
        {
            float companionNow = companion != null ? companion.GetHealth() : 0f;
            record.damageTaken = Mathf.Max(0f, pendingCompanionHealth - companionNow);
            var target = companion != null ? companion.ResolveTargetForUI() : null;
            float targetNow = target != null ? target.GetHealth() : 0f;
            record.damageDealt = Mathf.Max(0f, pendingTargetHealth - targetNow);
            record.hitLanded = record.damageDealt > 0f;
        }

        history.Add(record);
        AppendToLog(record);

        // The vote never blocks the game: resume planning immediately after it resolves.
        TurnManager.Instance.ResumeAfterVote();
    }

    private void AppendToLog(VoteRecord record)
    {
        try
        {
            File.AppendAllText(logPath, JsonUtility.ToJson(record) + "\n");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not append companion vote to {logPath}: {e.Message}");
        }
    }

    /// <summary>Aggregate stats, handy for checking the AI is improving during playtesting.</summary>
    public string Summary()
    {
        if (history.Count == 0) return "No companion votes recorded yet.";
        int good = history.Count(v => v.voteGood);
        float avgDamage = history.Average(v => v.damageDealt);
        return $"{history.Count} votes, {good} good ({100f * good / history.Count:0}% approval), " +
               $"avg damage dealt {avgDamage:0.00}. Log: {logPath}";
    }
}