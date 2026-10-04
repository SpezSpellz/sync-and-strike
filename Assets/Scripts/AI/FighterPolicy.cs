using UnityEngine;

/// <summary>
/// A pluggable decision brain for an AI fighter. <see cref="AIController"/> only ever talks to this
/// interface, so the concrete brain can be swapped without touching the turn loop.
///
/// Three implementations exist (or are planned):
/// <list type="bullet">
/// <item><see cref="FighterAI"/> — the rule-based brain. Used by the frozen enemy and as the
/// default / expert for warm-starting.</item>
/// <item><see cref="NeuralPolicy"/> + <see cref="PPOTrainer"/> — an on-device PPO policy the
/// companion continues to train against the player's own preferences.</item>
/// <item>(future) an ML-Agents ONNX policy trained offline and shipped frozen with the enemy.</item>
/// </list>
///
/// All policies share the same signature so they can be dropped in interchangeably.
/// </summary>
public abstract class FighterPolicy
{
    /// <summary>
    /// Produce one turn's intent. <paramref name="target"/> is the enemy to fight and
    /// <paramref name="ally"/> the friendly fighter (used only for spacing); either may be null.
    /// </summary>
    public abstract AIDecision Decide(CharacterController self, CharacterController target, CharacterController ally);
}