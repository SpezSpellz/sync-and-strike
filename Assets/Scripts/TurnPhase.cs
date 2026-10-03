public enum TurnPhase
{
    Planning,
    Simulating,
    /// <summary>Turn resolved, waiting for the player to rate the companion's move.</summary>
    AwaitingVote,
    Resolved
}