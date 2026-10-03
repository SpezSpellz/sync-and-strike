using UnityEngine;

/// <summary>
/// Owns a fighter's current state and performs transitions. Exiting a state always runs before
/// entering the next one, so
/// a move can never leak force into a hurt state.
/// </summary>
public class FighterStateMachine
{
    private readonly CharacterController owner;
    private FighterState current;
    private FighterState queued;

    public FighterState Current => current;
    public CombatState CurrentId => current != null ? current.Id : CombatState.Idle;

    public FighterStateMachine(CharacterController owner)
    {
        this.owner = owner;
    }

    /// <summary>Immediately changes state, running Exit on the old state first.</summary>
    public void Change(FighterState next)
    {
        if (next == null) return;
        if (current != null && current.GetType() == next.GetType())
        {
            // Re-entering the same state still resets its timers: a transition
            // always exits before entering.
        }
        current?.Exit();
        // Body collision is restored on every state exit, so a hit that cleared it only
        // suppresses body collision for the duration of that hurt state.
        owner.RestoreOpponentCollision();
        current = next;
        current.Enter();
        owner.OnStateChanged(current);
    }

    /// <summary>Queues a transition for the end of the current frame.</summary>
    public void Queue(FighterState next)
    {
        queued = next;
    }

    public void Step()
    {
        current?.Step();
        if (queued != null)
        {
            var next = queued;
            queued = null;
            Change(next);
        }
    }

    /// <summary>
    /// The current state, or a null-object idle when nothing has been entered yet. The UI can ask
    /// whether a move is usable before the fighter has entered its first state, so this must not
    /// throw.
    /// </summary>
    public FighterState CurrentOrIdle()
    {
        return current ?? new IdleState(owner);
    }

    public void ForceExit()
    {
        current?.Exit();
        current = null;
    }

    // --- Factory -----------------------------------------------------------
    // States are created here so callers never need to know which class implements which id.

    /// <summary>Builds the state for a move id, wiring up the move's animation data.</summary>
    public FighterState CreateForMove(CombatState id, AnimationData move)
    {
        switch (id)
        {
            case CombatState.Idle: return new IdleState(owner);
            case CombatState.Walk: return new WalkState(owner, move);
            case CombatState.Attack: return new AttackState(owner, move, CombatState.Attack);
            case CombatState.Dash: return new AttackState(owner, move, CombatState.Dash);
            case CombatState.Jump: return new JumpState(owner, move);
            case CombatState.Block: return new BlockState(owner, move);
            case CombatState.Landing: return new LandingState(owner);
            case CombatState.HurtGrounded: return new HurtGroundedState(owner);
            case CombatState.HurtAerial: return new HurtAerialState(owner);
            case CombatState.Knockdown: return new KnockdownState(owner);
            case CombatState.HardKnockdown: return new HardKnockdownState(owner);
            case CombatState.WallSlam: return new WallSlamState(owner);
            case CombatState.Fall: return new FallState(owner);
            default: return new IdleState(owner);
        }
    }
}