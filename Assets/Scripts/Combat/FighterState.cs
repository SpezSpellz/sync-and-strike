using System;
using UnityEngine;

public enum CombatState
{
    Idle,
    Walk,
    Attack,
    Dash,
    Jump,
    Block,
    Landing,
    HurtGrounded,
    HurtAerial,
    Knockdown,
    HardKnockdown,
    WallSlam,
    Fall,
}

/// <summary>
/// One fighter state: Enter / Step / Exit.
///
/// The important architectural rule, and the reason this class exists: **movement force is owned
/// by the state, never by the animation data.** A hurt state simply stops applying the
/// previous move's force because the state machine replaced it. Previously our animator applied
/// `continuousImpulse` every frame, so a looping walk kept pushing the fighter through its own
/// knockback.
/// </summary>
public abstract class FighterState
{
    protected readonly CharacterController owner;
    protected readonly CharacterPhysics physics;
    private int tick;

    protected FighterState(CharacterController owner)
    {
        this.owner = owner;
        this.physics = owner.Physics;
    }

    public int Tick => tick;
    public CharacterController Owner => owner;

    /// <summary>Which state this is, for transitions and the AI.</summary>
    public abstract CombatState Id { get; }

    public virtual void Enter() { }
    public virtual void Exit() { }

    /// <summary>Called once per simulation frame, before the position is integrated.</summary>
    public virtual void Step() { tick++; }

    /// <summary>True for HurtGrounded/HurtAerial/Knockdown/WallSlam.</summary>
    public virtual bool IsHurt => false;

    /// <summary>Whether the fighter may start a new move from here.</summary>
    public virtual bool CanAct => true;

    public virtual bool IsBlocking => false;

    /// <summary>Remaining hitstun, surfaced to the AI and the preview system.</summary>
    public virtual int HitstunRemaining => 0;

    /// <summary>Reset on every state entry.</summary>
    protected void ResetTick() { tick = 0; }

    /// <summary>Advances the animation one frame and reports whether it just finished.</summary>
    protected bool AdvanceAnimation()
    {
        return owner.Anim.StepFrame();
    }

    /// <summary>
    /// Advances the animation and reports completion. Used by move states that end when their
    /// animation runs out.
    /// </summary>
    protected void TickAnimationUntilDone(Action onDone)
    {
        if (AdvanceAnimation()) onDone?.Invoke();
    }
}