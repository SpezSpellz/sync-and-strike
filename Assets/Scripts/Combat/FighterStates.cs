using UnityEngine;

/// <summary>Standing still: normal friction and gravity.</summary>
public class IdleState : FighterState
{
    public IdleState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.Idle;

    public override void Enter()
    {
        ResetTick();
        owner.PlayIdleAnimation();
    }

    public override void Step()
    {
        base.Step();
        physics.ApplyForces(PhysicsConstants.GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
        owner.Anim.StepFrame();
    }
}

/// <summary>
/// Shared behaviour for hold actions: walk and block.
///
/// A hold reports its action complete once the minimum hold has elapsed, but keeps acting until
/// the turn actually resolves; TurnManager's ResolveTurn force-finishes everyone, which is what
/// ends the hold. That split is the whole point: reporting completion without the minimum would
/// end the turn on the very next frame and the hold would never visibly happen, while never
/// reporting at all would stall the turn until the frame cap.
///
/// Before this existed there was no completion path out of a hold at all, so every walk or block
/// froze the game for maxTurnFrames.
/// </summary>
public abstract class HoldState : FighterState
{
    protected readonly AnimationData move;
    private int holdFrames;
    private bool reported;

    protected HoldState(CharacterController owner, AnimationData move) : base(owner)
    {
        this.move = move;
    }

    public override void Enter()
    {
        ResetTick();
        reported = false;
        // firstActionable is authored per move (walkf 10, block 5) and already means "how long
        // this move must play before it can be acted out of".
        holdFrames = Mathf.Max(1, move != null ? move.firstActionable : 1);
        owner.Anim.PlayMove(move, loop: true);
    }

    public override void Step()
    {
        base.Step();
        ApplyHoldForces();
        owner.Anim.StepFrame();

        // Report once, then keep acting. CompleteMove() would also drop us to idle, which is why
        // this reports without changing state.
        if (!reported && Tick >= holdFrames)
        {
            reported = true;
            owner.ReportActionComplete();
        }
    }

    /// <summary>Movement applied every frame while the hold is active.</summary>
    protected abstract void ApplyHoldForces();
}

/// <summary>
/// Walking: a small continuous force per frame, opposed by ground friction.
///
/// This is the state whose behaviour the old `continuousImpulse` flag used to provide, and the
/// reason walking used to ignore knockback. Because the force lives here, entering any hurt state
/// stops it immediately.
/// </summary>
public class WalkState : HoldState
{
    public WalkState(CharacterController owner, AnimationData move) : base(owner, move) { }

    public override CombatState Id => CombatState.Walk;

    protected override void ApplyHoldForces()
    {
        // Force owned by the state, applied only while this state is active.
        physics.AddVelocity(owner.FacingVector * move.impulse.x);
        physics.ApplyForces(PhysicsConstants.GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
    }
}

/// <summary>
/// Guarding. The block state reduces knockback and applies chip damage, and it is held for as
/// long as the turn lasts.
/// </summary>
public class BlockState : HoldState
{
    public BlockState(CharacterController owner, AnimationData move) : base(owner, move) { }

    public override CombatState Id => CombatState.Block;
    public override bool IsBlocking => true;
    public override bool CanAct => true;

    protected override void ApplyHoldForces()
    {
        physics.ApplyForces(PhysicsConstants.GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
    }
}

/// <summary>
/// A committed attack (or dash). Plays the animation, applies the move impulse once on entry and
/// lets the hurt system interrupt it.
/// </summary>
public class AttackState : FighterState
{
    private readonly AnimationData move;
    private readonly CombatState id;

    /// <summary>
    /// <paramref name="id"/> distinguishes a dash from a normal attack. Both share this class
    /// because they behave identically apart from friction, but a dash gets its own lighter
    /// friction (0.05) and speed limit (40), which is a real difference in how far the move
    /// carries the fighter.
    /// </summary>
    public AttackState(CharacterController owner, AnimationData move, CombatState id = CombatState.Attack)
        : base(owner)
    {
        this.move = move;
        this.id = id;
    }

    public override CombatState Id => id;
    public override bool CanAct => false;

    public override void Enter()
    {
        ResetTick();
        owner.Anim.PlayMove(move, loop: false);
        // Attacks carry the character forward via a force applied on entry.
        if (move.impulse != Vector2.zero) physics.ApplyImpulse(move.impulse);
    }

    public override void Step()
    {
        base.Step();

        // A dash uses 0.05 friction and a 40px speed limit over its own frames, which
        // is what lets a dash outrun the normal ground friction. Attack moves use the default.
        if (Id == CombatState.Dash)
        {
            physics.ApplyForces(PhysicsConstants.DASH_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
            var v = physics.getVelocity();
            physics.setVelocity(
                Vector2.ClampMagnitude(v, PhysicsConstants.DASH_SPEED_LIMIT).x,
                v.y);
        }
        else
        {
            physics.ApplyForces(PhysicsConstants.GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
        }

        if (owner.Anim.StepFrame()) owner.CompleteMove();
    }
}

/// <summary>
/// Jumping. The jump force is applied from the animation's Jump frame event so that the burst
/// happens on the correct frame, so the burst is frame-scheduled.
/// </summary>
public class JumpState : FighterState
{
    private readonly AnimationData move;
    private bool jumped;

    public JumpState(CharacterController owner, AnimationData move) : base(owner)
    {
        this.move = move;
    }

    public override CombatState Id => CombatState.Jump;
    public override bool CanAct => false;

    public override void Enter()
    {
        ResetTick();
        jumped = false;
        owner.Anim.PlayMove(move, loop: false);
        if (move.impulse != Vector2.zero) physics.ApplyImpulse(move.impulse);
    }

    /// <summary>Called by the animation's Jump frame event.</summary>
    public void ApplyJumpForce(Vector2 velocity)
    {
        if (jumped) return;
        jumped = true;
        physics.AddVelocity(velocity);
    }

    public override void Step()
    {
        base.Step();
        // Airborne movement uses the light air friction so momentum is preserved.
        physics.ApplyForces(PhysicsConstants.AIR_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
        if (owner.Anim.StepFrame()) owner.CompleteMove();
    }
}

/// <summary>
/// Post-wall-slam recovery: normal air friction and gravity while dropping back
/// to the floor, then Landing. WallSlam hands off here when its duration elapses in mid-air.
/// </summary>
public class FallState : FighterState
{
    public FallState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.Fall;
    public override bool CanAct => false;

    public override void Enter()
    {
        ResetTick();
        // Resumes the normal idle animation rather than a dedicated fall anim.
        owner.PlayIdleAnimation();
    }

    public override void Step()
    {
        base.Step();
        // Uses plain gravity plus forces, i.e. the *limited* pass.
        physics.ApplyForces(PhysicsConstants.AIR_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
        owner.Anim.StepFrame();
        // Waits 3 ticks before checking, so the fighter clears the wall before landing.
        if (Tick > 3 && physics.IsGrounded) owner.SwitchTo(CombatState.Landing);
    }
}

/// <summary>
/// Landing recovery: a base of 4 frames, plus up to 5 more depending on how hard the
/// character came down.
/// </summary>
public class LandingState : FighterState
{
    private int remaining;

    public LandingState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.Landing;
    public override bool CanAct => false;

    public override void Enter()
    {
        ResetTick();
        // Heavier landings cost more recovery, up to the cap.
        float impact = Mathf.Clamp01(Mathf.Abs(physics.getVelocity().y) / Mathf.Max(0.001f, PhysicsConstants.MAX_FALL_SPEED));
        int extra = Mathf.RoundToInt(impact * PhysicsConstants.MAX_EXTRA_LANDING_LAG);
        remaining = PhysicsConstants.LANDING_LAG + extra;
    }

    public override void Step()
    {
        base.Step();
        physics.ApplyForces(PhysicsConstants.GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY, physics.IsGrounded);
        remaining--;
        if (remaining <= 0) owner.CompleteMove();
    }
}