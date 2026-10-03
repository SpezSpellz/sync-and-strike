using UnityEngine;

/// <summary>
/// Base for the hurt states. These never apply move force, which is exactly why entering one
/// stops a walk or dash from overriding the knockback.
/// </summary>
public abstract class HurtStateBase : FighterState
{
    protected int hitstun;
    protected bool wasGroundedOnHit;

    protected HurtStateBase(CharacterController owner) : base(owner) { }

    public override bool IsHurt => true;
    public override bool CanAct => false;
    public override int HitstunRemaining => hitstun;

    /// <summary>Knockback settled enough that the fighter is allowed to act again.</summary>
    protected bool Settled => hitstun <= 0 && Mathf.Abs(physics.getVelocity().x) < PhysicsConstants.GROUND_SETTLE_SPEED;
}

/// <summary>
/// A grounded victim is launched purely horizontally (vertical knockback is zeroed) and skids
/// with 0.05 friction. Touching a wall suppresses any further horizontal push so the fighter does
/// not jitter in the corner.
/// </summary>
public class HurtGroundedState : HurtStateBase
{
    public HurtGroundedState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.HurtGrounded;

    public override void Enter()
    {
        ResetTick();
        hitstun = owner.PendingHitstun;
        wasGroundedOnHit = true;
        physics.ZeroVerticalVelocity();
    }

    public override void Step()
    {
        base.Step();
        hitstun--;

        // 0.05 skid friction, normal gravity, and NO speed limit (hurt states use the unlimited
        // force pass, so a big grounded hit is not truncated by MAX_GROUND_SPEED).
        physics.ApplyForces(PhysicsConstants.HURT_GROUND_FRICTION_RATIO, PhysicsConstants.GRAVITY,
            physics.IsGrounded, limitSpeed: false);

        // Re-check the wall every tick and stop pushing into it.
        if (physics.TouchingWall) physics.ZeroHorizontalVelocity();

        if (!Settled) return;

        // This state does not itself transition to Knockdown; it ends on hitstun and hands
        // control back. A knockdown applied to a grounded victim only reaches Knockdown via
        // HurtAerial, which is why grounded knockdown moves must be authored as launchGrounded.
        owner.LeaveHurtState();
    }
}

/// <summary>
/// Uses half gravity (0.25), which is what makes launchers hang in the air instead of dropping
/// the victim immediately, bounces off walls at -0.85 plus 3 ticks of hitlag, and optionally
/// bounces off the ground.
/// </summary>
public class HurtAerialState : HurtStateBase
{
    // A 4-tick countdown. This was previously a bool, so a ground bounce lasted one frame
    // instead of four and the victim could be walked out of before the bounce finished.
    private int bounceFrames;
    private bool groundBounced;

    public HurtAerialState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.HurtAerial;

    public override void Enter()
    {
        ResetTick();
        hitstun = owner.PendingHitstun;
        wasGroundedOnHit = false;
        bounceFrames = owner.PendingGroundBounce ? PhysicsConstants.GROUND_BOUNCE_FRAMES : 0;
        groundBounced = false;
    }

    public override void Step()
    {
        base.Step();
        hitstun--;

        // Half gravity while airborne in hitstun, light air friction, capped fall speed, and NO
        // speed limit (the unlimited force pass). This is what lets a launcher keep its full
        // knockback instead of being clipped to MAX_AIR_SPEED.
        physics.ApplyForces(PhysicsConstants.HURT_AIR_FRICTION_RATIO, PhysicsConstants.HURT_AIR_GRAVITY,
            false, limitSpeed: false);

        HandleWallBounce();
        HandleLanding();

        // A knockdown extends hitstun until the victim has actually landed, so a launcher cannot
        // be walked out of before touching the floor.
        bool extendedHitstun = owner.PendingKnockdownExtendsHitstun
            && owner.PendingKnockdown && !groundBounced;

        if (!extendedHitstun && hitstun <= 0 && bounceFrames <= 0 && !owner.PendingKnockdown)
            owner.LeaveHurtState();
    }

    /// <summary>
    /// Rebounds airborne knockback off the stage wall instead of letting it die there.
    /// Skipped while in hitlag or during a ground bounce.
    /// </summary>
    private void HandleWallBounce()
    {
        if (owner.HitlagRemaining > 0 || bounceFrames > 0) return;
        if (!physics.BlockedByWall) return;

        // Use the velocity from the moment of impact: collision resolution has already zeroed
        // the current velocity, which would otherwise make the rebound a no-op.
        float impactX = physics.BlockedVelocity.x;
        if (Mathf.Abs(impactX) < 0.001f) return;

        physics.setVelocity(impactX * PhysicsConstants.WALL_BOUNCE_FACTOR, physics.getVelocity().y);

        if (owner.PendingWallSlam && owner.WallSlams < PhysicsConstants.MAX_WALL_SLAMS)
        {
            owner.RegisterWallSlam();
            owner.SwitchTo(CombatState.WallSlam);
            return;
        }

        // -0.85 rebound plus 3 ticks of hitlag.
        owner.AddHitlag(PhysicsConstants.WALL_BOUNCE_HITLAG);
    }

    private void HandleLanding()
    {
        // Hold the victim pinned to the floor for the remaining bounce frames.
        if (bounceFrames > 0)
        {
            bounceFrames--;
            return;
        }

        if (!physics.IsGrounded) return;
        var vel = physics.getVelocity();

        // Only air-ground-bounce while actually falling (vel.y >= 0), and only once.
        if (owner.PendingAirGroundBounce && !groundBounced && vel.y >= 0f)
        {
            groundBounced = true;
            physics.setVelocity(vel.x, -vel.y * PhysicsConstants.WALL_BOUNCE_FACTOR);
            bounceFrames = PhysicsConstants.GROUND_BOUNCE_FRAMES;
            return;
        }

        if (Tick <= Mathf.Max(0, owner.PendingMinimumGroundedFrames)) return;

        // Knockdown or death goes to Knockdown / HardKnockdown. Dying always knocks down
        // regardless of the hitbox's own knockdown flag.
        if (owner.PendingKnockdown || owner.IsDead())
        {
            owner.SwitchTo(owner.PendingHardKnockdown ? CombatState.HardKnockdown : CombatState.Knockdown);
            return;
        }

        // A non-knockdown launch ends in Landing, and a dead fighter still ends in Knockdown.
        if (owner.IsGrounded) owner.LeaveHurtState();
    }
}

/// <summary>
/// The fighter is pinned to the floor with 0.125 friction and cannot act until the state is
/// explicitly released.
/// </summary>
public class KnockdownState : HurtStateBase
{
    public KnockdownState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.Knockdown;

    public override void Enter()
    {
        ResetTick();
        hitstun = int.MaxValue;
        // Zero vertical velocity and force grounded on knockdown.
        physics.setVelocity(physics.getVelocity().x, 0f);
        physics.SnapToGround();
        // Being knocked down ends this fighter's action for the turn. Nothing else in this
        // state calls CompleteMove, so without this a knockdown stretched the turn out to the
        // frame cap instead of resolving like every other action.
        owner.ReportActionComplete();
    }

    public override void Step()
    {
        base.Step();
        // 0.125 skid friction, no speed limit (the unlimited force pass).
        physics.ApplyForces(PhysicsConstants.KNOCKDOWN_FRICTION_RATIO, PhysicsConstants.GRAVITY,
            physics.IsGrounded, limitSpeed: false);
        physics.setVelocity(physics.getVelocity().x, 0f);
    }

    /// <summary>Knockdown only ends when the turn ends or the fighter dies.</summary>
    public void Release()
    {
        owner.LeaveHurtState();
    }
}

/// <summary>
/// Identical physics to Knockdown, but tracked as a separate state and treated as off-the-ground
/// for combo purposes. Previously the landing handler returned Knockdown from both ternary
/// branches, so hardKnockdown was authored data that did nothing.
/// </summary>
public class HardKnockdownState : KnockdownState
{
    public HardKnockdownState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.HardKnockdown;
}

/// <summary>
/// Pins the fighter to the wall with escalating gravity and fall speed per slam, for a duration
/// that shortens with each additional slam, then drops into Knockdown.
/// </summary>
public class WallSlamState : HurtStateBase
{
    public WallSlamState(CharacterController owner) : base(owner) { }

    public override CombatState Id => CombatState.WallSlam;

    public override void Enter()
    {
        ResetTick();
        hitstun = int.MaxValue;
        // Body collision is disabled explicitly here: the fighter is pinned flat against the
        // wall and must not be shoved off it by the opponent.
        owner.SetCollidingWithOpponent(false);
        // Zero velocity on entry so the fighter hangs on the wall.
        physics.setVelocity(0f, 0f);
        physics.SnapToWall();
    }

    public override void Step()
    {
        base.Step();

        int slams = Mathf.Max(1, owner.WallSlams);
        float grav = PhysicsConstants.WALL_SLAM_GRAVITY + PhysicsConstants.WALL_SLAM_GRAVITY_PER_SLAM * (slams - 1);
        float fall = PhysicsConstants.WALL_SLAM_FALL_SPEED + PhysicsConstants.WALL_SLAM_FALL_PER_SLAM * (slams - 1);

        physics.ApplyCustomForces(PhysicsConstants.AIR_FRICTION_RATIO, grav, false, fall);
        physics.SnapToWall();

        // The duration shortens by 8 per slam, and reaching it while still airborne drops the
        // fighter into Fall rather than holding them on the wall. Ours used to fall through to
        // LeaveHurtState, which skipped the fall entirely.
        int duration = PhysicsConstants.WALL_SLAM_MIN_DURATION - PhysicsConstants.WALL_SLAM_DURATION_PER_SLAM * (slams - 1);
        if (Tick <= duration) return;

        if (physics.IsGrounded) owner.SwitchTo(CombatState.Knockdown);
        else owner.SwitchTo(CombatState.Fall);
    }
}