using UnityEngine;

/// <summary>
/// Resolves one hit into a knockback vector and a state transition.
///
/// Picks the victim's hurt state from the hitbox (not from the victim's position) and always
/// performs a state change. That state change is load-bearing: because hurt states own no move
/// force, it is what stops the victim's current move from continuing to push them around.
/// </summary>
public static class HitReaction
{
    public static bool Resolve(CharacterController victim, CharacterController attacker, HitboxData data, float facing)
    {
        if (victim == null || attacker == null) return false;
        // Allies can never connect, no matter who calls this.
        if (!CombatTeamUtility.AreEnemies(victim.Team, attacker.Team)) return false;

        bool grounded = victim.IsGrounded;
        // Air guarding is real: it just takes extra pushback via AIR_BLOCK_PUSHBACK_MODIFIER.
        // Gating this on grounded made an air block behave like a clean hit, dealing full damage
        // and full knockback.
        bool blocked = victim.IsBlocking;

        // knockback is stored as a magnitude plus a direction already baked together, so the
        // magnitude is its length and the direction is its normalised form, mirrored by facing.
        float magnitude = data.knockback.magnitude;
        Vector2 direction = magnitude > 0.0001f ? data.knockback / magnitude : Vector2.zero;

        // Repeating the same move inside one combo scales its knockback up, 1.25x per repeat
        // grounded and 1.05x airborne, capped at MAX_KNOCKBACK.
        int sameMoveUses = attacker.GetMoveUseCount(attacker.LastMoveUsed);
        if (sameMoveUses > 0)
            magnitude = PhysicsConstants.SameMoveKnockback(magnitude, sameMoveUses, grounded);

        // A purely grounded hit zeroes the vertical component before anything else looks at it,
        // but ONLY for that hit state. A hit marked launchGrounded keeps its vertical knockback
        // and lifts the victim instead.
        if (grounded && !data.launchGrounded)
        {
            direction.y = 0f;
            if (direction.sqrMagnitude > 0.0001f) direction.Normalize();
        }

        Vector2 knockback = direction * magnitude * new Vector2(facing, 1f);

        // Hitstun drops per repeat of a move already used in this combo, floored at half.
        int hitstun = data.hitstunTicks > 0 ? data.hitstunTicks : PhysicsConstants.HITSTUN_FRAMES;
        if (sameMoveUses > 0)
            hitstun = PhysicsConstants.SameMoveHitstun(hitstun, sameMoveUses);

        bool knockdown = data.knockdown;
        bool hardKnockdown = data.hardKnockdown;

        // After MAX_GROUNDED_HITS grounded hits, the next one launches, which bounds combo length.
        victim.RegisterGroundedHit(grounded && !data.launchGrounded);
        bool forcedLaunch = !grounded || victim.GroundedHitsTaken >= PhysicsConstants.MAX_GROUNDED_HITS;
        // The counter resets when the limit triggers the launch, so the next one needs a full run.
        if (forcedLaunch && grounded) victim.ConsumeGroundedHits();

        if (blocked)
        {
            ResolveBlocked(victim, attacker, data, knockback);
            return false;
        }

        // Grounded victims are launched purely horizontally. The launchGrounded case is the one
        // that keeps a vertical component, and the direction above already handled that.
        if (grounded && !data.launchGrounded && !forcedLaunch)
            knockback.y = 0f;

        victim.IncrementCombo(data.scaleCombo && data.incrementCombo, data.ComboScalingAmount);
        victim.ApplyDamage(ResolveDamage(attacker, data));

        // DI strength differs between grounded (3.5) and aerial (2.0) hurt.
        float diStrength = grounded ? PhysicsConstants.DI_STRENGTH_GROUNDED : PhysicsConstants.DI_STRENGTH_AERIAL;
        knockback = ApplyDI(victim, attacker, knockback, diStrength, data.DiModifier);

        // A victim already pressed against the stage wall has its horizontal knockback dropped
        // outright, so it does not jitter in the corner.
        if (grounded && victim.TouchingWall && !data.wallSlam)
            knockback.x = 0f;

        // The victim is nudged up on the frame they are launched, so a launcher that starts from
        // the ground does not immediately re-collide with the floor.
        bool launched = !grounded || data.launchGrounded || forcedLaunch;
        if (launched) victim.ApplyLaunchLift();

        victim.ApplyKnockback(knockback);

        // Record what the entered state needs to know.
        victim.Pending = new CharacterController.PendingHit
        {
            hitstun = hitstun,
            knockdown = knockdown,
            hardKnockdown = hardKnockdown,
            wallSlam = data.wallSlam,
            airGroundBounce = data.airGroundBounce,
            groundBounce = data.groundBounce,
            launched = launched,
            minimumGroundedFrames = data.minimumGroundedFrames,
            knockdownExtendsHitstun = data.knockdownExtendsHitstun,
        };

        // Attacker and victim freeze for their own separate durations, which is what produces a
        // move's plus frames on hit.
        attacker.AddHitlag(data.HitlagFrames);
        victim.AddHitlag(data.VictimHitlagFrames);

        // A hit carrying disable_collision (the default) clears body collision on the victim, so
        // the two pass through each other during hitstun and the mutual pushback alone separates
        // them.
        //
        // Without this, two fighters trading hits end up interpenetrating and stuck: AABB.sweep
        // returns a negative distance when the moving centre is already inside the target's
        // Minkowski-expanded box, and PhysicsManager clamps that to zero movement regardless of
        // travel direction, so neither can back out. Throws and grabs opt out via
        // keepCollisionOnHit.
        victim.SetCollidingWithOpponent(!data.DisableCollision);

        attacker.RegisterMoveUse(attacker.LastMoveUsed);

        // BOTH fighters are pushed apart on every connect, scaled by the attacker's current combo
        // length, and along the attacker/victim axis rather than facing.
        float pushback = PhysicsConstants.HitPushback(data.PushbackX, attacker.HitstunDecayComboCount);
        attacker.AddPushback(pushback);
        victim.AddPushback(pushback);

        // The state change: this is what stops the victim's move from applying force.
        CombatState next = ResolveHurtState(victim, grounded, forcedLaunch, data);
        victim.SwitchTo(next);
        return true;
    }

    /// <summary>
    /// Damage scales off quadratically with the attacker's combo count, floored so a long combo
    /// still chips for at least 1.
    /// </summary>
    private static float ResolveDamage(CharacterController attacker, HitboxData data)
    {
        if (data.damage <= 0) return 0f;

        int count = attacker.ComboCount - 1 + (data.damageProration > 0 ? data.damageProration : 0);
        float scaled = data.damage * PhysicsConstants.ComboStaleModifier(Mathf.Max(count, 0));
        return Mathf.Max(scaled, 1f);
    }

    private static CombatState ResolveHurtState(CharacterController victim, bool grounded,
        bool forcedLaunch, HitboxData data)
    {
        // The state comes from the hitbox, not from the victim's position. A hit marked
        // launchGrounded launches a grounded victim, which is what makes vertical attacks work.
        bool aerial = !grounded || data.launchGrounded || forcedLaunch;
        if (!aerial) return CombatState.HurtGrounded;

        // Knockdown is deliberately NOT entered here. It is only reachable from HurtAerial's
        // landing handler, so a knockdown hit still throws the victim up before they hit the
        // floor, rather than teleporting them there on the hit frame.
        return CombatState.HurtAerial;
    }

    /// <summary>
    /// Guarded hits divide the hit's own knockback by a divisor rather than scaling it by a flat
    /// fraction, and push the attacker back by half the blocker's pushback. An earlier version
    /// used a hand-picked 0.25 scale, which gave the blocker roughly a third of the correct
    /// pushback.
    /// </summary>
    private static void ResolveBlocked(CharacterController victim, CharacterController attacker, HitboxData data, Vector2 knockback)
    {
        // An active guard uses the tighter grounded divisor, so it gets noticeably more pushback.
        float divisor = victim.IsGrounded
            ? PhysicsConstants.BLOCK_GROUNDED_KNOCKBACK_DIVISOR
            : PhysicsConstants.BLOCK_KNOCKBACK_DIVISOR;

        float modifier = data.BlockPushback;

        // Which way the hit was travelling, i.e. which way the blocker gets pushed.
        float pushDir = Mathf.Sign(knockback.x);
        if (Mathf.Abs(pushDir) < 0.5f) pushDir = 1f;

        // blockPushbackReversible flips the pushback when the blocker is on the far side.
        if (data.blockPushbackReversible)
        {
            float facing = attacker.FacingVector.x;
            if ((pushDir > 0f && facing > 0f) || (pushDir < 0f && facing < 0f))
                modifier *= -data.BlockReversePushback;
        }

        // Dividing keeps the direction the hit would have sent the blocker. Taking an absolute
        // value threw that away and shoved them back toward the attacker instead of away.
        float hitKnockback = data.knockback.magnitude;
        float pushback = hitKnockback / divisor * modifier;

        // Air guarding absorbs most of the pushback.
        if (!victim.IsGrounded) pushback *= PhysicsConstants.AIR_BLOCK_PUSHBACK_MODIFIER;

        // Chip damage through the guard: damage / 3, times the hitbox's own chip modifier. A
        // blocking fighter stays on their feet rather than entering a hurt state, so the
        // knockback here is horizontal only.
        victim.ApplyDamage(data.damage * PhysicsConstants.BLOCK_CHIP_MODIFIER * data.ChipDamage);
        victim.ApplyKnockback(new Vector2(pushback * pushDir, 0f));

        // The attacker is shoved back by half the blocker's pushback, in the opposite direction.
        // Applied immediately rather than queued as pushback.
        attacker.ApplyVelocity(attacker.FacingVector * -(pushback * 0.5f));

        attacker.AddHitlag(data.HitlagFrames);
        victim.AddHitlag(data.VictimHitlagFrames);

        // Blockstun is the hitbox's plus_frames, plus its hitlag and a fixed bonus, plus extra
        // frames when guarding an airborne attacker.
        int blockstun = data.plusFrames + data.HitlagFrames + PhysicsConstants.BLOCK_HITLAG_BONUS;
        if (!victim.IsGrounded) blockstun += PhysicsConstants.VS_AERIAL_PLUS_FRAMES;

        victim.Pending = new CharacterController.PendingHit
        {
            hitstun = blockstun,
            knockdown = false,
            hardKnockdown = false,
            wallSlam = false,
            airGroundBounce = false,
            groundBounce = false,
            launched = false,
            minimumGroundedFrames = 0,
            knockdownExtendsHitstun = false,
        };
        // A blocking fighter stays on their feet rather than entering a hurt state.
        victim.AddHitstun(blockstun);
    }

    /// <summary>
    /// Applies the victim's knockback indirection: a force along the DI direction, scaled by the
    /// hurt state's DI strength and the hitbox's di_modifier.
    ///
    /// Magnitude comes from <see cref="PhysicsConstants.DiScaling"/>, which ramps from 1.0 to 6.0
    /// as the attacker's combo count approaches the limit, so a full-combo DI can add up to
    /// 6.0 * 3.5 = 21 pixels of force. An earlier version used the player's 0..1 stick power
    /// times a flat 0.98, which capped DI at 3.4 and made it almost irrelevant.
    ///
    /// A grounded victim only takes horizontal DI.
    /// </summary>
    private static Vector2 ApplyDI(CharacterController victim, CharacterController attacker,
        Vector2 knockback, float strength, float diModifier)
    {
        if (knockback == Vector2.zero) return knockback;

        float scaling = PhysicsConstants.DiScaling(attacker.ComboCount);
        var di = victim.KnockbackIndirectionVector;
        float power = scaling * victim.KnockbackIndirectionPower * strength * diModifier;
        Vector2 diForce = di * power;
        if (victim.IsGrounded) diForce.y = 0f;

        return knockback + diForce;
    }
}