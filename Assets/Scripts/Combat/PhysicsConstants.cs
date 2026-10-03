using UnityEngine;

/// <summary>
/// Central tuning table for the combat physics model.
///
/// The tuning unit is the pixel. A character is 0.7 Unity units tall, which is roughly 35
/// pixels, so one Unity unit is about 50 pixels and one pixel is <see cref="PX"/> Unity units.
/// Every distance and force below is written in pixels and multiplied by PX, so the numbers stay
/// readable and comparable against each other rather than being arbitrary floats.
///
/// Values are expressed as ratios wherever possible (friction as a fraction of current speed,
/// modifiers as multipliers) because ratios survive changes to the overall scale.
/// </summary>
public static class PhysicsConstants
{
    /// <summary>Size of one tuning pixel expressed in Unity units.</summary>
    public const float PX = 0.02f;

    // --- Movement attributes ---
    /// <summary>Downward acceleration applied every frame.</summary>
    public const float GRAVITY = 0.5f * PX;

    /// <summary>Grounded friction, expressed as a flat per-frame amount in pixels.</summary>
    public const float GROUND_FRICTION_PIXELS = 2.5f;

    /// <summary>Airborne friction, as a flat per-frame amount in pixels.</summary>
    public const float AIR_FRICTION_PIXELS = 0.2f;

    /// <summary>
    /// Ground friction is applied as a fraction of the current horizontal speed rather than as
    /// a flat amount. A flat 2.5px subtraction would only work if each move re-applied its force
    /// every frame so the two cancelled; move impulses are applied once here, so a ratio keeps
    /// the same feel. 2.5px against a 15px reference ground speed is a ~17% loss per frame.
    /// </summary>
    public const float GROUND_FRICTION_RATIO = GROUND_FRICTION_PIXELS / 15f;

    /// <summary>
    /// Air friction is 12x lighter than ground friction (0.2px vs 2.5px). This is the single
    /// most important movement trait: jump and dash momentum is preserved in the air and only
    /// bites once you land. Applying the same 0.8 decay everywhere destroys that distinction.
    /// </summary>
    public const float AIR_FRICTION_RATIO = AIR_FRICTION_PIXELS / 15f;

    /// <summary>
    /// Horizontal speed cap while grounded, only ever applied by the *limited* force pass.
    ///
    /// DELIBERATE DEVIATION: the reference model uses 15px. Ours is 25px because every authored
    /// move impulse (dash 1.0, slash 0.55, walk 0.015) is expressed in Unity units against this
    /// cap, so dropping it to 15px means rescaling all of them and re-tuning every move's
    /// distance. This constant no longer damages hitstun: see
    /// <see cref="CharacterPhysics.ApplyForces"/>, which skips the caps entirely inside hurt
    /// states, matching the reference model's split between limited and unlimited force passes.
    /// </summary>
    public const float MAX_GROUND_SPEED = 25f * PX;

    /// <summary>
    /// Cap on horizontal speed while airborne under the limited force pass (reference value
    /// 10px). Like <see cref="MAX_GROUND_SPEED"/> this is only reached outside hurt states.
    /// </summary>
    public const float MAX_AIR_SPEED = 12f * PX;

    /// <summary>
    /// Maximum downward speed (15px). Applied by the gravity pass, not the speed-limit pass, so
    /// it stays in force during knockback.
    /// </summary>
    public const float MAX_FALL_SPEED = 15f * PX;

    /// <summary>Speed below which a grounded character is considered settled.</summary>
    public const float GROUND_SETTLE_SPEED = 0.5f * PX;

    // --- Collision response ---
    /// <summary>
    /// Zero: characters are not bounced off walls, the velocity on the blocked axis is simply
    /// removed so knockback into a corner pins the character against it.
    /// </summary>
    public const float WALL_RESTITUTION = 0f;

    /// <summary>Extra skin width kept between a character and static geometry.</summary>
    public const float COLLIDER_SKIN = 0.001f;

    // --- Hurt states ---
    /// <summary>Grounded skid friction during hitstun.</summary>
    public const float HURT_GROUND_FRICTION_RATIO = 0.05f;

    /// <summary>Airborne drag during hitstun.</summary>
    public const float HURT_AIR_FRICTION_RATIO = 0.015f;

    /// <summary>
    /// Exactly half of the normal gravity (0.25 vs 0.5). This is what makes launchers hang in
    /// the air instead of dropping the victim immediately.
    /// </summary>
    public const float HURT_AIR_GRAVITY = 0.25f * PX;

    /// <summary>Fall-speed cap while airborne in hitstun (15px).</summary>
    public const float HURT_AIR_FALL_SPEED = 15f * PX;

    /// <summary>Skid friction while knocked down and pinned to the floor.</summary>
    public const float KNOCKDOWN_FRICTION_RATIO = 0.125f;

    // --- Wall slam ---
    /// <summary>Gravity while pinned to a wall.</summary>
    public const float WALL_SLAM_GRAVITY = 0.25f * PX;

    /// <summary>Fall-speed cap while pinned to a wall (5px, deliberately slow).</summary>
    public const float WALL_SLAM_FALL_SPEED = 5f * PX;

    /// <summary>Extra gravity per additional slam.</summary>
    public const float WALL_SLAM_GRAVITY_PER_SLAM = 0.2f * PX;

    /// <summary>Extra fall-speed cap per additional slam.</summary>
    public const float WALL_SLAM_FALL_PER_SLAM = 0.5f * PX;

    /// <summary>Base frames pinned to the wall, shortened by 8 for each extra slam.</summary>
    public const int WALL_SLAM_MIN_DURATION = 20;
    public const int WALL_SLAM_DURATION_PER_SLAM = 8;

    /// <summary>Rebound factor when airborne knockback strikes a wall.</summary>
    public const float WALL_BOUNCE_FACTOR = -0.85f;

    /// <summary>Extra hitlag ticks granted when bouncing off a wall.</summary>
    public const int WALL_BOUNCE_HITLAG = 3;

    /// <summary>Frames a victim stays pinned to the floor after a ground bounce.</summary>
    public const int GROUND_BOUNCE_FRAMES = 4;

    /// <summary>
    /// The victim is nudged up this far on the frame they are launched, so a launcher that
    /// starts from the ground does not immediately re-collide with the floor.
    /// </summary>
    public const float LAUNCH_LIFT = 1f * PX;

    /// <summary>How many times a combo can be slammed into a wall before it stops.</summary>
    public const int MAX_WALL_SLAMS = 3;

    /// <summary>
    /// A hit turns body collision off (see HitboxData.DisableCollision), but the two fighters
    /// stay body-blocked for this many ticks so they do not visibly pop apart on the exact frame
    /// of the hit.
    /// </summary>
    public const int HITLAG_COLLISION_TICKS = 4;

    /// <summary>
    /// Grounded hits a fighter can absorb before the next one launches them, forcing a combo to
    /// end. The counter resets once it triggers, so the next launch needs another full run.
    /// </summary>
    public const int MAX_GROUNDED_HITS = 7;

    /// <summary>
    /// Health fraction dealt as chip on each wall slam. This is an original addition: the
    /// reference model declares a 0.75 constant for this but never reads it, so its wall slams
    /// deal no damage of their own.
    /// </summary>
    public const float WALL_SLAM_DAMAGE_FRACTION = 0.075f;

    // --- Knockback indirection ---
    /// <summary>
    /// DI force multiplier while grounded.
    ///
    /// The PX factor is essential and was previously missing. This value is used as a raw
    /// velocity in HitReaction (it is added straight to the victim's knockback velocity), so it
    /// has to be in Unity units. Without the conversion it was 50x too large, which made the
    /// victim's DI roughly 8x the size of the hit's own knockback and completely overrode it:
    /// that is why no amount of tuning the hitbox knockback produced visible knockback. The
    /// intended ratio is 3.5px of DI against 10px of knockback, i.e. 35%.
    /// </summary>
    public const float DI_STRENGTH_GROUNDED = 3.5f * PX;

    /// <summary>DI force multiplier while airborne. See <see cref="DI_STRENGTH_GROUNDED"/>.</summary>
    public const float DI_STRENGTH_AERIAL = 2.0f * PX;

    /// <summary>DI magnitude with no combo on the attacker.</summary>
    public const float DI_MIN_SCALING = 1f;

    /// <summary>
    /// DI magnitude at full combo scaling. DI is worth far more than the victim's raw knockback:
    /// a full-combo DI can add up to 6.0 * 3.5 = 21 pixels of force, which is what stops long
    /// combos from being inescapable.
    /// </summary>
    public const float DI_MAX_SCALING = 6f;

    /// <summary>Combo count over which DI scales from min to max.</summary>
    public const int DI_COMBO_LIMIT = 15;

    /// <summary>
    /// DI magnitude ramps from <see cref="DI_MIN_SCALING"/> to <see cref="DI_MAX_SCALING"/> as
    /// the attacker's combo count approaches <see cref="DI_COMBO_LIMIT"/>. Scaling with combo
    /// length is the point: DI should be a real out for a fighter who is losing a long combo,
    /// and irrelevant on the first hit.
    /// </summary>
    public static float DiScaling(int attackerComboCount)
    {
        float ratio = Mathf.Clamp01(attackerComboCount / (float)DI_COMBO_LIMIT);
        return Mathf.Lerp(DI_MIN_SCALING, DI_MAX_SCALING, ratio);
    }

    // --- Pushback ---
    /// <summary>How fast pushback grows per hit already landed in the combo.</summary>
    public const float COMBO_PUSHBACK_COEFFICIENT = 0.4f;

    /// <summary>Base share of the mutual pushback a hit contributes.</summary>
    public const float DEFAULT_PUSHBACK_X = 1f;

    /// <summary>
    /// Pushback magnitude applied to the attacker, the victim and the victim's opponent on every
    /// connect:
    /// <c>(pushback_x + hitstun_decay_combo_count * 0.4) / 2</c>.
    ///
    /// Returned as a positive magnitude in UNITY units (hence the PX factor, previously missing,
    /// which made it 25x too large and shoved both fighters apart on every single hit); the
    /// caller resolves the direction, because pushback runs along the attacker/victim axis rather
    /// than along facing.
    /// </summary>
    public static float HitPushback(float pushbackX, int hitstunDecayComboCount)
    {
        return (pushbackX + hitstunDecayComboCount * COMBO_PUSHBACK_COEFFICIENT) * 0.5f * PX;
    }

    // --- Hitstun, hitlag and frame advantage ---
    /// <summary>Default hitstun for a hit that does not specify its own.</summary>
    public const int HITSTUN_FRAMES = 30;

    /// <summary>Default hitlag applied to the attacker.</summary>
    public const int HITLAG_FRAMES = 4;

    /// <summary>Default hitlag applied to the victim. Normally matches the attacker's.</summary>
    public const int VICTIM_HITLAG_FRAMES = 4;

    /// <summary>
    /// Extra blockstun granted on top of a blocked hit's hitlag. Blockstun is derived from
    /// hitlag rather than from hitstun, so a slow move does not automatically give its blocker
    /// more time to recover.
    /// </summary>
    public const int BLOCK_HITLAG_BONUS = 1;

    /// <summary>Extra blockstun when guarding an airborne attacker.</summary>
    public const int VS_AERIAL_PLUS_FRAMES = 2;

    /// <summary>Extra blockstun when guarding on the wrong height.</summary>
    public const int WRONG_HIT_HEIGHT_PLUS_FRAMES = 2;

    /// <summary>Chip damage through a guard, as a fraction of the hit's damage (damage / 3).</summary>
    public const float BLOCK_CHIP_MODIFIER = 0.33f;

    /// <summary>
    /// A blocked hit pushes the blocker back by the hit's own knockback DIVIDED by this, not
    /// scaled by a flat fraction of it. Dividing is what makes a light hit almost harmless on
    /// block while a heavy one still shoves, which is the correct relationship.
    /// </summary>
    public const float BLOCK_KNOCKBACK_DIVISOR = 3f;

    /// <summary>
    /// Tighter divisor used while actively guarding. A grounded guard uses this instead of
    /// <see cref="BLOCK_KNOCKBACK_DIVISOR"/>, so it gets noticeably more pushback.
    /// </summary>
    public const float BLOCK_GROUNDED_KNOCKBACK_DIVISOR = 1.5f;

    /// <summary>
    /// Guarding in the air absorbs most of the pushback. Only applies to an airborne blocker.
    /// </summary>
    public const float AIR_BLOCK_PUSHBACK_MODIFIER = 0.35f;

    /// <summary>
    /// Caps the *scalar* knockback a single hitbox may contribute, applied before DI is added.
    /// It is deliberately not a cap on the resulting velocity, because DI is added afterwards and
    /// is meant to be able to exceed it.
    /// </summary>
    public const float MAX_KNOCKBACK = 30f * PX;

    // --- Combo scaling ---
    /// <summary>Knockback multiplier per repeat of the same move while grounded.</summary>
    public const float SAME_MOVE_KNOCKBACK_GROUNDED = 1.25f;

    /// <summary>Knockback multiplier per repeat of the same move while airborne.</summary>
    public const float SAME_MOVE_KNOCKBACK_AERIAL = 1.05f;

    /// <summary>Hitstun lost per repeat of the same move.</summary>
    public const int SAME_MOVE_HITSTUN_DECREASE = 1;

    /// <summary>Combo length over which damage scaling reaches its floor.</summary>
    public const int MAX_STALES = 15;

    /// <summary>Damage multiplier floor at <see cref="MAX_STALES"/> combo hits.</summary>
    public const float MIN_STALE_MODIFIER = 0.2f;

    /// <summary>
    /// Repeating the same move inside one combo scales its knockback up by
    /// <see cref="SAME_MOVE_KNOCKBACK_GROUNDED"/>^n grounded
    /// (<see cref="SAME_MOVE_KNOCKBACK_AERIAL"/>^n airborne), capped at
    /// <see cref="MAX_KNOCKBACK"/>. This is what stops a combo of one repeated button from
    /// pinning someone in place forever: the hits drift further apart each time.
    /// </summary>
    public static float SameMoveKnockback(float knockback, int timesUsed, bool grounded)
    {
        float modifier = Mathf.Pow(grounded ? SAME_MOVE_KNOCKBACK_GROUNDED : SAME_MOVE_KNOCKBACK_AERIAL, timesUsed);
        return Mathf.Min(knockback * modifier, MAX_KNOCKBACK);
    }

    /// <summary>
    /// A move repeated within one combo loses hitstun per repeat, but never drops below half of
    /// what it started with, so a long string still connects.
    /// </summary>
    public static int SameMoveHitstun(int hitstun, int timesUsed)
    {
        return Mathf.Max(hitstun - SAME_MOVE_HITSTUN_DECREASE * (timesUsed + 1), hitstun / 2);
    }

    /// <summary>
    /// Quadratic damage scale-off over the length of a combo: 1.0 on the first hit, easing down
    /// to <see cref="MIN_STALE_MODIFIER"/> at <see cref="MAX_STALES"/> hits.
    /// </summary>
    public static float ComboStaleModifier(int count)
    {
        float ratio = (MAX_STALES - Mathf.Min(count, MAX_STALES)) / (float)MAX_STALES;
        return (1f - MIN_STALE_MODIFIER) * ratio * ratio + MIN_STALE_MODIFIER;
    }

    // --- Landing ---
    /// <summary>Base landing recovery frames.</summary>
    public const int LANDING_LAG = 4;

    /// <summary>Extra landing recovery available for a hard fall.</summary>
    public const int MAX_EXTRA_LANDING_LAG = 5;

    // --- Jump ---
    /// <summary>
    /// Default jump impulse in pixels, before <see cref="BASE_JUMP_SPEED"/> is added. Authored
    /// per jump state, because it genuinely varies: 8 for the old base character, 11 for the
    /// mutant, 8.5 for SwordGuy's jump and 7 for its double jump.
    ///
    /// This was previously 25, which is only the jump state's class-level default export and which
    /// no character ever uses. At 25 the impulse became 25.5px and a full straight-up jump reached
    /// ~1056px against a 400px ceiling, so every full jump slammed the ceiling. The real 8.5 gives
    /// ~132px, which is the only reason a 400px ceiling exists at all.
    /// </summary>
    public const float DEFAULT_JUMP_SPEED = 8.5f;

    /// <summary>Constant added to every jump's speed regardless of how it was authored.</summary>
    public const float BASE_JUMP_SPEED = 0.5f;

    /// <summary>Vertical exaggeration applied to the jump direction.</summary>
    public const float JUMP_Y_MODIFIER = 1.5f;

    /// <summary>Global vertical scaling, applied when the jump is not a combo escape.</summary>
    public const float GLOBAL_JUMP_MODIFIER = 0.85f;

    /// <summary>
    /// Whether this jump counts as a combo escape, which is the only case where the global
    /// reduction is skipped: the fighter must be mid-combo AND the opponent must be off the floor
    /// with them. A neutral jump is always reduced, and so is a jump during a combo where the
    /// opponent has already come down to the floor.
    /// </summary>
    public static bool IsComboEscape(int comboCount, bool opponentGrounded)
    {
        return comboCount > 0 && !opponentGrounded;
    }

    /// <summary>
    /// Fraction of horizontal speed kept when jumping. A jump throws away 75% of the speed
    /// carried into it, so hopping forward out of a dash travels far less than the dash did.
    /// </summary>
    public const float JUMP_X_SPEED_PRESERVED = 0.25f;

    /// <summary>Super jump impulse (17px, against a ~25.5px base jump).</summary>
    public const float SUPER_JUMP_FORCE = 17f * PX;

    /// <summary>
    /// Jump power curve. The input vector is first scaled by its own length squared and then
    /// averaged with itself: <c>force * (r^2 + 1) / 2</c>, i.e. a magnitude of
    /// <c>(r^3 + r) / 2</c> for an input of length r.
    ///
    /// Cubic, so control over height is genuinely useful: a half-power hop is far shorter than a
    /// full one, and most of the wheel's travel covers the low end of the range. This was
    /// previously implemented as the quadratic <c>(t^2 + t) / 2</c>, which agreed at full power
    /// but was up to 20% stronger through the middle of the range.
    /// </summary>
    public static float JumpCurve(float r)
    {
        r = Mathf.Clamp01(r);
        return r * (r * r + 1f) * 0.5f;
    }

    // --- Dash ---
    /// <summary>Dash friction over the dash's own frames, much lighter than standing friction.</summary>
    public const float DASH_FRICTION_RATIO = 0.05f;

    /// <summary>Cap on the whole dash velocity vector (40px).</summary>
    public const float DASH_SPEED_LIMIT = 40f * PX;

    /// <summary>Converts a pixel value to Unity units.</summary>
    public static float FromPixels(float pixels)
    {
        return pixels * PX;
    }
}