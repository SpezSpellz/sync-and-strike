using System;
using UnityEngine;

/// <summary>
/// The observable state captured at the moment an AI committed to a move, plus what
/// actually happened as a result. This is the feature/label pair that gets written to
/// the training log when the player votes on a companion move.
/// </summary>
[Serializable]
public struct AIDecisionContext
{
    // --- Features (state at decision time) ---
    public float selfX, selfY, selfVelX, selfVelY;
    public float selfGrounded;
    public float selfHealthFrac;
    public float selfHitstun;
    public float selfBusy;
    public float selfFlipped;

    // Target and ally positions are stored RELATIVE TO SELF.
    //
    // Absolute world X is meaningless to a fighter and actively harmful as a feature: the stage is
    // symmetric, so "I am at x = +11, near the right wall" and "I am at x = -11, near the left wall"
    // are the same tactical situation mirrored, but as raw numbers they are opposite inputs. A policy
    // fed absolute positions has to learn that symmetry from scratch and will happily behave
    // differently on the two sides of the screen. Relative coordinates are already symmetric, which
    // is why dx/dy existed separately and are now the primary representation.
    public float targetX, targetY, targetVelX, targetVelY;
    public float targetGrounded;
    public float targetHealthFrac;
    public float targetHitstun;
    public float targetBusy;
    public float targetBlocking;

    public float allyX, allyY;
    public bool hasAlly;

    /// <summary>Stage half-width, used to express absolute self position as a bounded fraction.</summary>
    public static float StageHalfWidth = 11.76f;

    /// <summary>
    /// Y offset of the arena floor, so self Y can be expressed as height above the floor rather than
    /// as a raw world coordinate.
    /// </summary>
    public static float FloorY = -1.19f;

    /// <summary>
    /// Ceiling height above the floor, the natural scale for vertical position. Jumps top out well
    /// below it, so this normalises height to roughly 0..1 without any clamping.
    /// </summary>
    public static float StageHeight = 8.6f;

    // Relative geometry, the most important signals for a fighter AI.
    public float dx, dy, distance;

    // --- Action ---
    public string moveId;
    public bool flipped;
    public float jumpPower, jumpAngle;
    public float diPower, diAngle;

    // --- Outcomes (filled in after the turn resolves) ---
    public float damageDealtToTarget;
    public float damageTakenBySelf;
    public bool hitLanded;
    public bool moveWhiffed;

    public static AIDecisionContext Capture(
        CharacterController self,
        CharacterController target,
        CharacterController ally,
        AIDecision decision)
    {
        var ctx = new AIDecisionContext
        {
            moveId = decision.moveId,
            flipped = decision.flipped,
            jumpPower = decision.jumpPower,
            jumpAngle = decision.jumpAngle,
            diPower = decision.diPower,
            diAngle = decision.diAngle,
            selfFlipped = self.GetFacingVector().x < 0f ? 1f : 0f,
            selfHitstun = self.HitstunRemaining,
            selfBusy = self.IsBusy ? 1f : 0f,
        };

        Vector2 sp = self.GetPosition();
        Vector2 sv = self.GetVelocity();

        // Self position as bounded fractions of the stage rather than raw world units. selfX stays
        // absolute ON PURPOSE here even though the others are relative: "which side of the stage am I
        // on" is genuinely useful, and dividing by the half-width maps it to about -1..1 instead of
        // -12..12. Everything else that describes a relationship is relative to self.
        ctx.selfX = sp.x / Mathf.Max(StageHalfWidth, 0.001f);
        ctx.selfY = (sp.y - FloorY) / Mathf.Max(StageHeight, 0.001f);

        // Velocities divided by the ground speed cap. MAX_GROUND_SPEED is the natural unit: a velocity
        // expressed as a fraction of the fastest legal ground speed lands in roughly -1..1, whereas
        // raw velocity is unbounded and grows with knockback.
        ctx.selfVelX = sv.x / Mathf.Max(PhysicsConstants.MAX_GROUND_SPEED, 0.001f);
        ctx.selfVelY = sv.y / Mathf.Max(PhysicsConstants.MAX_GROUND_SPEED, 0.001f);

        ctx.selfGrounded = self.IsGrounded ? 1f : 0f;
        ctx.selfHealthFrac = self.Data.maxHealth > 0f ? self.GetHealth() / self.Data.maxHealth : 0f;
        // Hitstun is a frame count that can reach PhysicsConstants' stun caps, so it is expressed as a
        // fraction of the longest possible stun instead of a raw count.
        ctx.selfHitstun = self.HitstunRemaining / Mathf.Max(PhysicsConstants.MAX_HITSTUN_FRAMES, 1f);

        if (target != null)
        {
            Vector2 tp = target.GetPosition();
            Vector2 tv = target.GetVelocity();
            ctx.dx = tp.x - sp.x;
            ctx.dy = tp.y - sp.y;
            // Relative, normalised by the same stage dimensions. This is the single most important pair
            // of features in the whole vector and it is now symmetric about self by construction.
            ctx.targetX = ctx.dx / Mathf.Max(StageHalfWidth, 0.001f);
            ctx.targetY = ctx.dy / Mathf.Max(StageHeight, 0.001f);
            ctx.targetVelX = tv.x / Mathf.Max(PhysicsConstants.MAX_GROUND_SPEED, 0.001f);
            ctx.targetVelY = tv.y / Mathf.Max(PhysicsConstants.MAX_GROUND_SPEED, 0.001f);
            ctx.targetGrounded = target.IsGrounded ? 1f : 0f;
            ctx.targetHealthFrac = target.Data.maxHealth > 0f ? target.GetHealth() / target.Data.maxHealth : 0f;
            ctx.targetHitstun = target.HitstunRemaining / Mathf.Max(PhysicsConstants.MAX_HITSTUN_FRAMES, 1f);
            ctx.targetBusy = target.IsBusy ? 1f : 0f;
            ctx.targetBlocking = target.IsBlocking ? 1f : 0f;
            ctx.distance = Mathf.Sqrt(ctx.dx * ctx.dx + ctx.dy * ctx.dy)
                           / Mathf.Max(StageHalfWidth, 0.001f);
        }

        if (ally != null && ally != self)
        {
            Vector2 ap = ally.GetPosition();
            ctx.allyX = (ap.x - sp.x) / Mathf.Max(StageHalfWidth, 0.001f);
            ctx.allyY = (ap.y - sp.y) / Mathf.Max(StageHeight, 0.001f);
            ctx.hasAlly = true;
        }

        return ctx;
    }

    /// <summary>
    /// Flat float vector in a fixed order, for feeding straight into a model later.
    ///
    /// Every feature is already scaled to roughly -1..1 by <see cref="Capture"/> using the stage
    /// dimensions and the physics constants. That is deliberate rather than incidental: the raw vector
    /// spanned about 200x in magnitude (targetHitstun is a frame count that can reach 120, while
    /// selfHealthFrac is 0..1), and feeding that into a single hidden layer produced gradients so large
    /// that virtually every PPO update was rescaled by the global-norm clip, making the effective
    /// learning rate a function of the clip rather than of <c>learningRate</c>. Scaling at capture time
    /// keeps the statistics roughly comparable across features, which is what a single hidden layer
    /// needs; it is not a substitute for full running-mean normalisation, but it removes the worst of
    /// the disparity without introducing a train/inference skew.
    /// </summary>
    public float[] ToFeatureVector()
    {
        return new[]
        {
            selfX, selfY, selfVelX, selfVelY, selfGrounded, selfHealthFrac, selfHitstun, selfBusy, selfFlipped,
            targetX, targetY, targetVelX, targetVelY, targetGrounded, targetHealthFrac, targetHitstun,
            targetBusy, targetBlocking,
            allyX, allyY, hasAlly ? 1f : 0f,
            dx, dy, distance,
        };
    }

    /// <summary>
    /// Feature names, in the same order as <see cref="ToFeatureVector"/>.
    ///
    /// The names encode each feature's unit, because they are the only place that information exists:
    /// the vector is an anonymous float array, and "targetX" alone no longer says whether it is a world
    /// position, a relative offset, or a normalised fraction. Anything reading a CSV of observations
    /// depends on these names to interpret a column.
    /// </summary>
    public static string[] FeatureNames()
    {
        return new[]
        {
            "selfX_frac","selfY_frac","selfVelX_cap","selfVelY_cap","selfGrounded","selfHealthFrac",
            "selfHitstun_frac","selfBusy","selfFlipped",
            "targetRelX_frac","targetRelY_frac","targetVelX_cap","targetVelY_cap","targetGrounded",
            "targetHealthFrac","targetHitstun_frac","targetBusy","targetBlocking",
            "allyRelX_frac","allyRelY_frac","hasAlly",
            "dx_frac","dy_frac","distance_frac",
        };
    }
}