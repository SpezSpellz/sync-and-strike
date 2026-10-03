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

    public float targetX, targetY, targetVelX, targetVelY;
    public float targetGrounded;
    public float targetHealthFrac;
    public float targetHitstun;
    public float targetBusy;
    public float targetBlocking;

    public float allyX, allyY;
    public bool hasAlly;

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
        ctx.selfX = sp.x; ctx.selfY = sp.y;
        ctx.selfVelX = sv.x; ctx.selfVelY = sv.y;
        ctx.selfGrounded = self.IsGrounded ? 1f : 0f;
        ctx.selfHealthFrac = self.Data.maxHealth > 0f ? self.GetHealth() / self.Data.maxHealth : 0f;

        if (target != null)
        {
            Vector2 tp = target.GetPosition();
            Vector2 tv = target.GetVelocity();
            ctx.targetX = tp.x; ctx.targetY = tp.y;
            ctx.targetVelX = tv.x; ctx.targetVelY = tv.y;
            ctx.targetGrounded = target.IsGrounded ? 1f : 0f;
            ctx.targetHealthFrac = target.Data.maxHealth > 0f ? target.GetHealth() / target.Data.maxHealth : 0f;
            ctx.targetHitstun = target.HitstunRemaining;
            ctx.targetBusy = target.IsBusy ? 1f : 0f;
            ctx.targetBlocking = target.IsBlocking ? 1f : 0f;
            ctx.dx = tp.x - sp.x;
            ctx.dy = tp.y - sp.y;
            ctx.distance = Mathf.Sqrt(ctx.dx * ctx.dx + ctx.dy * ctx.dy);
        }

        if (ally != null && ally != self)
        {
            Vector2 ap = ally.GetPosition();
            ctx.allyX = ap.x; ctx.allyY = ap.y;
            ctx.hasAlly = true;
        }

        return ctx;
    }

    /// <summary>Flat float vector in a fixed order, for feeding straight into a model later.</summary>
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

    /// <summary>Feature names, in the same order as <see cref="ToFeatureVector"/>.</summary>
    public static string[] FeatureNames()
    {
        return new[]
        {
            "selfX","selfY","selfVelX","selfVelY","selfGrounded","selfHealthFrac","selfHitstun","selfBusy","selfFlipped",
            "targetX","targetY","targetVelX","targetVelY","targetGrounded","targetHealthFrac","targetHitstun",
            "targetBusy","targetBlocking",
            "allyX","allyY","hasAlly",
            "dx","dy","distance",
        };
    }
}