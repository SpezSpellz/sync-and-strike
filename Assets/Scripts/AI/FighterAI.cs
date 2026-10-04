using System;
using UnityEngine;

/// <summary>
/// One turn's worth of AI intent. Deliberately serialisable so the same struct can be
/// written to the training log alongside the player's vote.
/// </summary>
[Serializable]
public struct AIDecision
{
    public string moveId;
    public bool flipped;
    public float jumpPower;
    public float jumpAngle;
    public float diPower;
    public float diAngle;
    /// <summary>Short human readable reason, useful while tuning the rules.</summary>
    public string rationale;

    public override string ToString()
    {
        return $"{moveId} (flip={flipped}, jump={jumpPower:0.00}@{jumpAngle:0.00}, di={diPower:0.00}@{diAngle:0.00}) {rationale}";
    }
}

/// <summary>
/// Rule-based ("algorithm based") fighter AI used by both the companion and the enemy.
/// It replaces the ML-Agents policy that used to drive the enemy so both sides play with the
/// same logic, and can later be swapped for a trained model using the collected vote data.
///
/// The brain only ever picks from <see cref="CharacterData.animations"/>, so the companion
/// automatically shares the player's moveset without any duplicated move list.
/// </summary>
public class FighterAI : FighterPolicy
{
    [Serializable]
    public class Personality
    {
        [Header("Distance")]
        [Tooltip("Below this distance the AI is happy to swing.")]
        public float attackRange = 1.35f;
        [Tooltip("Above this distance the AI tries to close the gap.")]
        public float approachRange = 2.6f;

        [Header("Aggression (0..1)")]
        [Tooltip("How often the AI pressures instead of blocking or repositioning.")]
        [Range(0f, 1f)] public float aggression = 0.6f;
        [Tooltip("Chance to guard when the opponent is attacking.")]
        [Range(0f, 1f)] public float blockChance = 0.35f;
        [Tooltip("Chance to jump in as an approach or mix-up.")]
        [Range(0f, 1f)] public float jumpChance = 0.25f;
        [Tooltip("How often the AI whiff-punishes an opponent stuck in recovery.")]
        [Range(0f, 1f)] public float punishChance = 0.5f;

        [Header("Companion behaviour")]
        [Tooltip("The companion keeps out of the player's way instead of crowding it.")]
        public bool avoidAlly = false;
        [Tooltip("Distance the companion tries to keep from its ally.")]
        public float allyBuffer = 1.6f;
    }

    private readonly System.Random random;
    private readonly Personality profile;

    /// <summary>Set by the last <see cref="Decide"/> call: true when the AI chose to hop away.</summary>
    private bool retreatJump;

    public FighterAI(Personality profile, int seed)
    {
        this.profile = profile;
        this.random = new System.Random(seed);
    }

    /// <summary>
    /// Chooses a move for <paramref name="self"/> against <paramref name="target"/>.
    /// <paramref name="ally"/> is the other fighter on the same team (the player, when this
    /// brain drives the companion) and is only used for spacing.
    /// </summary>
    public override AIDecision Decide(CharacterController self, CharacterController target, CharacterController ally)
    {
        retreatJump = false;

        var decision = new AIDecision
        {
            moveId = "idle",
            flipped = false,
            jumpPower = 1f,
            jumpAngle = Mathf.PI * 0.5f,
            diPower = 0f,
            diAngle = 0f,
            rationale = "no target",
        };

        if (self == null || target == null)
            return decision;

        Vector2 selfPos = self.GetPosition();
        Vector2 targetPos = target.GetPosition();

        float dx = targetPos.x - selfPos.x;
        float dy = targetPos.y - selfPos.y;
        float distance = Mathf.Sqrt(dx * dx + dy * dy);

        bool faceTarget = dx >= 0f;
        // Occasional feints so the AI is not trivially predictable.
        if (NextFloat() < 0.06f) faceTarget = !faceTarget;

        string move = PickMove(self, target, ally, dx, dy, distance, faceTarget, out string rationale);

        bool facingAway = retreatJump;
        decision.moveId = move;
        decision.rationale = rationale;
        // When hopping backwards to make room, face the opponent anyway so the
        // companion's hitbox still points at the enemy rather than at its ally.
        decision.flipped = facingAway ? faceTarget : !faceTarget;
        decision.jumpPower = retreatJump ? 0.5f : Mathf.Lerp(0.5f, 1f, NextFloat());
        decision.jumpAngle = retreatJump
            ? (faceTarget ? 0.5235987755982988f : 2.6179938779914944f) // minimum-height hop backwards
            : ComputeJumpAngle(dx, dy, faceTarget);
        decision.diPower = ComputeDIPower(distance);
        decision.diAngle = ComputeDIAngle(faceTarget);
        return decision;
    }

    private string PickMove(
        CharacterController self,
        CharacterController target,
        CharacterController ally,
        float dx, float dy, float distance, bool faceTarget, out string rationale)
    {
        rationale = "";

        // --- Threat detection -------------------------------------------------
        bool targetIsAttacking = target.IsBusy && !target.IsBlocking;
        bool targetIsRecovering = target.HitstunRemaining > 0;

        // --- Companion: don't crowd the player ---------------------------------
        bool tooCloseToAlly = false;
        if (profile.avoidAlly && ally != null && !ally.IsDead())
        {
            float allyDist = Mathf.Abs(ally.GetPosition().x - self.GetPosition().x);
            tooCloseToAlly = allyDist < profile.allyBuffer;
        }

        // --- Punish window ----------------------------------------------------
        // Hitting an opponent during their recovery frames is the strongest punish available.
        if (targetIsRecovering && distance <= profile.attackRange * 1.2f && NextFloat() < profile.punishChance)
        {
            rationale = "punish recovery";
            return AttackMove(self, dy);
        }

        // --- Defensive read ----------------------------------------------------
        if (targetIsAttacking && distance <= profile.attackRange * 1.4f && NextFloat() < profile.blockChance)
        {
            if (CanUse(self, "block"))
            {
                rationale = "block on read";
                return "block";
            }
        }

        // --- Positioning: back off so the player has room ----------------------
        if (tooCloseToAlly)
        {
            if (self.IsGrounded && CanUse(self, "jump") && NextFloat() < profile.jumpChance)
            {
                retreatJump = true;
                rationale = "make room for the player";
                return "jump";
            }
            rationale = "make room for the player";
            return CanUse(self, "dash") ? "dash" : "walkf";
        }

        // --- In range: swing or hold ------------------------------------------
        if (distance <= profile.attackRange && Mathf.Abs(dy) < 0.8f)
        {
            if (NextFloat() < profile.aggression)
            {
                rationale = "in range, swing";
                return AttackMove(self, dy);
            }
            if (CanUse(self, "block") && NextFloat() < 0.4f)
            {
                rationale = "hold ground";
                return "block";
            }
            rationale = "keep spacing";
            return CanUse(self, "walkf") ? "walkf" : "idle";
        }

        // --- Medium range: close or jump ---------------------------------------
        if (distance <= profile.approachRange)
        {
            if (dy > 0.35f)
            {
                // The opponent is above us: rise into a vertical attack.
                if (CanUse(self, "vertical_slash") && NextFloat() < 0.6f)
                {
                    rationale = "opponent is higher";
                    return "vertical_slash";
                }
                if (self.IsGrounded && NextFloat() < profile.jumpChance)
                {
                    rationale = "jump to close vertically";
                    return CanUse(self, "jump") ? "jump" : "super_jump";
                }
                rationale = "close the gap";
                return CanUse(self, "dash") ? "dash" : "walkf";
            }

            if (self.IsGrounded && NextFloat() < profile.jumpChance * 1.4f)
            {
                rationale = "jump mix-up";
                return CanUse(self, "jump") ? "jump" : "super_jump";
            }
            rationale = "close the gap";
            return CanUse(self, "dash") ? "dash" : "walkf";
        }

        // --- Long range: close hard -------------------------------------------
        if (distance > profile.approachRange * 1.6f && self.IsGrounded && CanUse(self, "super_jump"))
        {
            rationale = "close distance";
            return "super_jump";
        }

        rationale = "walk forward";
        return CanUse(self, "walkf") ? "walkf" : "idle";
    }

    /// <summary>Only return a move the character can actually perform right now.</summary>
    private bool CanUse(CharacterController self, string moveId)
    {
        if (!self.Data.HasMove(moveId)) return false;
        var data = self.Data.GetMove(moveId);
        if (data == null) return false;
        return self.CanUseMove(data);
    }

    private string AttackMove(CharacterController self, float dy)
    {
        if (dy > 0.25f && self.Data.HasMove("vertical_slash"))
            return "vertical_slash";
        if (self.Data.HasMove("horizontal_slash"))
            return "horizontal_slash";
        if (self.Data.HasMove("vertical_slash"))
            return "vertical_slash";
        return "idle";
    }

    private float ComputeJumpAngle(float dx, float dy, bool faceTarget)
    {
        // Home jumps towards the opponent but never steeper than about -0.34 on the
        // Y axis, so jumps are always at least ~20 degrees above horizontal.
        Vector2 dir = new Vector2(dx, Mathf.Max(dy * 0.6f, -0.204f));
        if (Mathf.Abs(dir.x) < 0.001f) dir.x = (faceTarget ? 1f : -1f) * 0.94f;
        dir = dir.normalized;
        // Expressed so 0 rad is straight up, matching the jump wheel and setJumpInfo.
        float angle = Mathf.Atan2(dir.x, dir.y);
        return Mathf.Clamp(angle, 0.5235987755982988f, 2.6179938779914944f);
    }

    private float ComputeDIPower(float distance)
    {
        // Knockback indirection matters most when you are about to be cornered,
        // so use it hardest at close range and almost never at long range.
        float t = Mathf.Clamp01(distance / 4f);
        float power = Mathf.Lerp(0.6f, 0.1f, t);
        return NextFloat() < 0.35f ? power : power * 0.3f;
    }

    private float ComputeDIAngle(bool faceTarget)
    {
        // Aim the DI slightly downward into the ground so it can save a cornered jump.
        return faceTarget ? -0.35f : Mathf.PI + 0.35f;
    }

    private float NextFloat()
    {
        return (float)random.NextDouble();
    }
}