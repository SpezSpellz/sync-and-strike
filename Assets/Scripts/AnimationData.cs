using UnityEngine;
using System;

[CreateAssetMenu(fileName = "NewAnimationData", menuName = "Data/AnimationData")]
public class AnimationData : ScriptableObject
{
    [Header("Identity")]
    public MoveType move;
    public string moveId;               // e.g. "punch", "uppercut", "sweep_kick"
    public string moveName;             // e.g. "Punch", "Uppercut", "Sweep Kick" (display name)
    public bool onlyGrounded;           // Must be on ground to execute this move when onlyGrounded is true (e.g. jump)
    public bool usableInKnockedback;    // This move is executable even in knockback animation when usableInKnockedback is true (e.g. burst)
    public Sprite icon;
    public RequiredInput requiredInput;

    [Header("Animation")]
    public Sprite[] frames;
    public bool loop = false;

    [Header("Frame Events")]
    public FrameEvent[] events;

    // Start from frame 1 not 0
    [Header("Frame Data")]
    public int firstActionable;

    [Header("Physics")]
    public bool continuousImpulse;
    public Vector2 impulse;
    public float knockback;

    /// <summary>
    /// Jump impulse in pixels, before <see cref="PhysicsConstants.BASE_JUMP_SPEED"/> is added.
    /// Authored per jump state, matching how the reference data stores it, because it genuinely
    /// varies by character and by jump type. Leave at 0 for
    /// <see cref="PhysicsConstants.DEFAULT_JUMP_SPEED"/>.
    /// </summary>
    public float jumpSpeed;

    /// <summary>Jump impulse in pixels, falling back to the default when unauthored.</summary>
    public float JumpSpeed => jumpSpeed > 0f ? jumpSpeed : PhysicsConstants.DEFAULT_JUMP_SPEED;
}

[Serializable]
public struct FrameEvent
{
    public int frame;
    public FrameEventType type;
    public HitboxData hitboxData;    // only used when type is SpawnHitbox
}

[Serializable]
public struct HitboxData
{
    public float offsetX;       // forward offset from character center
    public float offsetY;       // vertical offset from character center
    public float width;
    public float height;
    public int damage;
    public Vector2 knockback;
    public HitHeight hitHeight; // decides whether a grounded victim pops up or stays grounded

    // --- Core hit properties ---
    public int hitstunTicks;              // 0 uses PhysicsConstants.HITSTUN_FRAMES
    public bool knockdown;                // ends in a knockdown once the victim lands
    public bool hardKnockdown;            // knockdown variant that cannot be acted out of
    public bool wallSlam;                 // slam into the stage wall
    public bool groundBounce;             // bounce off the floor after a launch
    public bool airGroundBounce;          // bounce off the floor specifically from the air
    public float groundBounceKnockbackModifier;
    public bool knockdownExtendsHitstun;
    public int minimumGroundedFrames;
    public bool canCounterHit;

    // --- Per-hitbox hurt-state selection ---
    /// <summary>
    /// Marks this hit as a launcher, putting a grounded victim into HurtAerial instead of
    /// HurtGrounded so they keep the hit's vertical knockback rather than having it zeroed.
    /// Without this a vertical slash does nothing at all to someone standing on the floor.
    /// </summary>
    public bool launchGrounded;

    /// <summary>
    /// Body collision is disabled through a hit by default. Entering a hurt state from such a
    /// hit clears it, so two characters in hitstun pass through each other and the mutual
    /// pushback alone separates them. The flag is restored on every state exit.
    ///
    /// HitboxData is a struct on C# 9 with no field initialisers, and move assets authored before
    /// this field existed deserialise it as false, which would mean "always collide" rather than
    /// the intended default. <see cref="DisableCollision"/> therefore treats false as unset and
    /// resolves to true. Set <see cref="keepCollisionOnHit"/> explicitly only to KEEP collision
    /// through a hit (throws and grabs, which need to stay attached).
    /// </summary>
    public bool keepCollisionOnHit;

    public bool DisableCollision => !keepCollisionOnHit;

    // --- Knockback tuning ---
    /// <summary>Scales the DI force this hitbox contributes.</summary>
    public float diModifier;

    /// <summary>The hit's own share of the mutual pushback.</summary>
    public float pushbackX;

    // --- Frame advantage ---
    public int hitlagTicks;               // 0 uses PhysicsConstants.HITLAG_FRAMES (attacker)
    public int victimHitlag;              // 0 uses PhysicsConstants.VICTIM_HITLAG_FRAMES
    public int plusFrames;                // extra blockstun granted when this hit is blocked

    // --- Combo scaling ---
    public int damageProration;
    public bool scaleCombo;
    public bool incrementCombo;

    /// <summary>
    /// Default 1. A multi-hit move can advance the combo counter by more than one, which
    /// shifts the damage scale-off.
    /// </summary>
    public int comboScalingAmount;

    public int ComboScalingAmount => comboScalingAmount > 0 ? comboScalingAmount : 1;

    public float DiModifier => diModifier != 0f ? diModifier : 1f;
    public float PushbackX => pushbackX != 0f ? pushbackX : PhysicsConstants.DEFAULT_PUSHBACK_X;

    /// <summary>Attacker hitlag, defaulting to 4 when the field is unset.</summary>
    public int HitlagFrames => hitlagTicks > 0 ? hitlagTicks : PhysicsConstants.HITLAG_FRAMES;

    /// <summary>
    /// Victim hitlag, defaulting to the same value as the attacker's when unset. A hit that
    /// freezes the defender noticeably longer than the attacker is what gives a move its plus
    /// frames on hit.
    /// </summary>
    public int VictimHitlagFrames => victimHitlag > 0 ? victimHitlag : PhysicsConstants.VICTIM_HITLAG_FRAMES;

    // --- Guard properties ---
    // A blocked hit's knockback is DIVIDED by a divisor rather than scaled by a flat fraction,
    // and each hitbox can tune the result. All defaults are 1.
    //
    // These cannot carry initialisers: HitboxData is a struct and the project is on C# 9, which
    // has no struct field initialisers. Move assets authored before these fields existed also
    // deserialise them as 0, so read them through the accessors below, which treat 0 as "unset".
    // Same convention as groundBounceKnockbackModifier.
    public float blockPushbackModifier;         // tunes the blocker's pushback
    public bool blockPushbackReversible;       // flip pushback when blocked from the far side
    public float blockReversePushbackModifier; // scale for that reversed case
    public float chipDamageModifier;           // tunes chip damage

    /// <summary>Block pushback modifier, defaulting to 1 when unset.</summary>
    public float BlockPushback => blockPushbackModifier != 0f ? blockPushbackModifier : 1f;

    /// <summary>Reversed block pushback modifier, defaulting to 1 when unset.</summary>
    public float BlockReversePushback =>
        blockReversePushbackModifier != 0f ? blockReversePushbackModifier : 1f;

    /// <summary>Chip damage modifier, defaulting to 1 when unset.</summary>
    public float ChipDamage => chipDamageModifier != 0f ? chipDamageModifier : 1f;
}

public enum RequiredInput
{
    None,
    JumpWheel,
    DirectionWheel,
}

public enum FrameEventType
{
    SpawnHitbox,
    SpawnVFX,
    SpawnSFX,
    Block,
    Jump
}

/// <summary>
/// Which part of the body a hit connects with. High/Mid hits launch a grounded victim straight back,
/// Low hits keep them on the floor.
/// </summary>
public enum HitHeight
{
    High,
    Mid,
    Low,
}

public enum MoveType
{
    Idle,
    Movement,
    Attack,
    Special,
    Super,
    Defense,
    Hurt
}