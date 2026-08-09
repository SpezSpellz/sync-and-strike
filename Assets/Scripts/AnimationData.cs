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