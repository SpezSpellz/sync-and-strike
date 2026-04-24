using UnityEngine;
using System;

[CreateAssetMenu(fileName = "NewAnimationData", menuName = "Data/AnimationData")]
public class AnimationData : ScriptableObject
{
    [Header("Identity")]
    public MoveType move;
    public string moveId;      // e.g. "punch", "uppercut", "sweep_kick"
    public string moveName;    // e.g. "Punch", "Uppercut", "Sweep Kick" (display name)
    public Sprite icon;

    [Header("Animation")]
    public Sprite[] frames;
    public bool loop = false;

    [Header("Frame Events")]
    public FrameEvent[] events;

    [Header("Physics")]
    public Vector2 impulse;
    public float knockback;
}

[Serializable]
public struct FrameEvent
{
    public int frame;
    public FrameEventType type;
}

public enum FrameEventType
{
    ActivateHitbox,
    DeactivateHitbox,
    SpawnFX
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