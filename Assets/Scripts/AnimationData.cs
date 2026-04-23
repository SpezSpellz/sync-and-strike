using UnityEngine;
using System;

[Serializable]
public class AnimationData
{
    public Move move;
    public Sprite[] frames;
    public float frameTime = 0.1f;
    public bool loop = false; // most combat moves don't loop
    public FrameEvent[] events;
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