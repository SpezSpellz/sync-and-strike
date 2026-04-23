using System;
using UnityEngine;

public abstract class BaseMove
{
    private int tick = 0;

    public void update(Humanoid humanoid)
    {
        this.tick = this.isLooped() ? ((this.tick + 1) % this.getTickLength()) : Math.Min(this.tick + 1, this.getTickLength() - 1);
        var renderer = humanoid.GetComponent<SpriteRenderer>();
        renderer.sprite = this.getSprites()[this.tick];
    }

    public int getTickLength()
    {
        return this.getSprites().Length;
    }

    public bool isEnded()
    {
        return (this.tick + 1) >= this.getTickLength();
    }

    public abstract string getName();

    public abstract bool isLooped();

    public abstract bool canInterrupt();

    public abstract bool canHold();

    public abstract Sprite[] getSprites();
}
