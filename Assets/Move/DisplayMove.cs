using System;
using UnityEngine;

public class DisplayMove : BaseMove
{
    private Sprite[] sprites;
    private bool looped;
    private string name;

    public DisplayMove(String name, Sprite[] sprites, bool looped)
    {
        this.name = name;
        this.sprites = sprites;
        this.looped = looped;
    }

    public override Sprite[] getSprites()
    {
        return sprites;
    }

    public override bool isLooped()
    {
        return looped;
    }

    public override string getName()
    {
        return this.name;
    }

    public override bool canInterrupt()
    {
        return true;
    }

    public override bool canHold()
    {
        return true;
    }
}
