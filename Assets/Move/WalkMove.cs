using System;
using UnityEngine;

public class WalkMove : BaseMove
{
    private Sprite[] sprites;
    private string name;

    public WalkMove(String name, Sprite[] sprites)
    {
        this.name = name;
        this.sprites = sprites;
    }
    public override void update(Humanoid humanoid)
    {
        base.update(humanoid);
        var velo = humanoid.getVelocity();
        humanoid.setVelocity(velo.x + 0.128f, velo.y);
    }

    public override Sprite[] getSprites()
    {
        return sprites;
    }

    public override bool isLooped()
    {
        return false;
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
