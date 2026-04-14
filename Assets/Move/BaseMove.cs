using UnityEngine;

public abstract class BaseMove
{
    private const float SECONDS_PER_FRAME = 0.016f;
    private float time = 0;

    public void update(Humanoid humanoid)
    {
        this.time += Time.deltaTime;
        this.time = this.isLooped() ? (this.time % this.getTimeLength()) : Mathf.Min(this.time, this.getTimeLength());
        var renderer = humanoid.GetComponent<SpriteRenderer>();
        renderer.sprite = this.getSprites()[(int)(this.time * (1 / SECONDS_PER_FRAME))];
    }

    public float getTimeLength()
    {
        return this.getSprites().Length * SECONDS_PER_FRAME;
    }

    public bool isEnded()
    {
        return this.time >= this.getTimeLength();
    }

    public abstract string getName();

    public abstract bool isLooped();

    public abstract Sprite[] getSprites();
}
