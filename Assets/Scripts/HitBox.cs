using System;
public class HitBox : AABB
{
    public CharacterController attacker { get; private set; }
    private Action<CharacterController> onHit;
    private bool isHit = false;
    public HitBox(CharacterController attacker, Action<CharacterController> onHit, float minX, float minY, float maxX, float maxY) : base(minX, minY, maxX, maxY)
    {
        this.attacker = attacker;
        this.onHit = onHit;
    }

    public void OnHit(CharacterController target)
    {
        if (this.isHit)
            return;
        this.onHit(target);
        this.isHit = true;
    }
}