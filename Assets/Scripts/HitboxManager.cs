using UnityEngine;
using System.Collections.Generic;

public class HitboxManager : MonoBehaviour
{
    private List<HitBox> hitboxes = new();
    private List<HurtBox> hurtboxes = new();
    public static HitboxManager Instance;
    private void Awake()
    {
        Instance = this;
    }
    public void SubmitHitBox(HitBox hitbox)
    {
        this.hitboxes.Add(hitbox);
    }

    public void SubmitHurtBox(HurtBox hurtbox)
    {
        this.hurtboxes.Add(hurtbox);
    }

    public void Step()
    {
        foreach(HitBox hitbox in this.hitboxes)
        {
            foreach(HurtBox hurtbox in this.hurtboxes)
            {
                if (hitbox.attacker == hurtbox.owner)
                    continue;
                if (hitbox.intersectWith(hurtbox))
                    hitbox.OnHit(hurtbox.owner);
            }
        }
        this.hitboxes.Clear();
        this.hurtboxes.Clear();
    }
}