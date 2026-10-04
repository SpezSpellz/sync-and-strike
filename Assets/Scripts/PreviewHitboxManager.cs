using System.Collections.Generic;
using UnityEngine;

public class PreviewHitboxManager : MonoBehaviour
{
    public static PreviewHitboxManager Instance { get; private set; }
    [HideInInspector] public Arena Arena;
    private readonly List<HitBox> hitboxes = new();
    private readonly List<HurtBox> hurtboxes = new();

    private void Awake()
    {
        Instance = this;
        Arena = GetComponentInParent<Arena>();
    }

    public void SubmitHitBox(HitBox hitbox) => hitboxes.Add(hitbox);
    public void SubmitHurtBox(HurtBox hurtbox) => hurtboxes.Add(hurtbox);

    public void Step()
    {
        foreach (var hitbox in hitboxes)
        {
            foreach (var hurtbox in hurtboxes)
            {
                if (hitbox.attacker == hurtbox.owner) continue;
                if (hitbox.attacker != null && hurtbox.owner != null
                    && !CombatTeamUtility.AreEnemies(hitbox.attacker.Team, hurtbox.owner.Team))
                    continue;
                if (hitbox.intersectWith(hurtbox))
                    hitbox.OnHit(hurtbox.owner);
            }
        }

        hitboxes.Clear();
        hurtboxes.Clear();
    }
}