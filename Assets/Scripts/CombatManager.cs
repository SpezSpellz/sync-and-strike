using UnityEngine;
using System.Collections.Generic;

public class CombatManager : MonoBehaviour
{
    public static CombatManager Instance { get; private set; }

    private List<(HitboxController hitbox, Hurtbox hurtbox)> pendingHits = new();

    private void Awake()
    {
        Instance = this;
    }

    public void RegisterHit(HitboxController hitbox, Hurtbox hurtbox)
    {
        pendingHits.Add((hitbox, hurtbox));
    }

    public void ResolveAllHits()
    {
        if (pendingHits.Count == 0) return;

        bool isTrade = pendingHits.Count > 1;

        foreach (var (hitbox, hurtbox) in pendingHits)
        {
            Vector2 dir = (hurtbox.transform.position - hitbox.transform.position).normalized;
            Vector2 force = new Vector2(Mathf.Sign(dir.x) * hitbox.knockback.x, hitbox.knockback.y);
            hurtbox.owner.ApplyKnockback(force);

            if (isTrade)
                Debug.Log("Trade hit!");
        }

        pendingHits.Clear();
    }
}