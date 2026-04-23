using UnityEngine;
using System.Collections.Generic;

public class HitboxController : MonoBehaviour
{
    public int damage = 10;
    public Vector2 knockback = new Vector2(5f, 2f);

    private HashSet<Collider2D> alreadyHit = new();

    private void Awake()
    {
        gameObject.SetActive(false);
    }

    public void Activate()
    {
        alreadyHit.Clear();
        gameObject.SetActive(true);
    }

    public void Deactivate()
    {
        gameObject.SetActive(false);
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (alreadyHit.Contains(other)) return;

        if (other.TryGetComponent<Hurtbox>(out var hurtbox))
        {
            alreadyHit.Add(other);
            CombatManager.Instance.RegisterHit(this, hurtbox);
        }
    }
}