using UnityEngine;
using System.Collections.Generic;
using Unity.VisualScripting;

public class HitboxManager : MonoBehaviour
{
    [SerializeField]
    private GameObject hitboxPrefab;
    [SerializeField]
    private bool showHitboxes;
    private List<HitBox> hitboxes = new();
    private List<HurtBox> hurtboxes = new();
    private List<GameObject> hurtbox_visualizations = new();
    private List<GameObject> hitbox_visualizations = new();
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
        if (this.showHitboxes)
        {
            foreach (GameObject hitbox_view in this.hurtbox_visualizations)
            {
                Destroy(hitbox_view);
            }
            this.hurtbox_visualizations.Clear();
            this.hitbox_visualizations.RemoveAll((hitbox_view) =>
                {
                    var sprite_renderer = hitbox_view.GetComponent<SpriteRenderer>();
                    if (sprite_renderer.color.a < 0.01)
                    {
                        Destroy(hitbox_view);
                        return true;
                    }
                    sprite_renderer.color = new Color(sprite_renderer.color.r, sprite_renderer.color.g, sprite_renderer.color.b, sprite_renderer.color.a * 0.9f);
                    return false;
                }
            );
            foreach (HurtBox hurtbox in this.hurtboxes)
            {
                var hitbox_view = Instantiate(hitboxPrefab, this.transform);
                hitbox_view.GetComponent<SpriteRenderer>().color = new Color(1f, 1f, 0f, 0.5f);
                hitbox_view.transform.localScale = new Vector3(hurtbox.getWidth(), hurtbox.getHeight());
                hitbox_view.transform.position = hurtbox.getCenter();
                hurtbox_visualizations.Add(hitbox_view);
            }
        }
        foreach(HitBox hitbox in this.hitboxes)
        {
            if (this.showHitboxes)
            {
                var hitbox_view = Instantiate(hitboxPrefab, this.transform);
                hitbox_view.GetComponent<SpriteRenderer>().color = new Color(1f, 0f, 0f, 0.5f);
                hitbox_view.transform.localScale = new Vector3(hitbox.getWidth(), hitbox.getHeight());
                hitbox_view.transform.position = hitbox.getCenter();
                hitbox_visualizations.Add(hitbox_view);
            }
            foreach (HurtBox hurtbox in this.hurtboxes)
            {
                if (hitbox.attacker == hurtbox.owner)
                    continue;
                if (hitbox.intersectWith(hurtbox))
                {
                    hitbox.OnHit(hurtbox.owner);
                }
            }
        }
        this.hitboxes.Clear();
        this.hurtboxes.Clear();
    }
}