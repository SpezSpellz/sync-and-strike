using UnityEngine;

public class HealthBar : MonoBehaviour
{
    [SerializeField]
    private CharacterData characterData;
    [SerializeField]
    private bool flip;
    private RectTransform rectTransform;
    private float maxSize;

    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        this.maxSize = this.rectTransform.rect.width;
    }
    void Update()
    {
        float t = characterData.health / characterData.maxHealth;
        this.rectTransform.sizeDelta = new Vector2(this.maxSize * t, this.rectTransform.sizeDelta.y);
        this.rectTransform.localPosition = new Vector3((flip ? -((1-t)/2) : ((1-t)/2)) * this.maxSize, 0, 0);
    }
}
