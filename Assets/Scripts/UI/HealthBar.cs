using UnityEngine;

public class HealthBar : MonoBehaviour
{
    [SerializeField]
    private CharacterData characterData;
    private RectTransform rectTransform;
    private float maxSize;

    private void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        this.maxSize = this.rectTransform.rect.width;
    }
    void Update()
    {
        this.rectTransform.sizeDelta = new Vector2(this.maxSize * (characterData.health / characterData.maxHealth), this.rectTransform.sizeDelta.y);
    }
}
