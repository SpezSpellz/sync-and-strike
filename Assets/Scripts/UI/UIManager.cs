using UnityEngine;

public class UIManager : MonoBehaviour
{
    public static UIColorPalette Colors;

    public static UIManager Instance { get; private set; }
    [SerializeField] private UIColorPalette palette;
    [SerializeField] private RectTransform moveUI;
    [SerializeField] private RectTransform moveSettingUI;

    private void Awake()
    {
        Instance = this;
        Colors = palette;
    }

    public void HideMoveUI()
    {   
        moveUI.gameObject.SetActive(false);
        moveSettingUI.gameObject.SetActive(false);
    }
    public void ShowMoveUI()
    {
        moveUI.gameObject.SetActive(true);
        moveSettingUI.gameObject.SetActive(true);
    }
}