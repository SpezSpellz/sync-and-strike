using UnityEngine;

public class UIManager : MonoBehaviour
{
    public static UIColorPalette Colors;

    [SerializeField] private UIColorPalette palette;

    private void Awake()
    {
        Colors = palette;
    }
}