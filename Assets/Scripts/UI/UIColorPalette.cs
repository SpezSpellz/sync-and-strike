using UnityEngine;

[CreateAssetMenu(fileName = "UIColorPalette", menuName = "UI/Color Palette")]
public class UIColorPalette : ScriptableObject
{
    public Color primary;
    public Color secondary;
    public Color toggleOn;
    public Color toggleOff;
    public Color disabled;
    public Color moveSelected;
}