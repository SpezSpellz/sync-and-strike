using UnityEngine;

public class Moves : MonoBehaviour
{
    public Sprite[] slash_move_sprites;
    public BaseMove getSlashMove()
    {
        return new DisplayMove("slash", this.slash_move_sprites, false);
    }

    public Sprite[] walk_move_sprites;
    public BaseMove getWalkMove()
    {
        return new DisplayMove("walk", this.walk_move_sprites, true);
    }

    public Sprite[] vert_slash_move_sprites;
    public BaseMove getVerticalSlashMove()
    {
        return new DisplayMove("vertical_slash", this.vert_slash_move_sprites, false);
    }
    public static Moves Instance { get; private set; }
    void Awake()
    {
        if (Instance != null)
        {
            Debug.LogError("There is more than one instance!");
            return;
        }
        Instance = this;
    }
    
}
