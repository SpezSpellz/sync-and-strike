using System;
using System.Collections.Generic;
using UnityEngine;

public class Moves : MonoBehaviour
{
    public Sprite[] slash_move_sprites;
    public Sprite[] walk_move_sprites;
    public Sprite[] vert_slash_move_sprites;

    private Dictionary<string, Func<BaseMove>> moves = new Dictionary<string, Func<BaseMove>>();
    public static Moves Instance { get; private set; }

    public BaseMove getMove(string name)
    {
        return this.moves[name]();
    }
    void Awake()
    {
        if (Instance != null)
        {
            Debug.LogError("There is more than one instance!");
            return;
        }
        Instance = this;
        this.moves.Add("slash", () => new DisplayMove("slash", this.slash_move_sprites, false));
        this.moves.Add("walk", () => new WalkMove("walk", this.walk_move_sprites));
        this.moves.Add("vertical_slash", () => new DisplayMove("vertical_slash", this.vert_slash_move_sprites, false));
    }
    
}
