using UnityEngine;
using UnityEngine.UI;

public class MoveSelection : MonoBehaviour
{
    [SerializeField]
    private Button lockin_button;

    [SerializeField]
    private Button slash_button;

    [SerializeField]
    private Button walk_button;

    private string selected_move;

    void Start()
    {
        lockin_button.onClick.AddListener(() =>
        {
            World.Instance.getPlayer().setMove(Moves.Instance.getMove(this.selected_move));
        });
        slash_button.onClick.AddListener(() => selected_move = "slash");
        walk_button.onClick.AddListener(() => selected_move = "walk");
    }
}
