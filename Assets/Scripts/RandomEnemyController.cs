using UnityEngine;
using Random = System.Random;

public class RandomEnemyController : CharacterController
{

    private Random random = new Random();
    [SerializeField]
    private CharacterController target;
    private static string[] moves = { "block", "dash", "horizontal_slash", "idle", "jump", "super_jump", "vertical_slash", "walkf" };

    public override void RequestDecision()
    {
        if (this.IsDead())
        {
            TurnManager.Instance.ResetState();
            return;
        }
        this.Flip(random.Next(100) < 50);
        SelectMove(moves[random.Next(moves.Length - 1)]);
        TurnManager.Instance.SubmitMove(this);
    }
}
