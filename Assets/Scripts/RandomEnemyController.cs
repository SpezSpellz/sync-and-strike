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
        this.setKnockbackInfo((random.Next(100) / 100f), random.Next(360) * 0.017453292519943295f);
        this.setJumpInfo((random.Next(100) / 200f) + 0.5f, random.Next(360) * 0.017453292519943295f);
        SelectMove(moves[random.Next(moves.Length - 1)]);
        TurnManager.Instance.SubmitMove(this);
    }
}
