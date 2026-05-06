using UnityEngine;
using Random = System.Random;
class RuleBasedEnemyController : CharacterController
{
    [SerializeField]
    private CharacterController target;
    private Random random;
    public override void Start()
    {
        base.Start();
        this.random = new Random();
    }

    public override void RequestDecision()
    {
        if (this.IsDead())
        {
            TurnManager.Instance.ResetState();
            return;
        }
        this.Flip(target.GetPosition().x < this.GetPosition().x);
        SelectMove(random.Next(1) == 0 ? "horizontal_slash" : "vertical_slash");
        if (random.Next(5) == 0)
            SelectMove("jump");
        TurnManager.Instance.SubmitMove(this);
    }
}