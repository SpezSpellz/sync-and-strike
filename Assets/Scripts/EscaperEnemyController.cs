using UnityEngine;
using Random = System.Random;
class EscaperEnemyController : CharacterController
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
        bool face = target.GetPosition().x > this.GetPosition().x;
        if (random.Next(100) < 50)
            face = !face;
        this.Flip(face);
        if (random.Next(100) < 10 && Mathf.Abs(this.GetPosition().y - target.GetPosition().y) > 2f)
        {
            SelectMove(random.Next(1) == 0 ? "horizontal_slash" : "vertical_slash");
        } else {
            SelectMove("jump");

        }
        TurnManager.Instance.SubmitMove(this);
    }
}