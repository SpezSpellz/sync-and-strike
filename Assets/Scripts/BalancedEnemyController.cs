using UnityEngine;
using Random = System.Random;
class BalancedEnemyController : CharacterController
{
    [SerializeField]
    private CharacterController target;
    private Random random;
    private bool rule_based = false;
    private static string[] moves = { "block", "dash", "horizontal_slash", "idle", "jump", "super_jump", "vertical_slash", "walkf" };

    public override void Start()
    {
        base.Start();
        this.random = new Random();
    }

    public override void Load(SaveData savedata)
    {
        base.Load(savedata);
        this.rule_based = !this.rule_based;
    }

    public override void RequestDecision()
    {
        if(rule_based)
        {
            RequestRuleBasedDecision();
        } else
        {
            RequestRandomDecision();
        }
    }

    void RequestRandomDecision()
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

    void RequestRuleBasedDecision()
    {
        if (this.IsDead())
        {
            TurnManager.Instance.ResetState();
            return;
        }
        bool face = target.GetPosition().x < this.GetPosition().x;
        if (random.Next(5) == 0)
            face = !face;
        this.Flip(face);
        SelectMove(random.Next(1) == 0 ? "horizontal_slash" : "vertical_slash");
        if (random.Next(5) == 0)
            SelectMove("jump");
        TurnManager.Instance.SubmitMove(this);
    }
}