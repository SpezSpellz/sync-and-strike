class EnemyController : CharacterController
{
    private EnemyAgent agent;
    public override void Start()
    {
        base.Start();
        agent = GetComponent<EnemyAgent>();
    }

    public void ResetMetrics()
    {
        agent.ResetMetrics();
    }

    public override void RequestDecision()
    {
        this.agent.MakeDecision();
    }
}