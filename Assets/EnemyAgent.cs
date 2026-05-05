using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class EnemyAgent : Agent
{
    [SerializeField]
    private CharacterController target;
    private CharacterController playerController;
    private static string[] moves = { "block", "dash", "horizontal_slash", "idle", "jump", "super_jump", "vertical_slash", "walkf" };
    private float prevDist = 0;
    private float prevHealth;
    private float prevTargetHealth;
    public override void OnActionReceived(ActionBuffers actions)
    {
        var flip = actions.DiscreteActions[0] == 0;
        playerController.Flip(flip);
        playerController.SelectMove(moves[actions.DiscreteActions[1]]);
        TurnManager.Instance.SubmitMove(playerController);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(new Vector2(this.transform.position.x, this.transform.position.y));
        sensor.AddObservation(target.GetPosition());
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        this.playerController = GetComponent<CharacterController>();
        this.prevDist = (target.GetPosition() - playerController.GetPosition()).magnitude;
        this.prevHealth = this.playerController.GetHealth();
        this.prevTargetHealth = this.target.GetHealth();
    }

    public void MakeDecision()
    {
        if(this.playerController.IsDead())
        {
            AddReward(-10);
            EndEpisode();
            TurnManager.Instance.ResetState();
            return;
        }
        AddReward(this.playerController.GetHealth() - this.prevHealth);
        this.prevHealth = this.playerController.GetHealth();
        AddReward(this.prevTargetHealth - this.target.GetHealth());
        this.prevTargetHealth = this.target.GetHealth();
        var cd = (this.target.GetPosition() - playerController.GetPosition()).magnitude;
        AddReward(this.prevDist - cd);
        this.prevDist = cd;
        this.RequestDecision();
    }
}
