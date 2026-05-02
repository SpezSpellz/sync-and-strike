using NUnit.Framework;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class EnemyAgent : Agent
{
    [SerializeField]
    private Transform target;
    private CharacterController playerController;
    private static string[] moves = { "block", "dash", "horizontal_slash", "idle", "jump", "super_jump", "vertical_slash", "walkf" };
    private float prevDist = 0;
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
        sensor.AddObservation(new Vector2(this.target.position.x, this.target.position.y));
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        this.playerController = GetComponent<CharacterController>();
        this.prevDist = (this.target.position-this.transform.position).magnitude;
    }

    // Update is called once per frame
    void Update()
    {
        if (TurnManager.Instance.Phase != TurnPhase.Planning)
            return;
        if (TurnManager.Instance.IsSubmitted(playerController))
            return;
        var cd = (this.target.position - this.transform.position).magnitude;
        AddReward(this.prevDist - cd);
        this.prevDist = cd;
        this.RequestDecision();
    }
}
