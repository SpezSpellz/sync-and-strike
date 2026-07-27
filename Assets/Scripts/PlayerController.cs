using UnityEngine;

class PlayerController : CharacterController
{
    [SerializeField]
    private Transform EnemyPosition;

    public override void Start()
    {
        base.Start();
        base.TargetPosition = this.EnemyPosition;
    }
}