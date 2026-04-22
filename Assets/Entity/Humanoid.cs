using UnityEngine;

public class Humanoid : Entity
{

    protected BaseMove current_move = null;

    public override void Start()
    {
        base.Start();
        current_move = Moves.Instance.getWalkMove();
    }

    public override void Step()
    {
        if (current_move != null)
        {
            current_move.update(this);
            if(current_move.isEnded())
                current_move = Moves.Instance.getWalkMove();
        }
    }
}
