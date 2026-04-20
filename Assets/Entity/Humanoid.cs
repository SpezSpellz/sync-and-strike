using UnityEngine;

public class Humanoid : Entity
{

    protected BaseMove current_move = null;

    void Start()
    {
        current_move = Moves.Instance.getWalkMove();
        World.Instance.addEntity(this);
    }

    public void Step()
    {
        if (current_move != null)
        {
            current_move.update(this);
            if(current_move.isEnded())
                current_move = Moves.Instance.getWalkMove();
        }
    }

    void Update()
    {
        Step();
    }
}
