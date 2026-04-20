using UnityEngine;
using Random = System.Random;

public class RandomEnemy : Humanoid
{

    private Random random = new Random();
    public void Step()
    {
        base.Step();
        var physics = this.GetComponent<Physics>();
        float culVeloY = 0;
        float culVeloX = 0;
        if ((this.random.Next(5) == 0) && physics.isOnGround())
            culVeloY += 8;
        if (this.random.Next(2) == 0)
        {
            this.GetComponent<SpriteRenderer>().flipX = true;
            culVeloX -= 40;
        }
        if (this.random.Next(2) == 0)
        {
            this.GetComponent<SpriteRenderer>().flipX = false;
            culVeloX += 40;
        }
        if (this.random.Next(10) == 0)
        {
            if(this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getSlashMove();
        }
        if (this.random.Next(10) == 0)
        {
            if (this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getVerticalSlashMove();
        }
        physics.applyVelocity(culVeloX, culVeloY);
    }

    void Update()
    {
        Step();
    }
}
