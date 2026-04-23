using UnityEngine;
using Random = System.Random;

public class RandomEnemy : Humanoid
{

    private Random random = new Random();
    public override void Step()
    {
        base.Step();
        this.veloX /= 10;
        if ((this.random.Next(5) == 0) && this.isOnGround())
            this.veloY += 0.128f;
        if (this.random.Next(2) == 0)
        {
            this.GetComponent<SpriteRenderer>().flipX = true;
            this.veloX -= 0.128f;
        }
        if (this.random.Next(2) == 0)
        {
            this.GetComponent<SpriteRenderer>().flipX = false;
            this.veloX += 0.128f;
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
    }
}
