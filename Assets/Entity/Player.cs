using UnityEngine;

public class Player : Humanoid
{
    public override void Step()
    {
        base.Step();
        var physics = this.GetComponent<Physics>();
        float culVeloY = 0;
        float culVeloX = 0;
        if (Input.GetKey(KeyCode.Space) && physics.isOnGround())
            culVeloY += 8;
        if (Input.GetKey(KeyCode.A))
        {
            this.GetComponent<SpriteRenderer>().flipX = true;
            culVeloX -= 40;
        }
        if (Input.GetKey(KeyCode.D))
        {
            this.GetComponent<SpriteRenderer>().flipX = false;
            culVeloX += 40;
        }
        if (Input.GetKey(KeyCode.E))
        {
            if (this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getSlashMove();
        }
        if (Input.GetKey(KeyCode.R))
        {
            if (this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getVerticalSlashMove();
        }
        physics.applyVelocity(culVeloX, culVeloY);
        physics.Step();
    }
}
