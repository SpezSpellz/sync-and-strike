using UnityEngine;

public class Player : Humanoid
{
    public override void Step()
    {
        base.Step();
        this.veloX /= 10;
        bool onGround = this.isOnGround();
        if (onGround)
            this.veloY = 0;
        if (Input.GetKey(KeyCode.Space) && onGround)
            this.veloY += 0.128f;
        if (Input.GetKey(KeyCode.A))
        {
            this.GetComponent<SpriteRenderer>().flipX = true;
            this.veloX -= 0.128f;
        }
        if (Input.GetKey(KeyCode.D))
        {
            this.GetComponent<SpriteRenderer>().flipX = false;
            this.veloX += 0.128f;
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
    }
}
