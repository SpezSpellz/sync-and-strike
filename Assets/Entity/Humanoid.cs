using UnityEngine;

public class Humanoid : MonoBehaviour
{

    private BaseMove current_move = null;

    void Start()
    {
        current_move = Moves.Instance.getWalkMove();
    }

    void Update()
    {
        if (current_move != null)
        {
            current_move.update(this);
            if(current_move.isEnded())
                current_move = Moves.Instance.getWalkMove();
        }
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
            if(this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getSlashMove();
        }
        if (Input.GetKey(KeyCode.R))
        {
            if (this.current_move.getName() == "walk")
                this.current_move = Moves.Instance.getVerticalSlashMove();
        }
        physics.applyVelocity(culVeloX, culVeloY);
    }
}
