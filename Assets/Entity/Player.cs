public class Player : Humanoid
{
    public override void Step()
    {
        base.Step();
        this.veloX /= 10;
    }

    public void setMove(BaseMove move)
    {
        this.current_move = move;
    }
}
