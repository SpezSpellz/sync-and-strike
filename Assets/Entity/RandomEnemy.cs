using UnityEngine;
using Random = System.Random;

public class RandomEnemy : Humanoid
{

    private Random random = new Random();
    public override void Step()
    {
        base.Step();
        this.veloX /= 10;
    }
}
