using System;
using UnityEngine;

public class Physics : MonoBehaviour
{
    const float FLOOR = -0.3f;
    const float CEILING = 50;
    const float GRAVITY = 9.8f;
    private float velocityX = 0;
    private float velocityY = 0;
    private bool onGround = true;

    public void applyVelocity(float velocityX, float velocityY)
    {
        this.velocityX += velocityX;
        this.velocityY += velocityY;
    }

    public void Step()
    {
        var dt = 0.016f;
        velocityY -= GRAVITY * dt;
        this.velocityX /= 10;
        var newX = this.transform.position.x + velocityX * dt;
        var newY = this.transform.position.y + velocityY * dt;
        this.onGround = false;
        if(newY < FLOOR)
        {
            this.onGround = true;
            this.velocityY = 0;
            newY = FLOOR;
        }
        if (newY > CEILING)
        {
            this.velocityY = 0;
            newY = CEILING;
        }
        this.transform.position = new Vector3(newX, newY, 0);
    }

    public bool isOnGround()
    {
        return this.onGround;
    }
}
