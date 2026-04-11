using System;
using Unity.Mathematics;
using UnityEngine;

public class MovementTest : MonoBehaviour
{
    private double cur = 0;
    const UInt32 SPEED = 5;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        cur += Time.deltaTime * SPEED;
        transform.position = new Vector3(4.53f, (float)math.sin(this.cur));
    }
}
