using UnityEngine;

public class CameraFollow : MonoBehaviour
{

    void Update()
    {
        int count = 0;
        Vector3 pos_sum = Vector3.zero;
        foreach(Entity entity in World.Instance.getEntities())
        {
            if(entity is not Humanoid)
                continue;
            pos_sum += entity.transform.position;
            ++count;
        }
        pos_sum = pos_sum / count;
        transform.position = new Vector3(pos_sum.x, pos_sum.y, -10);
    }
}
