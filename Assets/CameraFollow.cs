using UnityEngine;

public class CameraFollow : MonoBehaviour
{

    void Update()
    {
        int count = 0;
        Vector3 pos_sum = Vector3.zero;
        TurnManager.Instance.ForEachPlayer(entity =>
        {
            pos_sum += entity.transform.position;
            ++count;
        });
        if (count <= 0)
            return;
        pos_sum = pos_sum / count;
        transform.position = new Vector3(pos_sum.x, pos_sum.y, -10);
    }
}
