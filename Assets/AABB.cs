using UnityEngine;

public class AABB
{
    public float minX;
    public float minY;
    public float maxX;
    public float maxY;

    public AABB(float minX, float minY, float maxX, float maxY) { 
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
    }

    public AABB expand(float x, float y) {
        return new AABB(
            this.minX - x/2,
            this.minY - y/2,
            this.maxX + x/2,
            this.maxY + y/2
        );
    }

    public Vector2 getCenter()
    {
        return new Vector2((this.minX + this.maxX) / 2, (this.minY + this.maxY) / 2);
    }

    public float getWidth()
    {
        return this.maxX - this.minX;
    }

    public float getHeight()
    {
        return this.maxY - this.minY;
    }
    public struct RayHit
    {
        public float Distance;
        public bool HitVertical;
    }
    public RayHit rayIntersect(float begX, float begY, float endX, float endY)
    {
        float xdiff = endX - begX;
        float ydiff = endY - begY;
        float dist = Mathf.Sqrt(xdiff * xdiff + ydiff * ydiff);

        float xn = xdiff / dist;
        float yn = ydiff / dist;

        // Entry/Exit times for X
        float tx1 = (minX - begX) / xn;
        float tx2 = (maxX - begX) / xn;
        float tminX = Mathf.Min(tx1, tx2);
        float tmaxX = Mathf.Max(tx1, tx2);

        // Entry/Exit times for Y
        float ty1 = (minY - begY) / yn;
        float ty2 = (maxY - begY) / yn;
        float tminY = Mathf.Min(ty1, ty2);
        float tmaxY = Mathf.Max(ty1, ty2);

        // Final entry/exit
        float tmin = Mathf.Max(tminX, tminY);
        float tmax = Mathf.Min(tmaxX, tmaxY);

        if (tmax < 0 || tmin > tmax || tmin > dist)
            return new RayHit { Distance = float.NaN };

        // Determine the side:
        // If tmin came from the Y-axis calculation, we hit a Top or Bottom edge.
        // If tmin came from the X-axis calculation, we hit a Left or Right edge.
        bool hitVertical = tminX >= tminY;

        return new RayHit { Distance = tmin, HitVertical = hitVertical };
    }

    public RayHit sweep(AABB other, float veloX, float veloY)
    {
        var center = this.getCenter();
        var inter = other.expand(this.getWidth(), this.getHeight()).rayIntersect(center.x, center.y, center.x + veloX, center.y + veloY);
        if (!float.IsNaN(inter.Distance))
            return inter;
        return new RayHit { Distance = float.PositiveInfinity };
    }
}