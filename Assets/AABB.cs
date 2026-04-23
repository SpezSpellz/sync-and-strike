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
    public float rayIntersect(float begX, float begY, float endX, float endY)
    {
        float xdiff = endX - begX;
        float ydiff = endY - begY;
        float dist = Mathf.Sqrt(xdiff * xdiff + ydiff * ydiff);

        float xn = xdiff / dist;
        float yn = ydiff / dist;

        float tx1 = (this.minX - begX)/xn;
        float tx2 = (this.maxX - begX)/xn;

        float tmin = Mathf.Min(tx1, tx2);
        float tmax = Mathf.Max(tx1, tx2);

        float ty1 = (this.minY - begY)/yn;
        float ty2 = (this.maxY - begY)/yn;

        tmin = Mathf.Max(tmin, Mathf.Min(ty1, ty2));
        tmax = Mathf.Min(tmax, Mathf.Max(ty1, ty2));

        if (tmax < 0 || tmin > tmax)
            return float.NaN;
        float ret = (tmin < 0) ? tmax : tmin;
        if (ret > dist)
            return float.NaN;
        return ret;
    }

    public float sweep(AABB other, float veloX, float veloY)
    {
        var center = this.getCenter();
        float inter = other.expand(this.getWidth(), this.getHeight()).rayIntersect(center.x, center.y, center.x + veloX, center.y + veloY);
        if (!float.IsNaN(inter))
            return inter;
        return float.PositiveInfinity;
    }
}