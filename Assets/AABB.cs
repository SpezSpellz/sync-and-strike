using System;
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
        => RayIntersectBounds(minX, minY, maxX, maxY, begX, begY, endX, endY);

    /// <summary>
    /// The ray/AABB test against EXPLICIT bounds.
    ///
    /// Split out of <see cref="rayIntersect"/> so <see cref="sweep"/> can test the Minkowski-expanded
    /// box without materialising it. The old form built a throwaway AABB per pair test via
    /// <see cref="expand"/>, and sweep is the innermost loop of the physics step - it runs once per
    /// collider pair per sweep pass, so that was the single largest source of per-frame garbage.
    /// The arithmetic is byte-for-byte the same as before.
    /// </summary>
    private static RayHit RayIntersectBounds(float minX, float minY, float maxX, float maxY,
                                             float begX, float begY, float endX, float endY)
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
        // Minkowski-expand the other box by THIS box's half-extents and ray-cast from this box's
        // centre. Computed inline instead of building an expanded AABB, so a sweep pair allocates
        // nothing. Bounds are identical to expand(this.getWidth(), this.getHeight()).
        float halfW = (maxX - minX) * 0.5f;
        float halfH = (maxY - minY) * 0.5f;
        float cx = (minX + maxX) * 0.5f;
        float cy = (minY + maxY) * 0.5f;
        var inter = RayIntersectBounds(
            other.minX - halfW, other.minY - halfH,
            other.maxX + halfW, other.maxY + halfH,
            cx, cy, cx + veloX, cy + veloY);
        if (!float.IsNaN(inter.Distance))
            return inter;
        return new RayHit { Distance = float.PositiveInfinity };
    }

    public bool intersectWith(AABB other)
    {
        return (this.minX <= other.maxX && this.maxX >= other.minX) &&
         (this.minY <= other.maxY && this.maxY >= other.minY);
    }
}