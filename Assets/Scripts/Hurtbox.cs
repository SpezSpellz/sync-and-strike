public class HurtBox : AABB
{
    public CharacterController owner { get; private set; }
    public HurtBox(CharacterController owner, float minX, float minY, float maxX, float maxY) : base(minX, minY, maxX, maxY)
    {
        this.owner = owner;
    }
}