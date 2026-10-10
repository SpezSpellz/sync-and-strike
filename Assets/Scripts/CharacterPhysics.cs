using UnityEngine;

public class CharacterPhysics : PhysicsCollider
{
    private CharacterData characterData;
    private float veloX = 0.0f;
    private float veloY = 0.0f;
    private bool isPreview = false;
    public bool IsGrounded { get; private set; }

    private Arena _arena;

    /// <summary>
    /// The arena this collider belongs to.
    ///
    /// Resolved lazily rather than in Initialize/Start, because CharacterController.Update() runs
    /// BEFORE Start() on a freshly created fighter. Caching the arena in Start left it null here, so
    /// this fell back to PhysicsManager.Instance, which in a multi-arena training scene is some other
    /// arena's manager. The symptoms were a NullReferenceException on the very first Update and, worse,
    /// the fighter registering itself with the wrong TurnManager so its arena never saw it.
    /// </summary>
    private Arena ArenaFor(Component c)
    {
        if (_arena == null) _arena = GetComponentInParent<Arena>();
        return _arena;
    }

    private PhysicsManager PhysManager()
    {
        var arena = ArenaFor(this);
        return arena != null ? arena.PhysicsManager : PhysicsManager.Instance;
    }

    private PreviewPhysicsManager PPManager()
    {
        var arena = ArenaFor(this);
        return arena != null ? arena.PreviewPhysicsManager : PreviewPhysicsManager.Instance;
    }

    private IndexSet<PhysicsCollider> RegisteredObjects()
    {
        if (isPreview)
        {
            var pm = PPManager();
            return pm != null ? pm.GetRegisteredObjects() : new IndexSet<PhysicsCollider>();
        }
        var phys = PhysManager();
        return phys != null ? phys.GetRegisteredObjects() : new IndexSet<PhysicsCollider>();
    }
    /// <summary>Number of simulation frames the character has been standing on the ground.</summary>
    public int GroundedFrames { get; private set; } = 0;

    /// <summary>True when the body is pressed against a wall or the ceiling this frame.</summary>
    public bool TouchingWall { get; private set; } = false;

    /// <summary>
    /// A much lower friction than normal walking, so a grounded
    /// victim in hitstun skids to a stop quickly. Set while in hitstun.
    /// </summary>
    public bool InHurtStun { get; set; } = false;

    public void Initialize(CharacterData characterData, bool isPreview = false)
    {
        this.characterData = characterData;
        this.isPreview = isPreview;
        ArenaFor(this);
    }

    public override void Start()
    {
        ArenaFor(this);
        base.Start();
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Vector2 direction = FacingDirection();
        veloX += impulse.x * direction.x;
        veloY += impulse.y;
    }

    public void AddVelocity(Vector2 velocity)
    {
        veloX += velocity.x;
        veloY += velocity.y;
    }

    /// <summary>
    /// Pushback is not a velocity impulse: the value is queued and re-applied along the axis
    /// towards the opponent, which keeps pushing the two fighters apart for as long as the value
    /// is outstanding. We resolve the axis at the moment of the hit and consume the result on the
    /// next integration, so a pushback survives hitlag.
    /// </summary>
    public void AddPushback(float magnitude, float directionX)
    {
        pendingPushback += magnitude * directionX;
    }

    /// <summary>Wipes outstanding pushback. Used by burst, which cancels the separation entirely.</summary>
    public void ClearPushback()
    {
        pendingPushback = 0f;
    }

    private float pendingPushback;

    /// <summary>
    /// This fighter's collision box, REFRESHED IN PLACE and returned.
    ///
    /// A fresh AABB used to be allocated on every call, and the physics step calls this once per
    /// collider pair per sweep pass (plus DetectGround and UpdateWallContact) - tens of throwaway
    /// objects per fighter per simulation frame, which showed up as ~1 ms/turn of fighter Step time
    /// in the training profiler. Reusing one instance is safe because every caller consumes the box
    /// immediately and each collider owns its own instance, so two live references never alias the
    /// same buffer. This is a pure allocation change: the bounds computed are identical.
    /// </summary>
    private readonly AABB _box = new AABB(0f, 0f, 0f, 0f);

    public override AABB getBoundingBox()
    {
        // characterData is null until Initialize() runs, which happens from CharacterController.Start().
        // Unity runs Update BEFORE Start on a newly created object, so a physics body can be stepped in
        // that first frame. Every caller already null-checks the returned AABB, so bailing out here is
        // safe and turns a hard crash into "no collider this frame".
        if (characterData == null) return null;

        float halfW = characterData.width * 0.5f;
        float halfH = characterData.height * 0.5f;
        Vector3 p = transform.position;
        _box.minX = p.x - halfW;
        _box.minY = p.y - halfH;
        _box.maxX = p.x + halfW;
        _box.maxY = p.y + halfH;
        return _box;
    }

    /// <summary>
    /// Gravity is a per-object attribute so projectiles and hovering
    /// moves can opt out. Flying moves simply leave gravity off in their frame data.
    /// </summary>
    public bool GravityEnabled { get; set; } = true;

    public override bool hasGravity()
    {
        return GravityEnabled;
    }

    public override Vector2 getPosition()
    {
        return new Vector2(transform.position.x, transform.position.y);
    }

    public override Vector2 getVelocity()
    {
        return new Vector2(veloX, veloY);
    }

    public override void setPosition(float x, float y)
    {
        transform.position = new Vector3(x, y, transform.position.z);
    }

    public override void setVelocity(float x, float y)
    {
        veloX = x;
        veloY = y;
    }

    /// <summary>
    /// Allies (player + companion) never push each other apart, and neither does a fighter that
    /// is currently ignoring body collision.
    ///
    /// A hitbox exports <c>disable_collision</c> with a default of TRUE, and entering a hurt
    /// state from such a hit clears this flag. So two characters in hitstun pass straight through
    /// each other; the separation you see comes from the mutual pushback, not from their bodies.
    /// The flag is restored to true on every state exit.
    ///
    /// We never had this field, so fighters stayed body-blocked for the whole of hitstun.
    /// </summary>
    public bool CollidingWithOpponent { get; set; } = true;

    /// <summary>
    /// Allies (player + companion) never push each other apart.
    /// Static geometry always collides.
    /// </summary>
    public override bool CollidesWith(PhysicsCollider other)
    {
        if (other is StaticCollider) return true;
        if (other is CharacterPhysics otherCharacter)
        {
            if (CombatTeamUtility.AreAllies(characterData.team, otherCharacter.characterData.team)) return false;
            if (!CollidingWithOpponent) return false;
            return true;
        }
        return true;
    }

    public override void Step()
    {
        if (isPreview)
        {
            PPManager().StepFor(this);
        }
        else
        {
            PhysManager().StepFor(this);
        }

        // Apply outstanding pushback on the tick it comes due, before integrating.
        if (pendingPushback != 0f)
        {
            veloX += pendingPushback;
            pendingPushback = 0f;
        }

        DetectGround();
        UpdateWallContact();
        GroundedFrames = IsGrounded ? GroundedFrames + 1 : 0;
    }

    /// <summary>
    /// Applies friction, gravity and (optionally) the speed caps for the current tick. The values
    /// are passed in explicitly because states need differing numbers (half gravity in
    /// HurtAerial, 0.05 friction in HurtGrounded, escalating gravity in WallSlam) rather than one
    /// global set.
    /// </summary>
    /// <param name="limitSpeed">
    /// False skips the speed caps entirely, giving the unlimited force pass.
    ///
    /// This is the single most important physics distinction in the model: the limited pass
    /// clamps to MAX_GROUND_SPEED / MAX_AIR_SPEED, and the unlimited one does not. Every hurt
    /// state uses the unlimited pass while WallSlam uses the limited one.
    ///
    /// Previously every state went through the clamped path, so a launcher was silently truncated:
    /// <c>vertical_slash</c> applies 0.3 units of upward knockback but MAX_AIR_SPEED is 0.24, so a
    /// fifth of every launch was deleted on the very first frame. That is also why
    /// MAX_GROUND_SPEED had been inflated to 25px to keep dashes looking right.
    /// </param>
    public void ApplyForces(float frictionRatio, float gravity, bool grounded, bool limitSpeed = true)
    {
        ApplyCustomForces(frictionRatio, gravity, grounded, PhysicsConstants.MAX_FALL_SPEED, limitSpeed);
    }

    public void ApplyCustomForces(float frictionRatio, float gravity, bool grounded, float fallSpeedCap,
        bool limitSpeed = true)
    {
        veloX *= (1f - frictionRatio);

        // Gravity plus a downward-only fall speed cap. The cap is part of
        // the gravity call, not the speed-limit pass, so it stays in force during knockback.
        veloY -= gravity;
        if (veloY < -fallSpeedCap) veloY = -fallSpeedCap;

        if (grounded)
        {
            if (limitSpeed)
                veloX = Mathf.Clamp(veloX, -PhysicsConstants.MAX_GROUND_SPEED, PhysicsConstants.MAX_GROUND_SPEED);
            if (Mathf.Abs(veloX) < PhysicsConstants.GROUND_SETTLE_SPEED) veloX = 0f;
            if (veloY < 0f) veloY = 0f;
        }
        else if (limitSpeed)
        {
            veloX = Mathf.Clamp(veloX, -PhysicsConstants.MAX_AIR_SPEED, PhysicsConstants.MAX_AIR_SPEED);
        }
    }

    public void ZeroHorizontalVelocity() { veloX = 0f; }
    public void ZeroVerticalVelocity() { veloY = 0f; }

    /// <summary>Pins a knocked-down fighter to the floor.</summary>
    public void SnapToGround()
    {
        setPosition(transform.position.x,
            FindFloorY() + characterData.height * 0.5f + PhysicsConstants.COLLIDER_SKIN);
        ZeroVerticalVelocity();
    }

    /// <summary>Pins the fighter against the stage wall for the slam duration.</summary>
    public void SnapToWall()
    {
        foreach (PhysicsCollider other in RegisteredObjects().getList())
        {
            if (!(other is StaticCollider)) continue;
            AABB box = other.getBoundingBox();
            if (box == null) continue;
            if (!(box.getWidth() < box.getHeight())) continue;   // vertical walls only
            float y = transform.position.y;
            if (box.getCenter().x < 0f) setPosition(box.maxX + characterData.width * 0.5f, y);
            else setPosition(box.minX - characterData.width * 0.5f, y);
            ZeroHorizontalVelocity();
            return;
        }
    }

    private float FindFloorY()
    {
        float best = float.NegativeInfinity;
        foreach (PhysicsCollider other in RegisteredObjects().getList())
        {
            if (!(other is StaticCollider)) continue;
            AABB box = other.getBoundingBox();
            if (box == null) continue;
            if (!(box.getWidth() >= box.getHeight())) continue;   // floors/ceilings only
            float top = box.getCenter().y < 0f ? box.maxY : box.minY;
            if (top > best && top <= transform.position.y + 0.5f) best = top;
        }
        return best == float.NegativeInfinity ? transform.position.y : best;
    }

    void OnDrawGizmos()
    {
        var data = characterData != null ? characterData : GetComponent<CharacterData>();
        if (data == null) return;

        Vector3 feet = transform.position + Vector3.down * (data.height * 0.5f);
        Gizmos.color = Color.red;
        Gizmos.DrawLine(feet, feet + Vector3.down * GroundContactTolerance);
    }
    /// <param name="blockedByStageWall">
    /// True only when a *vertical* static collider (the stage wall) blocked this fighter. The floor
    /// is deliberately excluded, and the opponent is a separate case again. touching_wall is a
    /// stage-boundary test only.
    /// </param>
    /// <param name="blockedVelocity">Velocity at the moment of impact, before the axis was zeroed.</param>
    public void SetWallContact(bool blockedByStageWall, Vector2 blockedVelocity)
    {
        TouchingWall = blockedByStageWall;
        BlockedByWall = blockedByStageWall && blockedVelocity != Vector2.zero;
        // Only meaningful when a stage wall actually stopped us. Previously this stayed set after
        // ANY collision, including being blocked by the OPPONENT, so HurtAerial's BOUNCE_FACTOR
        // -0.85 rebound (and even a WallSlam) fired on hitting the other fighter.
        BlockedVelocity = blockedByStageWall ? blockedVelocity : Vector2.zero;
    }

    /// <summary>Velocity the fighter had when it struck a wall. Collision resolution zeroes
    /// the blocked axis, so HurtAerial needs this to apply the wall rebound.</summary>
    public Vector2 BlockedVelocity { get; private set; } = Vector2.zero;

    /// <summary>True only when the stage wall itself blocked this fighter.</summary>
    public bool BlockedByWall { get; private set; } = false;

    /// <summary>
    /// <c>touching_wall</c> is a stage-boundary test, not a general overlap test: it is set
    /// only when the collision box reaches the left or right edge of the stage. It is never set by
    /// the floor and never by the opponent.
    ///
    /// This used to probe every StaticCollider, which wrongly included the floor. It only avoided
    /// a permanent false positive by accident: PhysicsManager rests characters COLLIDER_SKIN clear
    /// of the floor and AABB.intersectWith uses inclusive comparisons, so that 0.001 gap was the
    /// only thing keeping TouchingWall false while standing on the ground. Any change that closed
    /// that gap would have silently zeroed ALL grounded knockback, because both HitReaction and
    /// HurtGroundedState drop horizontal push whenever TouchingWall is set.
    ///
    /// Only vertical static colliders count, matching the convention SnapToWall already uses.
    /// </summary>
    public void UpdateWallContact()
    {
        TouchingWall = IsOverlappingStageWall();
    }

    private bool IsOverlappingStageWall()
    {
        AABB self = getBoundingBox();
        if (self == null) return false;
        IndexSet<PhysicsCollider> objects = RegisteredObjects();
        if (objects == null) return false;
        foreach (PhysicsCollider other in objects.getList())
        {
            if (other == this) continue;
            if (!(other is StaticCollider)) continue;
            AABB box = other.getBoundingBox();
            if (box == null) continue;
            if (!(box.getWidth() < box.getHeight())) continue;   // vertical walls only
            if (Overlap(self, box)) return true;
        }
        return false;
    }

    /// <summary>
    /// Strict overlap. AABB.intersectWith uses inclusive comparisons, so two boxes that are
    /// exactly flush count as intersecting, which is what made the floor read as a "wall".
    /// </summary>
    private static bool Overlap(AABB a, AABB b)
    {
        return a.minX < b.maxX && a.maxX > b.minX
            && a.minY < b.maxY && a.maxY > b.minY;
    }

    public void DetectGround()
    {
        IsGrounded = veloY <= 0f && IsTouchingGround();
    }

    /// <summary>Collision skin plus a small allowance for floating-point rounding.</summary>
    private const float GroundContactTolerance = PhysicsConstants.COLLIDER_SKIN + 0.001f;

    /// <summary>
    /// Uses the same collision boxes as movement, both in the arena and in previews. A surface
    /// counts only when it is horizontally under the fighter and its top is at the fighter's feet.
    /// The small tolerance includes the gap left by PhysicsManager's collision skin.
    /// </summary>
    private bool IsTouchingGround()
    {
        AABB self = getBoundingBox();
        IndexSet<PhysicsCollider> objects = RegisteredObjects();
        if (self == null || objects == null) return false;

        float feetY = self.minY;

        foreach (PhysicsCollider other in objects.getList())
        {
            if (other == this || !CollidesWith(other)) continue;
            AABB box = other.getBoundingBox();
            if (box == null) continue;

            if (self.maxX <= box.minX || self.minX >= box.maxX) continue;

            float gap = feetY - box.maxY;
            if (Mathf.Abs(gap) <= GroundContactTolerance) return true;
        }
        return false;
    }

    public Vector2 FacingDirection()
    {
        return transform.localScale.x > 0 ? Vector2.right : Vector2.left;
    }
}
