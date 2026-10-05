using System;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(CharacterAnimation))]
[RequireComponent(typeof(CharacterPhysics))]
[RequireComponent(typeof(CharacterData))]
public class CharacterController : MonoBehaviour
{
    const string CONTINUE = "continue";

    private CharacterAnimation anim;
    private CharacterPhysics physics;
    private CharacterData characterData;
    private PreviewController previewController;
    private FighterStateMachine stateMachine;

    public int id { get; private set; }
    public CharacterAnimation Anim => anim;
    public CharacterPhysics Physics => physics;
    public CharacterData Data => characterData;
    public FighterStateMachine States => stateMachine;
    public AnimationData[] animations => characterData.animations;

    /// <summary>
    /// Running combat counters for this fighter. Written from HitReaction when a hit connects and
    /// from ExecuteMove when an attack is committed. Read by the training runner to divide the shared
    /// 2v1 win bonus by damage contribution, and by the telemetry CSV for the whiff rate.
    /// A plain field rather than a property so HitReaction can mutate it in place without a setter.
    /// </summary>
    public readonly CombatStats CombatStats = new CombatStats();
    public CombatState State => stateMachine != null ? stateMachine.CurrentId : CombatState.Idle;

    public bool IsGrounded => physics != null && physics.IsGrounded;
    public bool IsBusy => stateMachine != null && !stateMachine.CurrentOrIdle().CanAct;
    public bool IsKnockedBack => stateMachine != null && stateMachine.CurrentOrIdle().IsHurt;
    public bool IsBlocking => stateMachine != null && stateMachine.CurrentOrIdle().IsBlocking;
    public int HitstunRemaining => stateMachine != null ? stateMachine.CurrentOrIdle().HitstunRemaining : 0;
    public int HitlagRemaining { get; private set; } = 0;
    public int ComboCount { get; private set; } = 0;

    public string SelectedMove { get; private set; } = CONTINUE;
    public string LastSubmittedMove { get; protected set; } = CONTINUE;
    public string LastMoveUsed { get; private set; } = "";

    public float JumpDirection { get; private set; } = Mathf.PI * 0.5f;
    public float JumpPower { get; private set; } = 1f;
    public float PreviewJumpDirection { get; private set; } = Mathf.PI * 0.5f;
    public float PreviewJumpPower { get; private set; } = 1f;
    public float KnockbackIndirectionDirection { get; private set; } = 0f;
    public float KnockbackIndirectionPower { get; private set; } = 0f;
    public float PreviewKnockbackIndirectionDirection { get; private set; } = 0f;
    public float PreviewKnockbackIndirectionPower { get; private set; } = 0f;
    public Vector3 PreviewScale { get; private set; }
    public Transform TargetPosition { get; protected set; }
    public bool IsPreviewReady => previewController != null && characterData != null && anim != null;

    public CombatTeam Team => characterData.team;
    public bool TouchingWall => physics != null && physics.TouchingWall;

    // --- Body collision against the opponent ---
    /// <summary>
    /// Whether this fighter's body blocks the opponent's, defaulting to true. Cleared when a hit
    /// that disables collision puts this fighter into a hurt state, and restored on every state
    /// exit.
    /// </summary>
    private bool collidingWithOpponent = true;

    /// <summary>
    /// The hitlag length the last hit set. Kept so the collision grace window can be measured as
    /// ticks elapsed since the hit.
    /// </summary>
    public int HitlagApplied { get; private set; } = 0;

    /// <summary>
    /// The effective body-collision state.
    ///
    /// The second clause is the important one: even when a hit has turned body collision off, the
    /// two fighters stay body-blocked for HITLAG_COLLISION_TICKS so they do not visibly pop apart
    /// on the exact frame of the hit.
    /// </summary>
    public bool IsCollidingWithOpponent
    {
        get
        {
            bool inHurtGrace = stateMachine != null
                && stateMachine.Current is HurtStateBase
                && HitlagApplied - HitlagRemaining < PhysicsConstants.HITLAG_COLLISION_TICKS;
            return collidingWithOpponent || inHurtGrace;
        }
    }

    /// <summary>Sets the raw flag and pushes it into the physics layer that does the sweeping.</summary>
    public void SetCollidingWithOpponent(bool value)
    {
        collidingWithOpponent = value;
        if (physics != null) physics.CollidingWithOpponent = value;
    }

    /// <summary>
    /// Body collision is restored on every state exit, so a hit that cleared it only
    /// suppresses collision for the duration of that hurt state.
    /// </summary>
    public void RestoreOpponentCollision() => SetCollidingWithOpponent(true);
    public bool WasGroundedWhenHit { get; private set; } = true;

    public int WallSlams { get; private set; } = 0;
    public int GroundedHitsTaken { get; private set; } = 0;

    // --- Data the entered hurt state needs (set by HitReaction before switching) ---
    public PendingHit Pending { get; set; }
    public int PendingHitstun => Pending.hitstun;
    public bool PendingKnockdown => Pending.knockdown;
    public bool PendingHardKnockdown => Pending.hardKnockdown;
    public bool PendingWallSlam => Pending.wallSlam;
    public bool PendingAirGroundBounce => Pending.airGroundBounce;
    public bool PendingGroundBounce => Pending.groundBounce;
    public bool PendingLaunched => Pending.launched;
    public int PendingMinimumGroundedFrames => Pending.minimumGroundedFrames;
    public bool PendingKnockdownExtendsHitstun => Pending.knockdownExtendsHitstun;

    public struct PendingHit
    {
        public int hitstun;
        public bool knockdown;
        public bool hardKnockdown;
        public bool wallSlam;
        public bool airGroundBounce;
        public bool groundBounce;
        public bool launched;
        public int minimumGroundedFrames;
        public bool knockdownExtendsHitstun;
    }

    private Action onMoveComplete;
    private bool actionReported;

    /// <summary>
    /// Whether this fighter has already reported its action complete for the current turn.
    /// TurnManager uses this to name whoever stalled a turn when the frame cap trips.
    /// </summary>
    public bool HasReportedAction => actionReported;

    // Direction in radians
    public void setJumpInfo(float power, float direction, bool previewOnly = true)
    {
        float clampedPower = Mathf.Clamp(power, 0.5f, 1f);
        float clampedDirection = Mathf.Clamp(direction, 0.5235987755982988f, 2.6179938779914944f);
        if (!previewOnly)
        {
            JumpPower = clampedPower;
            JumpDirection = clampedDirection;
        }
        PreviewJumpPower = clampedPower;
        PreviewJumpDirection = clampedDirection;
    }

    // Direction in radians
    public void setKnockbackInfo(float power, float direction, bool previewOnly = true)
    {
        float clampedPower = Mathf.Clamp(power, 0f, 1f);
        float clampedDirection = Mathf.Clamp(direction, 0f, 6.283185307179586f);
        if (!previewOnly)
        {
            KnockbackIndirectionPower = clampedPower;
            KnockbackIndirectionDirection = clampedDirection;
        }
        PreviewKnockbackIndirectionPower = clampedPower;
        PreviewKnockbackIndirectionDirection = clampedDirection;
    }

    public Vector2 KnockbackIndirectionVector =>
        new Vector2(Mathf.Cos(KnockbackIndirectionDirection), Mathf.Sin(KnockbackIndirectionDirection));

    public Vector2 FacingVector => physics != null ? physics.FacingDirection() : Vector2.right;

    /// <summary>
    /// The other fighter, used for the few places that need to know what the opponent is doing.
    /// Null when this fighter is alone, which the preview sandbox is.
    /// </summary>
    public CharacterController Opponent
    {
        get
        {
            var manager = Turn;
            if (manager == null) return null;
            var all = manager.GetAllPlayers();
            for (int i = 0; i < all.Count; i++)
            {
                var other = all[i];
                if (other != null && other != this && !other.IsDead()) return other;
            }
            return null;
        }
    }

    /// <summary>
    /// The force is the jump curve of the input direction (cubic, see
    /// <see cref="PhysicsConstants.JumpCurve"/>), so a half-power hop is much shorter than a full
    /// hop, then scales Y by 1.5 and, unless this is a combo escape, by the global 0.85 modifier.
    /// </summary>
    /// <param name="jumpSpeed">Impulse in pixels, before the base speed is added.</param>
    public Vector2 JumpVelocity(float power, float direction, float jumpSpeed)
    {
        float t = Mathf.Clamp01((power - 0.5f) / 0.5f);
        float strength = PhysicsConstants.JumpCurve(t);
        Vector2 dir = new Vector2(Mathf.Cos(direction), Mathf.Sin(direction));
        float force = PhysicsConstants.FromPixels(jumpSpeed + PhysicsConstants.BASE_JUMP_SPEED);

        // Previously the global reduction was applied unconditionally. Skipping it during a combo
        // escape is what makes escaping a juggle feel like it clears the opponent.
        var opponent = Opponent;
        bool comboEscape = PhysicsConstants.IsComboEscape(ComboCount, opponent != null && opponent.physics.IsGrounded);
        dir.y *= PhysicsConstants.JUMP_Y_MODIFIER * (comboEscape ? 1f : PhysicsConstants.GLOBAL_JUMP_MODIFIER);
        return dir * strength * force;
    }

    /// <summary>
    /// Scales away 75% of the horizontal speed carried into the jump, which is why jumping
    /// forward out of a dash travels much less far than the dash did. Vertical speed is discarded
    /// outright, so a jump taken during a launch does not stack leftover height on top of the new
    /// impulse.
    /// </summary>
    public void ApplyJumpMomentum()
    {
        var vel = physics.getVelocity();
        physics.setVelocity(vel.x * PhysicsConstants.JUMP_X_SPEED_PRESERVED, 0f);
    }

    /// <summary>
    /// Nudges the victim up on the frame they are launched.
    /// </summary>
    public void ApplyLaunchLift()
    {
        var pos = physics.getPosition();
        physics.setPosition(pos.x, pos.y + PhysicsConstants.LAUNCH_LIFT);
    }

    /// <summary>
    /// Adds knockback to the current velocity.
    ///
    /// Deliberately unclamped: MAX_KNOCKBACK caps the hitbox's own knockback scalar, and
    /// there is no re-clamp after DI is added, so a full-combo DI legitimately pushes a victim
    /// well past it. Clamping here instead silently ate the DI.
    /// </summary>
    public void ApplyKnockback(Vector2 knockback)
    {
        physics.AddVelocity(knockback);
    }

    /// <summary>Applies an immediate velocity change.</summary>
    public void ApplyVelocity(Vector2 velocity) => physics.AddVelocity(velocity);

    /// <summary>
    /// A queued mutual separation impulse, consumed on the next integration. Not an immediate
    /// velocity change, so it survives hitlag.
    /// </summary>
    public void AddPushback(float magnitude)
    {
        // Direction is away from the opponent, not along facing.
        float dir = TargetPosition != null
            ? Mathf.Sign(TargetPosition.position.x - transform.position.x)
            : FacingVector.x;
        if (Mathf.Approximately(dir, 0f)) dir = FacingVector.x;
        physics.AddPushback(magnitude, dir);
    }

    public struct SaveData
    {
        public float health;
        public Vector2 pos;
        public Vector3 localScale;
        public Vector2 velocity;
    }

    private void Awake()
    {
        anim = GetComponent<CharacterAnimation>();
        physics = GetComponent<CharacterPhysics>();
        characterData = GetComponent<CharacterData>();
        physics.Initialize(characterData);
        PreviewScale = transform.localScale;
        previewController = GetComponentInChildren<PreviewController>();

        stateMachine = new FighterStateMachine(this);
        anim.OnFrameEvent = OnFrameEvent;
        _arena = GetComponentInParent<Arena>();
    }

    private Arena _arena;

    /// <summary>
    /// The arena this fighter belongs to, resolved lazily from the ancestor chain. Null in a
    /// single-arena scene, in which case every manager falls back to its static Instance.
    ///
    /// Resolved lazily on every use rather than cached in Awake: Unity runs Update() before Start()
    /// on a newly created fighter, so anything that asks during the first frame would otherwise get
    /// null and silently fall through to another arena's static manager.
    /// </summary>
    protected Arena ArenaFor(Component c)
    {
        if (c == null) return null;
        if (_arena == null) _arena = c.GetComponentInParent<Arena>();
        return _arena;
    }

    private TurnManager Turn => ArenaFor(this) != null ? ArenaFor(this).TurnManager : TurnManager.Instance;
    private HitboxManager Hitbox => ArenaFor(this) != null ? ArenaFor(this).HitboxManager : HitboxManager.Instance;

    private void Update()
    {
        physics.DetectGround();
        physics.UpdateWallContact();
    }

    public virtual void Start()
    {
        previewController?.Initialize(characterData);
        anim.Initialize(characterData.animations);
        stateMachine.Change(new IdleState(this));
        this.id = Turn.RegisterPlayer(this);
    }

    public void PlayIdleAnimation() => anim.PlayIdle();

    public void OnStateChanged(FighterState state) { }

    // --- Damage / combo ----------------------------------------------------

    public void Damage(float damage)
    {
        if (damage <= 0f) return;
        characterData.health = Mathf.Max(0, characterData.health - damage);
        if (characterData.health <= 0f) ComboCount = 0;
    }

    /// <summary>Alias used by the hit resolver.</summary>
    public void ApplyDamage(float damage) => Damage(damage);

    /// <summary>
    /// A hitbox with scale_combo off only starts a combo
    /// (combo_count <= 0) and does not extend an existing one, but visible_combo_count still
    /// rises. hitstun_decay_combo_count drives the mutual pushback and is incremented alongside.
    /// </summary>
    public void IncrementCombo(bool scaleCombo = true, int amount = 1)
    {
        if (scaleCombo || ComboCount == 0)
        {
            ComboCount += amount;
            HitstunDecayComboCount++;
        }
    }

    /// <summary>Hits landed this combo, which scales how hard each connect pushes both fighters apart.</summary>
    public int HitstunDecayComboCount { get; private set; } = 0;

    private readonly Dictionary<string, int> moveUses = new Dictionary<string, int>();

    /// <summary>
    /// How many times each move has landed this combo. Drives
    /// the same-move knockback increase and the same-move hitstun decrease.
    /// </summary>
    public int GetMoveUseCount(string moveId)
    {
        if (string.IsNullOrEmpty(moveId)) return 0;
        return moveUses.TryGetValue(moveId, out int uses) ? uses : 0;
    }

    public void RegisterMoveUse(string moveId)
    {
        if (string.IsNullOrEmpty(moveId)) return;
        moveUses.TryGetValue(moveId, out int uses);
        moveUses[moveId] = uses + 1;
    }

    /// <summary>
    /// Only a hit that resolves to HurtGrounded counts. The counter is never cleared on any
    /// other hit, so an airborne or launcher hit leaves the run intact; it is reset per turn and
    /// when the 7-hit limit fires.
    /// </summary>
    public void RegisterGroundedHit(bool countsAsGrounded)
    {
        if (countsAsGrounded) GroundedHitsTaken++;
    }

    /// <summary>
    /// Resets the grounded-hit counter once the limit has forced the victim airborne, so the
    /// next launch requires another full run of grounded hits.
    /// </summary>
    public void ConsumeGroundedHits() => GroundedHitsTaken = 0;

    public void RegisterWallSlam()
    {
        WallSlams++;
        // Small chip damage per slam; see PhysicsConstants.WALL_SLAM_DAMAGE_FRACTION.
        Damage(characterData.maxHealth * PhysicsConstants.WALL_SLAM_DAMAGE_FRACTION);
    }

    public bool IsDead() => characterData.health <= 0;
    public float GetHealth() => characterData.health;

    public Vector2 GetPosition() => physics.getPosition();
    public Vector2 GetVelocity() => physics.getVelocity();
    public Vector2 GetFacingVector() => FacingVector;

    public SaveData Save()
    {
        return new SaveData
        {
            pos = physics.getPosition(),
            health = characterData.health,
            localScale = transform.localScale,
            velocity = physics.getVelocity(),
        };
    }

    public virtual void Load(SaveData savedata)
    {
        physics.setPosition(savedata.pos.x, savedata.pos.y);
        physics.setVelocity(savedata.velocity.x, savedata.velocity.y);
        characterData.health = savedata.health;
        transform.localScale = savedata.localScale;
        ComboCount = 0;
        HitlagRemaining = 0;
        GroundedHitsTaken = 0;
        WallSlams = 0;
        HitstunDecayComboCount = 0;
        moveUses.Clear();
        physics.ClearPushback();
        // A match reset returns everyone to their start, so any facing preference the player set
        // no longer applies and auto-facing should resume.
        FacingChosenByPlayer = false;
        PreviewScale = transform.localScale;
        stateMachine?.Change(new IdleState(this));
        physics.DetectGround();
        HideMovePreview();
    }

    // --- Frame events ------------------------------------------------------

    private void OnFrameEvent(FrameEvent frameEvent)
    {
        switch (frameEvent.type)
        {
            case FrameEventType.SpawnHitbox:
                SpawnHitbox(frameEvent.hitboxData);
                break;
            case FrameEventType.Jump:
                // Drop 75% of the carried horizontal speed first, so a jump
                // out of a dash travels far less than the dash did.
                ApplyJumpMomentum();
                if (stateMachine.CurrentOrIdle() is JumpState jump)
                    jump.ApplyJumpForce(JumpVelocity(JumpPower, JumpDirection, anim.CurrentMove.JumpSpeed));
                break;
            case FrameEventType.Block:
                // Guarding is expressed by being in BlockState, not a flag.
                break;
        }
    }

    private void SpawnHitbox(HitboxData data)
    {
        if (data.damage == 0 && data.knockback == Vector2.zero) return;

        // Count the ATTACK here, where the hitbox is actually put into the world, rather than when
        // the move was committed. A hit only ever connects through the callback below, so this is
        // the exact denominator a whiff rate needs: every swing that was thrown, whether or not it
        // reached anybody. Counting at commit time instead would fold in moves that never got as far
        // as spawning a hitbox, which would understate the whiff rate.
        CombatStats.RecordAttack();

        float facing = FacingVector.x;
        float cx = transform.position.x + facing * data.offsetX;
        float cy = transform.position.y + data.offsetY;
        LastMoveUsed = anim.CurrentMove != null ? anim.CurrentMove.moveId : "";

        Hitbox.SubmitHitBox(new HitBox(
            this,
            target =>
            {
                if (target == null || target == this) return;
                HitReaction.Resolve(target, this, data, facing);
            },
            cx - data.width * 0.5f, cy - data.height * 0.5f,
            cx + data.width * 0.5f, cy + data.height * 0.5f));
    }

    /// <summary>
    /// Applies a hit to this fighter. Real hitboxes go through the HitBox callback; this is the
    /// same entry point exposed for tests and scripted scenarios.
    /// </summary>
    public bool ReceiveHit(CharacterController attacker, Vector2 knockback, float damage,
        HitHeight hitHeight = HitHeight.Mid)
    {
        var data = new HitboxData
        {
            damage = Mathf.RoundToInt(damage),
            knockback = knockback,
            hitHeight = hitHeight,
            groundBounceKnockbackModifier = 1f,
        };
        return HitReaction.Resolve(this, attacker, data, attacker != null ? attacker.FacingVector.x : 1f);
    }

    // --- Move selection ----------------------------------------------------

    public bool CanUseMove(AnimationData moveData)
    {
        if (moveData == null) return false;
        if (moveData.moveId == "idle") return true;
        if (stateMachine == null) return false;
        if (BlockstunRemaining > 0) return false;
        if (stateMachine.Current != null && !stateMachine.Current.CanAct) return false;
        if (moveData.onlyGrounded && !IsGrounded) return false;
        if (!moveData.usableInKnockedback && IsKnockedBack) return false;
        return true;
    }

    public void ResetMove() => SelectedMove = CONTINUE;
    public void ResetPreviewScale() => PreviewScale = transform.localScale;

    public bool TrySelectMove(string moveId)
    {
        if (moveId == CONTINUE) { SelectedMove = CONTINUE; return true; }
        if (string.IsNullOrEmpty(moveId)) { SelectedMove = CONTINUE; return false; }

        var move = characterData.GetMove(moveId);
        if (move == null) { SelectedMove = CONTINUE; return false; }
        if (!CanUseMove(move)) { SelectedMove = CONTINUE; return false; }

        SelectedMove = moveId;
        return true;
    }

    public void SelectMove(string moveId) => TrySelectMove(moveId);

    /// <summary>
    /// Orients the fighter. <paramref name="flipped"/> means facing left.
    ///
    /// PreviewScale is derived from transform.localScale here rather than tracked as an
    /// independent value. They used to be two separate writes with nothing keeping them equal, so
    /// they could drift apart: the Flip toggle reads PreviewScale while the sprite uses
    /// localScale, and the toggle would then report the opposite of what is on screen.
    ///
    /// <paramref name="previewOnly"/> flips just the preview ghost, leaving the real sprite alone.
    /// That is what the enemy's read-only mirror panel uses.
    /// </summary>
    public void Flip(bool flipped, bool previewOnly = false)
    {
        if (!previewOnly && flipped != (transform.localScale.x < 0f))
            transform.localScale = new Vector3(-transform.localScale.x, transform.localScale.y, transform.localScale.z);

        float magnitude = Mathf.Abs(transform.localScale.x);
        PreviewScale = new Vector3(
            flipped ? -magnitude : magnitude,
            transform.localScale.y,
            transform.localScale.z);
    }

    /// <summary>
    /// Whether the player chose this fighter's facing for the current turn with Flip.
    /// TurnManager clears the choice when the next planning phase begins.
    /// </summary>
    public bool FacingChosenByPlayer { get; private set; }

    /// <summary>Faces the assigned target unless the player chose a direction for this turn.</summary>
    public void FaceTarget()
    {
        if (FacingChosenByPlayer || TargetPosition == null) return;

        float deltaX = TargetPosition.position.x - transform.position.x;
        if (Mathf.Abs(deltaX) <= 0.01f) return;

        bool faceLeft = deltaX < 0f;
        if (faceLeft != IsFlipped) Flip(faceLeft);
    }

    public void ClearFacingChoice() => FacingChosenByPlayer = false;

    /// <summary>
    /// Flips and records a one-turn player choice. The AI calls <see cref="Flip"/> directly.
    /// </summary>
    public void FlipAndRememberFacing(bool flipped)
    {
        Flip(flipped);
        FacingChosenByPlayer = true;
    }

    /// <summary>True when this fighter is facing left, i.e. its sprite x-scale is negative.</summary>
    public bool IsFlipped => transform.localScale.x < 0f;

    public HurtBox getHurtBox()
    {
        return new HurtBox(
            this,
            transform.position.x - characterData.width * 0.5f,
            transform.position.y - characterData.height * 0.5f,
            transform.position.x + characterData.width * 0.5f,
            transform.position.y + characterData.height * 0.5f);
    }

    public virtual void RequestDecision() { }

    // --- Turn loop ---------------------------------------------------------

    public void AddHitlag(int frames)
    {
        HitlagRemaining = Mathf.Max(HitlagRemaining, frames);
        // Record the hitlag length alongside the countdown just set, which is what the collision
        // grace window is measured against.
        HitlagApplied = HitlagRemaining;
    }

    /// <summary>
    /// A blocking fighter gets a short blockstun during which they cannot act. Tracked
    /// separately from hitstun because a block does not interrupt the fighter into a hurt state.
    /// </summary>
    public void AddHitstun(int frames) => BlockstunRemaining = Mathf.Max(BlockstunRemaining, frames);

    public int BlockstunRemaining { get; private set; } = 0;

    /// <summary>Called by HitReaction to move into a specific state.</summary>
    public void SwitchTo(CombatState state)
    {
        stateMachine.Change(stateMachine.CreateForMove(state, anim.CurrentMove));
    }

    /// <summary>Called by a hurt state once it is done; returns to idle or landing.</summary>
    public void LeaveHurtState()
    {
        if (IsGrounded) stateMachine.Change(new LandingState(this));
        else stateMachine.Change(new IdleState(this));
    }

    /// <summary>
    /// Reports that this fighter's action for the turn is finished, WITHOUT leaving the current
    /// state.
    ///
    /// Hold actions (walk, block) use this: they satisfy the turn's completion requirement but keep
    /// acting until the turn actually resolves, at which point TurnManager force-finishes everyone.
    /// Before this split there was only CompleteMove(), which both reported and switched to idle, so
    /// a hold could never complete and the turn stalled until maxTurnFrames.
    /// </summary>
    public void ReportActionComplete()
    {
        actionReported = true;
        var cb = onMoveComplete;
        if (cb == null) return;
        onMoveComplete = null;   // report at most once per turn
        cb();
    }

    /// <summary>Ends the move's animation, returns to idle, and reports completion.</summary>
    public void CompleteMove()
    {
        stateMachine.Change(new IdleState(this));
        ReportActionComplete();
    }

    /// <summary>
    /// Runs once per turn, not per match: the combo counters, grounded-hit counter, wall slam
    /// counter and move-use table all reset between turns.
    ///
    /// Without this the launch-on-7-grounded-hits rule and the 3-wall-slam limit both leaked
    /// across turns, so by the second round every grounded hit launched.
    /// </summary>
    public void ResetComboState()
    {
        ComboCount = 0;
        HitstunDecayComboCount = 0;
        moveUses.Clear();
        GroundedHitsTaken = 0;
        WallSlams = 0;
    }

    public void Step()
    {
        // Hitlag freezes both animation and physics.
        if (HitlagRemaining > 0)
        {
            HitlagRemaining--;
            return;
        }

        // Push the grace-aware value down to the layer that does the sweeping, so the two
        // fighters stay body-blocked for the first few ticks of a hurt state even though the hit
        // already cleared the raw flag.
        physics.CollidingWithOpponent = IsCollidingWithOpponent;

        if (BlockstunRemaining > 0) BlockstunRemaining--;

        stateMachine.Step();
        physics.Step();
        Hitbox.SubmitHurtBox(getHurtBox());
    }

    public void ExecuteMove(string moveId, Action onComplete = null)
    {
        onMoveComplete = onComplete;
        actionReported = false;

        var move = moveId == CONTINUE ? characterData.GetMove("idle") : characterData.GetMove(moveId);
        if (move == null)
        {
            // Always resolve the turn even with no usable move, otherwise it never finishes.
            move = characterData.GetMove("idle");
            if (move == null) { CompleteMove(); return; }
        }

        // Idle plays through to its last frame and then completes, which resolves the turn.
        CombatState target = (moveId == CONTINUE || move.moveId == "idle")
            ? CombatState.Attack
            : ClassifyMove(move);

        stateMachine.Change(stateMachine.CreateForMove(target, move));
    }

    private static CombatState ClassifyMove(AnimationData move)
    {
        switch (move.moveId)
        {
            case "walkf": return CombatState.Walk;
            case "dash": return CombatState.Dash;
            case "jump":
            case "super_jump": return CombatState.Jump;
            case "block": return CombatState.Block;
            default:
                return move.move == MoveType.Movement ? CombatState.Walk : CombatState.Attack;
        }
    }

    /// <summary>Clears any in-flight state so a stuck fighter cannot block the next turn.</summary>
    public void ForceFinishMove()
    {
        onMoveComplete = null;
        actionReported = true;
        HitlagRemaining = 0;
        // Blockstun must be cleared here too. It is only decremented inside Step(), which runs
        // solely while Phase == Simulating, so a block landing near the end of a turn survived
        // into the planning phase. CanUseMove then rejected every move except idle, leaving
        // the player looking at an almost empty control bar.
        BlockstunRemaining = 0;
        anim.ResetToIdle();
        stateMachine.ForceExit();
        stateMachine.Change(new IdleState(this));
    }

    public virtual void ResetDecisionMetrics() { }

    // --- Preview -----------------------------------------------------------

    public void ShowMovePreview(AnimationData moveData)
    {
        previewController?.StopPreview();
        previewController?.StartPreview(moveData, this);
    }

    public void ResumePreview()
    {
        previewController?.StopPreview();
        if (anim.HasActiveMove)
        {
            previewController?.StartPreview(anim.CurrentMove, this, anim.CurrentFrameIndex, true);
            return;
        }
        foreach (var move in characterData.animations)
        {
            if (move.moveId == "idle")
            {
                previewController?.StartPreview(move, this);
                return;
            }
        }
    }

    public void HideMovePreview() => previewController?.StopPreview();
}


