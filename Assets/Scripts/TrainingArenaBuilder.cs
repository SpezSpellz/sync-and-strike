using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds a complete training arena at runtime so a headless training run needs no hand-authored
/// scene. Each call to <see cref="Build"/> creates an independent <see cref="Arena"/> with its own
/// turn manager, physics, and hitbox managers, containing a 2v1 fight and a simple four-sided stage.
///
/// The layout deliberately mirrors the shipped game, which is ALSO 2v1 simultaneous: one TurnManager
/// on 'GameManager' with Player, Companion and Enemy all registered against it, everyone picking
/// blind and acting in the same tick. Training anything other than that shape produces a policy
/// that scores well here and plays badly in the real game, so the turn system is NOT changed for
/// training - the arena is changed to match the game instead.
///
/// Nothing here runs unless <see cref="TrainingMode.enabled"/>; in the editor it is a no-op so it
/// cannot disturb normal play.
/// </summary>
public class TrainingArenaBuilder : MonoBehaviour
{
    [Header("Layout")]
    [Tooltip("Horizontal spacing between arena roots. Arenas do not interact, so this is only "
             + "cosmetic, but keep it wide enough that sprites do not overlap visually.")]
    public float arenaSpacing = 30f;

    [Header("Stage (world units)")]
    /// <summary>
    /// Y of the floor SURFACE, not of the floor collider's centre.
    ///
    /// Read off SampleScene's Floor collider, which is centred at -1.1877 and is 0.1953 thick, so the
    /// surface a fighter actually stands on is -1.09005. The previous -1.1877 was the collider CENTRE,
    /// which put the training floor about 0.1 lower than the shipped one and changed how high fighters
    /// spawned and how far they fell.
    /// </summary>
    public float floorY = -1.09005f;

    /// <summary>
    /// Y of the ceiling's UNDERSIDE, not of the ceiling collider's centre.
    ///
    /// The authored Ceiling collider is centred at 7.40995 and is 1 unit thick, so the surface that
    /// stops a jump is at 6.90995. Using the centre instead made the training ceiling half a unit
    /// higher than the real one, so jumps were tuned against a ceiling that does not exist in the game.
    /// </summary>
    public float ceilingY = 6.90995f;

    /// <summary>
    /// X of the wall collider's CENTRE. The inner face therefore lands at
    /// <c>wallX - wallThickness/2 = 11.0</c>.
    ///
    /// This was previously applied as the INNER FACE, which pushed the training walls 0.755 further out
    /// per side and made every arena 1.51 units wider than the one that ships - a silent train/serve
    /// mismatch in fighter spacing, on top of the wall height bug.
    /// </summary>
    public float wallX = 11.75569f;

    public float stageWidth = 25.022758f;
    public float floorThickness = 0.1953f;
    public float ceilingThickness = 1f;
    public float wallThickness = 1.5113791f;

    /// <summary>
    /// How far the walls extend BELOW the floor surface.
    ///
    /// This is the value whose absence caused fighters to fall out of the world. Collision resolves by
    /// Minkowski-expanding the wall by the fighter's box, so a grounded fighter is only blocked if the
    /// wall's bottom is within half a fighter height of its body. The walls used to span y [0, 7.41]
    /// while the floor surface was at -1.19: a grounded fighter's box is y [-1.19, -0.49], the expanded
    /// wall's bottom was -0.35, and the two never overlapped. The fighter was therefore not blocked by
    /// the wall at all - it walked off the inner face, ran to the floor's edge and dropped forever.
    ///
    /// 3.0 reproduces the authored wall, which spans y [-4.09, 7.91].
    /// </summary>
    public float wallOvershoot = 3f;

    /// <summary>How far the walls extend ABOVE the ceiling surface. The authored wall tops out 1.0 above it.</summary>
    public float wallHeadroom = 1f;

    [Header("Fighters")]
    [Tooltip("Health each spawned fighter starts a match with.")]
    public float fighterHealth = 100f;

    [Tooltip("Fighter collision size in world units. Must match the authored fighters, since the "
             + "move hitboxes and knockdown logic are sized against it.")]
    public float fighterWidth = 0.5f;
    public float fighterHeight = 0.7f;

    [Tooltip("Root scale for a spawned fighter. MUST match the authored scene fighters (3), which is "
             + "what the game is balanced and framed around.")]
    public float fighterRenderScale = 3f;

    [Header("Fighters (2v1)")]
    [Tooltip("Half the starting distance between the fight lines, in world units.")]
    public float startSeparation = 2.5f;

    [Tooltip("How far behind the player the companion starts, in world units. Staggering them keeps "
             + "them from spawning inside each other and makes the opening readable on screen.")]
    public float companionStandoff = 1.7f;

    [Header("Rendering")]
    [Tooltip("Give spawned fighters a SpriteRenderer so the training run is visible in the viewport. "
             + "Harmless when running -nographics.")]
    public bool renderFighters = true;

    [Tooltip("Only the arena at this index gets a visible renderer and camera. Rendering every "
             + "arena is pointless since they all overlap on one camera, and it costs a lot.")]
    [Range(0, 63)] public int renderArenaIndex = 0;

    [Tooltip("Prefab to clone for each fighter. Must carry an AIController plus its CharacterData, "
             + "CharacterAnimation and CharacterPhysics. If empty, a bare fighter is built from scratch.")]
    public GameObject fighterPrefab;

    /// <summary>The two allies this arena built, in the order they were spawned.</summary>
    public readonly List<CharacterController> allies = new List<CharacterController>();

    /// <summary>The lone enemy this arena built.</summary>
    public CharacterController enemy;

    /// <summary>
    /// Builds one arena whose root is offset by <paramref name="index"/> along X, and returns its
    /// Arena component.
    /// </summary>
    public Arena Build(int index)
    {
        if (!TrainingMode.enabled)
        {
            Debug.LogWarning("TrainingArenaBuilder.Build called outside a training run; ignoring.");
            return null;
        }

        allies.Clear();
        enemy = null;

        var root = new GameObject($"TrainingArena_{index}");
        root.transform.SetParent(transform, false);
        root.transform.localPosition = new Vector3(index * arenaSpacing, 0f, 0f);

        var arena = root.AddComponent<Arena>();

        // Managers. Arena.Awake resolves its own parent by walking up, and the root IS the Arena,
        // so each manager finds this arena.
        var turn = root.AddComponent<TurnManager>();
        var physics = root.AddComponent<PhysicsManager>();
        var hitbox = root.AddComponent<HitboxManager>();
        var previewPhysics = root.AddComponent<PreviewPhysicsManager>();
        var previewHitbox = root.AddComponent<PreviewHitboxManager>();
        var preview = root.AddComponent<PreviewManager>();

        arena.Wire(turn, physics, hitbox, previewPhysics, previewHitbox, preview);

        BuildStage(root.transform);
        ValidateStage();

        // Two allies on the Player team against one Enemy. NOT arbitrary teams: AreEnemies() returns
        // false whenever either side is Neutral, so Neutral fighters would pass through each other.
        bool render = renderFighters && index == renderArenaIndex;

        // Spawned in turn-relevant order: player first, then the companion behind it, then the enemy.
        // Every fighter picks blind and acts in the same tick, so registration order does not affect
        // the fight, but it keeps the log readable.
        var player = SpawnFighter(root.transform, "Player", CombatTeam.Player,
                                  -startSeparation, render, facingLeft: false);
        var companion = SpawnFighter(root.transform, "Companion", CombatTeam.Player,
                                     -startSeparation - companionStandoff, render, facingLeft: false);
        var foe = SpawnFighter(root.transform, "Enemy", CombatTeam.Enemy,
                               startSeparation, render, facingLeft: true);
        allies.Add(player);
        allies.Add(companion);
        enemy = foe;

        // Explicit ally wiring. CompanionController resolves this itself in the shipped scene via a
        // serialized reference, but a code-built fighter has no serialized reference to point at,
        // and both FighterAI spacing and the ally observation features depend on it being non-null.
        var companionAi = companion.GetComponent<CompanionController>();
        if (companionAi != null) companionAi.SetAllyForTraining(player);
        var playerAi = player.GetComponent<CompanionController>();
        if (playerAi != null) playerAi.SetAllyForTraining(companion);

        // Both allies run the companion role on ONE network. The player slot is a stand-in for the
        // human, but giving it its own network would mean two writers racing on
        // companion_policy.json, so the last save would silently discard the other's progress.
        if (playerAi != null && companionAi != null) playerAi.SharePolicyWith(companionAi);

        // The enemy only ever has one hostile, but giving it the far ally as an "ally" reference is
        // wrong - they are on the same team, so leave enemyTarget null and let it resolve the
        // closest hostile each turn.

        if (render)
        {
            // Disable the scene's own follow camera, otherwise it fights us for the view.
            foreach (var existing in FindObjectsByType<CameraFollow>(FindObjectsSortMode.None))
                existing.enabled = false;

            var camGo = new GameObject("TrainingCamera");
            camGo.transform.SetParent(root.transform, false);
            // TrainingCamera carries [RequireComponent(typeof(Camera))], so the camera comes with it.
            var cam = camGo.AddComponent<TrainingCamera>();
            cam.FrameStage(wallX + wallThickness,
                           floorY - floorThickness,
                           floorY,
                           ceilingY + ceilingThickness);
        }

        return arena;
    }

    /// <summary>
    /// One-line description of the built stage, logged once per run.
    ///
    /// Exists because the stage is invisible in every number a run reports. Two arenas that differ only
    /// in wall position train and report identically, so the geometry has to be stated explicitly if it
    /// is ever going to be noticed disagreeing with the shipped scene.
    /// </summary>
    public string StageSummary()
    {
        float wallBottom = floorY - wallOvershoot;
        float wallTop = ceilingY + wallHeadroom;
        return $"floor surface y={floorY:0.###}, ceiling underside y={ceilingY:0.###}, "
               + $"walls y=[{wallBottom:0.###}, {wallTop:0.###}] at |x|={wallX:0.###} "
               + $"(inner face {wallX - wallThickness * 0.5f:0.###}), "
               + $"floor {stageWidth:0.##} wide";
    }

    private void BuildStage(Transform parent)
    {
        // Floor and ceiling are wide-and-short; walls are narrow-and-tall. PhysicsManager keys the
        // stage-wall test off width < height, so these proportions are load-bearing.
        AddStatic(parent, "Floor", new Vector2(0f, floorY - floorThickness * 0.5f),
                  new Vector3(stageWidth, floorThickness, 1f));
        AddStatic(parent, "Ceiling", new Vector2(0f, ceilingY + ceilingThickness * 0.5f),
                  new Vector3(stageWidth, ceilingThickness, 1f));

        // The walls span from below the floor surface to above the ceiling surface, rather than being
        // given a height of their own. Deriving the extent from the two surfaces it has to enclose is
        // what makes the noclip fix structural: change floorY or ceilingY and the walls follow, so
        // there is no longer a second number that can silently disagree with them.
        float wallBottom = floorY - wallOvershoot;
        float wallTop = ceilingY + wallHeadroom;
        float wallHeight = wallTop - wallBottom;
        float wallCenterY = (wallTop + wallBottom) * 0.5f;

        // No thickness subtraction on X: wallX is the collider's centre, matching the authored scene.
        AddStatic(parent, "WallLeft", new Vector2(-wallX, wallCenterY),
                  new Vector3(wallThickness, wallHeight, 1f));
        AddStatic(parent, "WallRight", new Vector2(wallX, wallCenterY),
                  new Vector3(wallThickness, wallHeight, 1f));
    }

    /// <summary>
    /// Assert that the stage actually encloses a fighter, and say which field to fix if it does not.
    ///
    /// Exists because the wall bug was invisible from every number the run reported. Matches completed,
    /// damage was dealt, policies improved - a fighter that fell out of the world simply stopped being
    /// an obstacle, and nothing about a win rate distinguishes that from a policy that learned to win.
    /// The only symptom was a visual glitch in a viewport that most training runs do not even render.
    ///
    /// Cheap, run once per arena at build time, and it names the offending field rather than asserting a
    /// bare invariant.
    /// </summary>
    private void ValidateStage()
    {
        float wallBottom = floorY - wallOvershoot;
        float wallTop = ceilingY + wallHeadroom;
        float wallHeight = wallTop - wallBottom;
        float innerFace = wallX - wallThickness * 0.5f;
        float floorEdge = stageWidth * 0.5f;

        // The check that matters: a grounded fighter's body must reach into the wall's vertical span.
        // With no overshoot the walls float entirely above the floor and the fighter walks straight out.
        if (wallBottom >= floorY)
        {
            Debug.LogError($"[Training] Stage is open at the bottom: walls start at y={wallBottom:0.###} "
                           + $"but the floor surface is y={floorY:0.###}. wallOvershoot must be > 0 - "
                           + "without it a grounded fighter is not blocked by the wall and falls out of "
                           + "the world.");
        }

        if (wallTop <= ceilingY)
        {
            Debug.LogError($"[Training] Stage is open at the top: walls end at y={wallTop:0.###} but the "
                           + $"ceiling surface is y={ceilingY:0.###}. wallHeadroom must be > 0.");
        }

        // The wall must stand ON the floor. If the inner face is beyond the floor's edge, a fighter
        // blocked by the wall is standing on nothing and falls as soon as it touches it.
        if (innerFace <= 0f || innerFace > floorEdge)
        {
            Debug.LogError($"[Training] Stage wall inner face at |x|={innerFace:0.###} does not sit on "
                           + $"the floor, which ends at |x|={floorEdge:0.###}. Adjust wallX or "
                           + "stageWidth so the wall stands on the floor.");
        }

        if (wallHeight <= 0f || wallThickness <= 0f)
        {
            Debug.LogError("[Training] Stage wall has non-positive size; check wallOvershoot, "
                           + "wallHeadroom and wallThickness.");
        }
    }

    private void AddStatic(Transform parent, string name, Vector2 pos, Vector3 scale)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = new Vector3(pos.x, pos.y, 0f);
        go.transform.localScale = scale;
        // Registration happens in PhysicsCollider.Start, which resolves this collider's arena by
        // walking up to the root. The collider is a child of the arena root, so it registers with
        // the right per-arena PhysicsManager without any extra wiring.
        go.AddComponent<StaticCollider>();
    }

    private CharacterController SpawnFighter(Transform parent, string name, CombatTeam team,
                                             float x, bool render, bool facingLeft)
    {
        GameObject go;
        if (fighterPrefab != null)
        {
            go = Instantiate(fighterPrefab, parent);
            go.name = name;
        }
        else
        {
            go = BuildBareFighter(parent, name, team);
        }

        var data = go.GetComponent<CharacterData>();
        var isEnemy = team == CombatTeam.Enemy;

        if (data != null)
        {
            data.team = team;
            data.health = fighterHealth;
            data.maxHealth = fighterHealth;
            // A bare fighter has no authored size, and CharacterPhysics.getBoundingBox is built from
            // these. Left at zero the fighter has no hitbox at all, so nothing can ever hit it and
            // every match would end 0-0 with no damage and no learning signal.
            if (data.width <= 0f) data.width = fighterWidth;
            if (data.height <= 0f) data.height = fighterHeight;
            // Never let a spawned training fighter present a rating prompt or a target reference.
            data.isPlayerSide = false;
        }

        // Place the collision box at the floor contact height from the first planning frame.
        float collisionHeight = data != null ? data.height : fighterHeight;
        go.transform.localPosition = new Vector3(
            x, floorY + collisionHeight * 0.5f + PhysicsConstants.COLLIDER_SKIN, 0f);

        // Scale BEFORE the controller's Start() runs. TurnManager.RegisterPlayer snapshots
        // p.Save(), which stores localScale, so whatever scale is set here is what the match starts
        // from and what ResetState() restores.
        ApplyRenderScale(go, facingLeft, render);

        if (render) AttachRenderer(go, isEnemy);

        return go.GetComponent<CharacterController>();
    }

    /// <summary>
    /// Set the fighter's root scale to match the authored scene fighters, with X negative for a
    /// left-facing fighter.
    ///
    /// This is not only about size. In this game the root X scale IS the facing flag:
    /// CharacterPhysics.getFacing() reads `localScale.x > 0`, while
    /// CharacterController.IsFacingLeft reads `localScale.x < 0`. SetFacingLeft() negates X while
    /// preserving Math.Abs(X), so scale and facing are the same value and change together.
    ///
    /// The authored fighters use 3 (Enemy uses -3). The previous code derived a scale from
    /// sprite.bounds.size.y, which is the whole 128px atlas CELL rather than the character inside
    /// it - the character art only occupies about 41x44 of those 128 pixels - and it also ignored
    /// the authored 3 entirely. That made every training fighter render roughly 4x too small.
    /// </summary>
    private void ApplyRenderScale(GameObject go, bool facingLeft, bool render)
    {
        float magnitude = fighterRenderScale;
        if (!render && fighterPrefab != null)
        {
            // A prefab already carries its own authored scale; do not stomp it.
            magnitude = Mathf.Abs(go.transform.localScale.x);
            if (magnitude <= 0.0001f) magnitude = fighterRenderScale;
        }
        float signed = facingLeft ? -magnitude : magnitude;
        go.transform.localScale = new Vector3(signed, magnitude, 1f);
    }

    /// <summary>
    /// Give a code-built fighter a SpriteRenderer and point its CharacterAnimation at it, so the
    /// animation actually shows. Colour-codes the two sides: allies blue, enemy red.
    /// </summary>
    private void AttachRenderer(GameObject go, bool isEnemy)
    {
        if (go.GetComponent<SpriteRenderer>() != null) return; // prefab already has one

        var renderer = go.AddComponent<SpriteRenderer>();
        renderer.sortingOrder = 1;
        renderer.color = isEnemy
            ? CombatTeamUtility.DebugColor(CombatTeam.Enemy)
            : CombatTeamUtility.DebugColor(CombatTeam.Player);

        var anim = go.GetComponent<CharacterAnimation>();
        if (anim != null) anim.SetSpriteRenderer(renderer);

        // A SpriteRenderer with a null sprite draws nothing at all, so a spawned fighter would be
        // invisible even while it fought correctly. Seed it with the idle frame immediately instead
        // of waiting for the first IdleState.Enter to call PlayIdle.
        //
        // Deliberately NO scale maths here. sprite.bounds is the atlas cell, not the art, so scaling
        // by it shrinks the character instead of normalising it. ApplyRenderScale owns the scale.
        var data = go.GetComponent<CharacterData>();
        if (data != null && data.HasMove("idle"))
        {
            var idle = data.GetMove("idle");
            if (idle != null && idle.frames != null && idle.frames.Length > 0)
                renderer.sprite = idle.frames[0];
        }
    }

    /// <summary>
    /// Minimal fighter built from code, for when no prefab is assigned. Deliberately omits a
    /// SpriteRenderer: training runs headless with -nographics and never draws anything, and
    /// CharacterAnimation tolerates a null renderer.
    /// </summary>
    private GameObject BuildBareFighter(Transform parent, string name, CombatTeam team)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);

        // Component order matters: CharacterController.Awake runs as soon as the controller is added
        // and immediately does GetComponent for the other three, so they must already be present.
        go.AddComponent<CharacterData>();
        go.AddComponent<CharacterPhysics>();
        go.AddComponent<CharacterAnimation>();

        // The AI type follows the slot, because the role decides which policy (and which weights
        // file) the fighter learns. Putting EnemyController on both fighters - as this did when the
        // arena was 1v1 - gave every fighter RoleEnemy, so they all trained one shared network and
        // overwrote each other's enemy_policy.json on every save.
        if (team == CombatTeam.Enemy) go.AddComponent<EnemyController>();
        else go.AddComponent<CompanionController>();

        return go;
    }
}
