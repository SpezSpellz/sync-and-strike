using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Groups one fight's per-instance managers so that N fights can run in parallel inside a single
/// Unity process (the training scenario). Put an Arena component on each arena root and wire its
/// child managers into the fields below.
///
/// In a single-arena scene such as the built-in SampleScene you may have zero Arena components at
/// all: every manager resolves to itself / its static Instance as a fallback, preserving the
/// original single-arena behaviour.
///
/// Note that this scopes the *simulation* managers. UI, camera, and input (UIManager, CameraFollow,
/// Buttons) are still resolved through their static Instance fields and are therefore only
/// supported as a single global scene object. During headless multi-arena training those
/// components are detached or disabled.
/// </summary>
public class Arena : MonoBehaviour
{
    [SerializeField] private PhysicsManager physicsManager;
    [SerializeField] private PreviewPhysicsManager previewPhysicsManager;
    [SerializeField] private HitboxManager hitboxManager;
    [SerializeField] private PreviewHitboxManager previewHitboxManager;
    [SerializeField] private PreviewManager previewManager;
    [SerializeField] private Moves moves;
    [SerializeField] private TurnManager turnManager;
    [SerializeField] private CompanionVoteManager companionVoteManager;
    [SerializeField] private UIManager uiManager;

    /// <summary>
    /// Every accessor below auto-wires itself from this arena's children on first use.
    ///
    /// This is deliberately not just a convenience. An arena assembled in code (see
    /// TrainingArenaBuilder) does not get its inspector references assigned, and every unguarded
    /// <c>arena.someManager.Method()</c> call then threw NullReferenceException once per frame,
    /// 85k times in a single session. Resolving lazily means a half-wired arena degrades to
    /// "that feature is unavailable" instead of throwing every frame.
    /// </summary>
    private T Resolve<T>(ref T field) where T : Component
    {
        if (field == null) field = GetComponentInChildren<T>();
        return field;
    }

    /// <summary>This arena's physics manager. Never null once the arena has been built.</summary>
    public PhysicsManager PhysicsManager => Resolve(ref physicsManager);

    /// <summary>This arena's preview physics manager.</summary>
    public PreviewPhysicsManager PreviewPhysicsManager => Resolve(ref previewPhysicsManager);

    /// <summary>This arena's hitbox resolver. Never null once the arena has been built.</summary>
    public HitboxManager HitboxManager => Resolve(ref hitboxManager);

    /// <summary>This arena's preview hitbox resolver.</summary>
    public PreviewHitboxManager PreviewHitboxManager => Resolve(ref previewHitboxManager);

    /// <summary>This arena's preview driver.</summary>
    public PreviewManager PreviewManager => Resolve(ref previewManager);

    /// <summary>This arena's move display data. May be absent; only the legacy Cap code uses it.</summary>
    public Moves Moves => Resolve(ref moves);

    /// <summary>This arena's turn loop. Never null once the arena has been built.</summary>
    public TurnManager TurnManager => Resolve(ref turnManager);

    /// <summary>
    /// This arena's companion vote manager. Genuinely optional: arenas built for training have no
    /// voting at all, so callers must null-check this one.
    /// </summary>
    public CompanionVoteManager CompanionVoteManager => Resolve(ref companionVoteManager);

    /// <summary>
    /// This arena's UI. Genuinely optional: a headless training arena has none, so callers must
    /// null-check this one.
    /// </summary>
    public UIManager UIManager => Resolve(ref uiManager);

    /// <summary>
    /// Assign the managers explicitly. Used when an arena is assembled in code rather than in the
    /// editor, where the lazy resolver would otherwise have to search for each one.
    /// </summary>
    public void Wire(TurnManager turn, PhysicsManager physics, HitboxManager hitbox,
                     PreviewPhysicsManager previewPhysics, PreviewHitboxManager previewHitbox,
                     PreviewManager preview)
    {
        turnManager = turn;
        physicsManager = physics;
        hitboxManager = hitbox;
        previewPhysicsManager = previewPhysics;
        previewHitboxManager = previewHitbox;
        previewManager = preview;
        // UI and voting are intentionally left null: a headless arena has neither.
    }

    /// <summary>Cached lookup result for the <paramref name="c"/> component's ancestor chain.</summary>
    public static Arena Of(Component c) => c != null ? c.GetComponentInParent<Arena>() : null;

    /// <summary>
    /// Every fighter inside this arena.
    ///
    /// Target resolution MUST use this rather than a scene-wide FindObjectsByType: with several
    /// arenas alive, a global search lets a fighter in one arena pick an opponent in another, which
    /// silently trains the wrong matchup.
    /// </summary>
    public List<CharacterController> FightersInArena()
    {
        var result = new List<CharacterController>();
        var found = GetComponentsInChildren<CharacterController>(true);
        for (int i = 0; i < found.Length; i++) result.Add(found[i]);
        return result;
    }

    /// <summary>
    /// Fighters in the same arena as <paramref name="c"/>, or every fighter in the scene when it has
    /// no arena (the single-arena case, where there is nothing to scope against).
    /// </summary>
    public static List<CharacterController> FightersNear(Component c)
    {
        var arena = Of(c);
        if (arena != null) return arena.FightersInArena();
        return new List<CharacterController>(
            FindObjectsByType<CharacterController>(FindObjectsSortMode.None));
    }

    private void Reset()
    {
        // Eagerly populate the serialized fields in the editor so they are visible/assignable. At
        // runtime the accessors above do this lazily instead.
        physicsManager = GetComponentInChildren<PhysicsManager>();
        previewPhysicsManager = GetComponentInChildren<PreviewPhysicsManager>();
        hitboxManager = GetComponentInChildren<HitboxManager>();
        previewHitboxManager = GetComponentInChildren<PreviewHitboxManager>();
        previewManager = GetComponentInChildren<PreviewManager>();
        moves = GetComponentInChildren<Moves>();
        turnManager = GetComponentInChildren<TurnManager>();
        companionVoteManager = GetComponentInChildren<CompanionVoteManager>();
        uiManager = GetComponentInChildren<UIManager>();
    }
}