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
    public PhysicsManager physicsManager;
    public PreviewPhysicsManager previewPhysicsManager;
    public HitboxManager hitboxManager;
    public PreviewHitboxManager previewHitboxManager;
    public PreviewManager previewManager;
    public Moves moves;
    public TurnManager turnManager;
    public CompanionVoteManager companionVoteManager;
    public UIManager uiManager;

    /// <summary>Cached lookup result for the <paramref name="c"/> component's ancestor chain.</summary>
    public static Arena Of(Component c) => c != null ? c.GetComponentInParent<Arena>() : null;

    private void Reset()
    {
        // Auto-wire to the direct children of this root on editor "Reset" so assembling a new
        // arena scene is a matter of dropping the managers under one root.
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