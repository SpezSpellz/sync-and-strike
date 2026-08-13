using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PreviewManager : MonoBehaviour
{
    public static PreviewManager Instance { get; private set; }
    private readonly List<PreviewController> previews = new();

    private void Awake() => Instance = this;

    public void RegisterPreview(PreviewController preview) => previews.Add(preview);
    public void UnregisterPreview(PreviewController preview) => previews.Remove(preview);

    private void Update()
    {
        // step previews (they advance animation + call preview-physics)
        var dt = Time.deltaTime;
        foreach (var p in previews.ToList()) p.Step(dt);

        // after stepping, resolve preview hitboxes
        PreviewHitboxManager.Instance.Step();

        // sync-loop logic: when every preview reached end-of-cycle, restart them all
        if (previews.Count > 0 && previews.All(p => p.IsFinishedOneCycle))
        {
            foreach (var p in previews) p.Restart();
        }
    }

    public PreviewController GetPreviewByOwner(CharacterController owner)
    {
        return previews.FirstOrDefault(p => p.Owner == owner);
    }
}