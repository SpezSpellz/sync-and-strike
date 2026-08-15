using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PreviewManager : MonoBehaviour
{
    public static PreviewManager Instance { get; private set; }
    private readonly List<PreviewController> previews = new();

    private void Awake() => Instance = this;

    public void RegisterPreview(PreviewController preview)
    {
        if (preview == null || previews.Contains(preview))
            return;
        previews.Add(preview);
    }
    public void UnregisterPreview(PreviewController preview)
    {
        if (preview == null) return;
        previews.Remove(preview);
    }

    public void RestartAllPreviews()
    {
        foreach (var p in previews.ToList())
        {
            if (p != null)
                p.Restart();
        }
    }

    private void Update()
    {
        // step previews (they advance animation + call preview-physics)
        foreach (var p in previews.ToList()) p.Step(Time.deltaTime);

        // after stepping, resolve preview hitboxes
        PreviewHitboxManager.Instance.Step();
    }

    public PreviewController GetPreviewByOwner(CharacterController owner)
    {
        return previews.FirstOrDefault(p => p.Owner == owner);
    }
}