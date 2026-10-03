using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Unity.VisualScripting;
using System.Linq;
using System.Collections.Generic;

public class MoveSelectionUI : MonoBehaviour
{
    [SerializeField] private GameObject moveButtonPrefab;
    [SerializeField] private GameObject moveColumnPrefab;
    [SerializeField] private GameObject confirmColumnPrefab;
    [SerializeField] private GameObject textButtonPrefab;
    [SerializeField] private GameObject toggleButtonPrefab;
    [SerializeField] private CharacterController owner;
    [SerializeField] private JumpWheel jumpWheel;


    private Transform columnMovement;
    private Transform columnAttack;
    private Transform columnDefense;
    private Transform columnSpecial;
    private Transform columnSuper;
    private MoveButton selectedMoveButton;
    private List<MoveButton> allMoveButton = new List<MoveButton>();

    public int confirmColumnWidth = 160;
    public int confirmButtonHeight = 60;

    public int columnWidth = 160;
    public int columnHeight = 280;
    public int moveButtonWidth = 80;
    public int moveButtonHeight = 80;
    public bool isControllable = true;

    private int nameBoxHeight = 20;
    private bool previewQueued;


    private void Awake()
    {
        // confirm column
        GridLayoutGroup columnConfirm = Instantiate(confirmColumnPrefab, transform).GetComponent<UnityEngine.UI.GridLayoutGroup>();
        columnConfirm.gameObject.name = "Confirm";
        columnConfirm.cellSize = new Vector2(confirmColumnWidth, confirmButtonHeight);
        columnConfirm.spacing = new Vector2(0, 5);

        // lock in button
        GameObject lockInButtonObj = Instantiate(textButtonPrefab, columnConfirm.transform);
        lockInButtonObj.GetComponentInChildren<TextMeshProUGUI>().text = "Lock In";

        LockIn lockIn = lockInButtonObj.AddComponent<LockIn>();
        lockIn.Initialize(isControllable);

        // flip button
        GameObject flipToggleObj = Instantiate(toggleButtonPrefab, columnConfirm.transform);
        flipToggleObj.GetComponentInChildren<TextMeshProUGUI>().text = "Flip";

        Flip flip = flipToggleObj.AddComponent<Flip>();
        flip.Initialize(owner, isControllable);

        columnMovement = CreateColumn("Movement", Vector2.zero);
        columnAttack   = CreateColumn("Attack", Vector2.zero);
        columnSpecial  = CreateColumn("Special", Vector2.zero);
        columnSuper    = CreateColumn("Super", Vector2.zero);
        columnDefense  = CreateColumn("Defense", Vector2.zero);


        AnimationData[] allMoves = Resources.LoadAll<AnimationData>("Characters/Swordsman/AnimationData"); // CHANGE THIS PATH TO MATCH YOUR CHARACTER

        foreach (AnimationData move in allMoves)
        {
            Transform column = GetColumn(move.move);
            if (column == null) continue;

            GameObject buttonObj = Instantiate(moveButtonPrefab, column);
            buttonObj.name = move.moveId;
            MoveButton moveButton = buttonObj.GetComponent<MoveButton>();
            moveButton.Initialize(move, isControllable, owner, this, jumpWheel);
            allMoveButton.Add(moveButton);
        }
    }

    private void OnEnable()
    {
        ClearSelection();
        jumpWheel.Hide();
        if (owner == null) return;
        if (owner.IsPreviewReady)
        {
            StartIdlePreview();
            return;
        }
        previewQueued = true;
        RefreshButtons();
    }

    private void Update()
    {
        if (!previewQueued)
            return;
    
        if (owner == null)
            return;
    
        if (!owner.IsPreviewReady)
            return;
    
        previewQueued = false;
        StartIdlePreview();
    }
    
    private void StartIdlePreview()
    {
        owner.ResumePreview();
        RefreshButtons();
    }

    /// <summary>
    /// Restores every button, then hides the ones the owner genuinely cannot use right now.
    ///
    /// A non-controllable panel is the enemy's read-only mirror of its own moveset, so it always
    /// shows everything: hiding buttons there tells the player nothing useful, and because both
    /// panels share the same bottom bar, launching the enemy used to wipe out half the controls
    /// on screen, which read as the controls disappearing.
    /// </summary>
    private void RefreshButtons()
    {
        foreach (MoveButton moveButton in allMoveButton)
        {
            if (moveButton == null) continue;
            moveButton.gameObject.SetActive(true);
            if (isControllable && !moveButton.IsUsable())
                moveButton.gameObject.SetActive(false);
        }
    }

    private Transform CreateColumn(string name, Vector2 spacing)
    {
        GameObject fullColumn = Instantiate(moveColumnPrefab, transform);
        fullColumn.name = name;

        TextMeshProUGUI nameText = fullColumn.transform.Find("MoveName").GetComponent<TextMeshProUGUI>();
        nameText.text = name;
        Transform moveIcons = fullColumn.transform.Find("MoveIcons");
        GridLayoutGroup grid = moveIcons.GetComponent<GridLayoutGroup>();

        grid.gameObject.name = "Moves";

        grid.cellSize = new Vector2(moveButtonWidth, moveButtonHeight);
        grid.spacing = spacing;

        var layout = grid.GetComponent<LayoutElement>();
        if (layout == null)
            layout = grid.gameObject.AddComponent<LayoutElement>();

        layout.minWidth = columnWidth;
        layout.minHeight = columnHeight;
        layout.preferredWidth = columnWidth;
        layout.preferredHeight = columnHeight;

        // brute force (in case layout system still fights)
        RectTransform rt = grid.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(columnWidth, columnHeight);

        return grid.transform;
    }

    private Transform GetColumn(MoveType moveType)
    {
        switch (moveType)
        {
            case MoveType.Idle:     return columnMovement;
            case MoveType.Movement: return columnMovement;
            case MoveType.Attack:   return columnAttack;
            case MoveType.Defense:  return columnDefense;
            case MoveType.Special:  return columnSpecial;
            case MoveType.Super:    return columnSuper;
            default: return null;
        }
    }

    public void SelectMoveButton(MoveButton button)
    {
        if (selectedMoveButton == button)
            return;

        if (selectedMoveButton != null)
            selectedMoveButton.SetSelected(false);

        selectedMoveButton = button;
        selectedMoveButton.SetSelected(true);
    }

    public void ClearSelection()
    {
        if (selectedMoveButton != null)
        {
            selectedMoveButton.SetSelected(false);
            selectedMoveButton = null;
        }
        if (owner != null)
            owner.HideMovePreview();
    }
}
