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
    private bool initialRefreshComplete;


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
    // OnENABLE()) PRECEDES START() so you have to check the ground AFTER (ground load at start) 
    // so I added ground check at StartIdlePreview()
    {
        ClearSelection();
        jumpWheel.Hide();
        previewQueued = false;
        if (owner == null) return;

        //initialRefresComplete for start up ground check
        if (initialRefreshComplete && owner.IsPreviewReady)
        {
            StartIdlePreview();
            return;
        }


        // On the first scene load, OnEnable runs before the floor registers in Start.
        // Refresh in Update, when the contact check can see the floor.
        previewQueued = true;
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
        initialRefreshComplete = true;
        StartIdlePreview();
    }
    
    private void StartIdlePreview()
    {
        owner.Physics.DetectGround();
        owner.ResumePreview();
        RefreshButtons();
    }

    /// <summary>
    /// Show only moves the panel's fighter can currently use, including in the enemy's read-only panel.
    /// </summary>
    private void RefreshButtons()
    {
        foreach (MoveButton moveButton in allMoveButton)
        {
            if (moveButton == null) continue;
            moveButton.gameObject.SetActive(true);
            if (!moveButton.IsUsable())
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
