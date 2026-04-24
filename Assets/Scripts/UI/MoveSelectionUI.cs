using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;

public class MoveSelectionUI : MonoBehaviour
{
    [SerializeField] private GameObject moveButtonPrefab;
    [SerializeField] private GameObject moveColumnPrefab;
    [SerializeField] private GameObject textButtonPrefab;

    private Transform columnMovement;
    private Transform columnAttack;
    private Transform columnDefense;
    private Transform columnSpecial;
    private Transform columnSuper;

    public int confirmColumnWidth = 160;
    public int confirmButtonHeight = 60;

    public int columnWidth = 160;
    public int columnHeight = 300;
    public int moveButtonWidth = 80;
    public int moveButtonHeight = 80;


    private void Start()
    {
        GridLayoutGroup columnConfirm = Instantiate(moveColumnPrefab, transform).GetComponent<UnityEngine.UI.GridLayoutGroup>();
        columnConfirm.gameObject.name = "Confirm";
        columnConfirm.cellSize = new Vector2(confirmColumnWidth, confirmButtonHeight);
        columnConfirm.spacing = new Vector2(0, 5);

        GameObject lockInButtonObj = Instantiate(textButtonPrefab, columnConfirm.transform);
        lockInButtonObj.GetComponentInChildren<TextMeshProUGUI>().text = "Lock In";

        lockInButtonObj.AddComponent<LockIn>();

        columnMovement = CreateColumn("Movement", Vector2.zero);
        columnAttack   = CreateColumn("Attack", Vector2.zero);
        columnDefense  = CreateColumn("Defense", Vector2.zero);
        columnSpecial  = CreateColumn("Special", Vector2.zero);
        columnSuper    = CreateColumn("Super", Vector2.zero);


        AnimationData[] allMoves = Resources.LoadAll<AnimationData>("Characters/Swordsman/AnimationData"); // CHANGE THIS PATH TO MATCH YOUR CHARACTER

        foreach (AnimationData move in allMoves)
        {
            Transform column = GetColumn(move.move);
            if (column == null) continue;

            GameObject buttonObj = Instantiate(moveButtonPrefab, column);
            MoveButton moveButton = buttonObj.GetComponent<MoveButton>();
            moveButton.Initialize(move);
        }
    }

    private Transform CreateColumn(string name, Vector2 spacing)
    {
        GridLayoutGroup grid = Instantiate(moveColumnPrefab, transform)
                                .GetComponent<GridLayoutGroup>();

        grid.gameObject.name = name;

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
}