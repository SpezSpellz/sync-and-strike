using UnityEngine;

/// <summary>
/// Which side a combatant fights for.
///
/// Note that <see cref="CombatTeam"/> identifies a specific fighter (the player, the
/// companion, the enemy) while <see cref="CombatSide"/> identifies who they are fighting
/// for. The player and the companion are different fighters on the SAME side, which is why
/// they pass through each other and cannot damage each other.
/// </summary>
public enum CombatTeam
{
    Neutral = 0,
    Player = 1,
    Companion = 2,
    Enemy = 3,
}

public enum CombatSide
{
    Neutral = 0,
    PlayerSide = 1,
    EnemySide = 2,
}

public static class CombatTeamUtility
{
    public static CombatSide SideOf(CombatTeam team)
    {
        switch (team)
        {
            case CombatTeam.Player:
            case CombatTeam.Companion:
                return CombatSide.PlayerSide;
            case CombatTeam.Enemy:
                return CombatSide.EnemySide;
            default:
                return CombatSide.Neutral;
        }
    }

    /// <summary>True when both fighters are on the same side, so they must not push or hurt each other.</summary>
    public static bool AreAllies(CombatTeam a, CombatTeam b)
    {
        CombatSide sa = SideOf(a);
        CombatSide sb = SideOf(b);
        if (sa == CombatSide.Neutral || sb == CombatSide.Neutral) return true;
        return sa == sb;
    }

    public static bool AreEnemies(CombatTeam a, CombatTeam b)
    {
        CombatSide sa = SideOf(a);
        CombatSide sb = SideOf(b);
        if (sa == CombatSide.Neutral || sb == CombatSide.Neutral) return false;
        return sa != sb;
    }

    public static Color DebugColor(CombatTeam team)
    {
        switch (team)
        {
            case CombatTeam.Player: return new Color(0.66f, 0.64f, 1f);
            case CombatTeam.Companion: return new Color(0.4f, 0.9f, 0.6f);
            case CombatTeam.Enemy: return new Color(1f, 0.48f, 0.5f);
            default: return Color.white;
        }
    }
}