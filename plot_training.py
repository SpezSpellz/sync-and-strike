#!/usr/bin/env python
"""Summarize and plot Synced Strike PPO training telemetry.

The game writes one long-format row per fighter per finished match to
``%USERPROFILE%/AppData/LocalLow/DefaultCompany/Sync-and-Strike/training_logs/training-*.csv``.
This script reads those files, prints a health summary, and writes one PNG per metric.

Usage
-----
    python plot_training.py                      # newest CSV in the log dir
    python plot_training.py path/to/file.csv     # a specific CSV
    python plot_training.py a.csv b.csv          # overlay several runs
    python plot_training.py --out plots_dir      # where PNGs go

Run it with the project's virtualenv: ``.venv\\Scripts\\python.exe plot_training.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path(os.path.expanduser("~")) / (
    "AppData/LocalLow/DefaultCompany/Sync-and-Strike/training_logs"
)

# Columns the game may not always emit (older runs). Missing ones are skipped.
METRICS = [
    ("reward_last", "reward (last turn)", "reward"),
    ("entropy", "categorical entropy (masked), nats", "entropy"),
    ("entropy_unmasked", "categorical entropy (unmasked), nats", "entropy_unmasked"),
    ("action_share_top1", "top-1 action share", "action_share_top1"),
    ("legal_action_count", "legal actions available", "legal_action_count"),
    ("explained_variance", "critic explained variance", "explained_variance"),
    ("value_loss", "critic value loss", "value_loss"),
    ("approx_kl", "approx KL (epoch 0)", "approx_kl"),
    ("kl_all_epochs", "approx KL (all epochs, vs target)", "kl_all"),
    ("epochs_run", "epochs run before KL early stop", "epochs_run"),
    ("clip_fraction", "clip fraction", "clip_fraction"),
    ("advantage_magnitude", "|advantage|", "advantage_magnitude"),
    ("whiff_rate", "whiff rate", "whiff_rate"),
    ("damage_dealt_to_enemy", "damage dealt to enemy", "damage_dealt"),
    ("damage_taken", "damage taken", "damage_taken"),
    ("hits_landed", "hits landed", "hits_landed"),
    ("attacks_thrown", "attacks thrown", "attacks_thrown"),
    ("health_left", "health remaining", "health_left"),
    ("turns", "turns per match", "turns"),
    ("frames_turn", "sim frames per turn", "frames_turn"),
]


# Columns that hold text. These must NOT be run through pd.to_numeric: doing so silently turns
# the opponent tokens ("random"/"rule"/"live") into NaN, which is what happened before - every
# opponent-aware analysis came back empty and looked like "no curriculum data" rather than
# "the column was destroyed on load".
STRING_COLUMNS = ("fighter", "role", "team", "result", "opponent", "source")

# Opponents the curriculum can grade a policy against. The evaluated side plays its own network
# and faces one of these; the other side's rows carry opponent == "live_eval" and are NOT graded.
GRADED_OPPONENTS = ("random", "rule")


def newest_csv() -> Path:
    files = sorted(LOG_DIR.glob("training-*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No training-*.csv found in {LOG_DIR}")
    return files[-1]


def true_opponent(df: pd.DataFrame) -> pd.Series:
    """The source of the side OPPOSING each fighter, rebuilt per match.

    The C# writer originally recorded each fighter's OWN behaviour source under the name
    ``opponent``, which is inverted. Rather than only handling files written after that was fixed,
    this reconstructs the real value from data already on disk.

    The reconstruction is exact and needs no version sniffing: within one match every row belonging to
    a side carries that same side's source, so the value appearing on the OTHER side's rows is by
    definition the opponent's source. That holds whether the column stored own-source or
    opponent-source, so this single rule is correct for old and new files alike.
    """
    if "opponent" not in df.columns or "team" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    is_enemy = df["team"].astype(str).str.contains("Enemy", case=False, na=False)

    def first_valid(s: pd.Series):
        v = s.dropna()
        return v.iloc[0] if len(v) else np.nan

    side = (df.assign(_enemy=is_enemy)
              .groupby(["source", "match", "_enemy"])["opponent"]
              .agg(first_valid)
              .unstack("_enemy"))
    # _enemy False = ally rows, True = enemy rows.
    ally_side = side[False] if False in side.columns else pd.Series(np.nan, index=side.index)
    enemy_side = side[True] if True in side.columns else pd.Series(np.nan, index=side.index)

    ally_opp = enemy_side.reindex(pd.MultiIndex.from_arrays([df["source"], df["match"]])).to_numpy()
    enemy_opp = ally_side.reindex(pd.MultiIndex.from_arrays([df["source"], df["match"]])).to_numpy()
    return pd.Series(np.where(is_enemy, enemy_opp, ally_opp), index=df.index, name="opponent_true")


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            sys.exit(f"CSV not found: {p}")
        df = pd.read_csv(p)
        df["source"] = p.stem
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Numeric coercion: telemetry can contain blank cells where a policy was null.
    for c in df.columns:
        if c in STRING_COLUMNS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["opponent_true"] = true_opponent(df)

    # Legacy CSVs predate is_learner. Derive it rather than falling back to "everything is a
    # learner", because that fallback silently mixes scripted sparring partners into the policy
    # curves - which is exactly what makes a run where nothing is learned still look like it has
    # trends. TrainingMatchRunner assigns the learner by arena index parity (even arenas train the
    # companion, odd arenas train the enemy) and eval matches record nothing, so both are
    # reconstructible. Approximate only in that it cannot see a role that was mid-switch.
    if "is_learner" not in df.columns and "arena" in df.columns:
        even_arena = (df["arena"] % 2) == 0
        is_ally = ~df["team"].astype(str).str.contains("Enemy", case=False, na=False)
        derived = np.where(even_arena, is_ally, ~is_ally)
        if "is_eval" in df.columns:
            derived = np.where(df["is_eval"] == 1, 0, derived)
        df["is_learner"] = derived
        print("note: is_learner absent; derived from arena parity (legacy approximation).")

    return df


def role_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, match, role), carrying that ROLE's own win flag.

    The ``result`` column is TEAM-relative: ``AlliesWin`` means the two allies won. So the enemy
    role won on ``AlliesLose``, and a Draw means nobody won. Treating ``result`` as a per-role
    outcome is the easiest way to misread this file - it reports the enemy's wins backwards.
    """
    cols = [c for c in ("source", "match", "role", "arena", "result", "opponent",
                    "opponent_true", "is_eval", "is_learner", "turns") if c in df]
    m = df.groupby(["source", "match", "role", "arena"], as_index=False).first()
    allies_won = m["result"].eq("AlliesWin")
    allies_lost = m["result"].eq("AlliesLose")
    m["role_win"] = np.where(m["role"] == "enemy", allies_lost, allies_won)
    m["ally_win"] = allies_won
    return m


def summarize(df: pd.DataFrame) -> None:
    line = "=" * 72
    print(line)
    print(f"runs: {', '.join(sorted(df['source'].unique()))}")
    print(f"rows: {len(df):,}")

    if "match" in df:
        per_match = df.groupby(["source", "match"], as_index=False).first()
    else:
        per_match = df
    print(f"matches: {len(per_match):,}")

    if "result" in df:
        counts = per_match["result"].value_counts()
        total = counts.sum()
        print("\nResults:")
        for name in ("AlliesWin", "AlliesLose", "Draw"):
            n = int(counts.get(name, 0))
            print(f"  {name:<11} {n:>6}  ({100.0 * n / max(total, 1):5.1f}%)")
        decisive = int(counts.get("AlliesWin", 0)) + int(counts.get("AlliesLose", 0))
        if decisive:
            print(
                f"  ally win rate (decisive): "
                f"{100.0 * counts.get('AlliesWin', 0) / decisive:5.1f}%"
            )

    # Per-role wins. Printed because 'result' is team-relative and reading the enemy row's
    # Ally-win figure as an enemy result reports its losses as wins.
    if "role" in df:
        rv = role_view(df)
        print("\nWin rate by role (result is team-relative; the enemy won on AlliesLose):")
        for role, g in rv.groupby("role"):
            wr = 100.0 * g["role_win"].mean()
            print(f"  {role:<10} {wr:5.1f}%   (n={len(g)} matches)")
        if "is_learner" in rv:
            L = rv[rv["is_learner"] == 1]
            print(f"\nLearner rows: {len(L)} / {len(rv)}"
                  f"  (a role is the learner on only half the arenas by design)")
    print(line)
    roles = df["role"].dropna().unique() if "role" in df else []
    for role in roles:
        sub = df[df["role"] == role]
        print(f"\nrole = {role}  ({len(sub):,} rows)")
        for col, label, _ in METRICS:
            if col not in sub or sub[col].notna().sum() == 0:
                continue
            first = sub.sort_values("match")[col].head(max(1, len(sub) // 10)).mean()
            last = sub.sort_values("match")[col].tail(max(1, len(sub) // 10)).mean()
            if col in ("entropy", "entropy_unmasked", "whiff_rate", "action_share_top1", "clip_fraction"):
                fmt = "{:.3f}"
            else:
                fmt = "{:.3g}"
            print(
                f"  {label:<32} mean {fmt.format(sub[col].mean()):>10}   "
                f"first10% {fmt.format(first):>10}  last10% {fmt.format(last):>10}"
            )

    # Collapse warnings.
    for col, label in [
        ("network_poisoned", "poisoned weight snapshots"),
        ("nonfinite_grad_steps", "non-finite gradient steps"),
    ]:
        if col in df and df[col].max() > 0:
            print(f"\nWARNING: {label}: max {df[col].max():g}")
    print(line)


def rolling(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()


# Mirrors OpponentCurriculum's defaults. Passed in so a sweep (-mix.advanceRule=...) can be
# reflected here without editing the script.
ADV_RANDOM = 0.85
ADV_RULE = 0.70
REGRESS_RULE = 0.50
ADVANCE_WINDOW = 16


def gate_status(df: pd.DataFrame, adv_random: float, adv_rule: float) -> None:
    """How close each role is to a curriculum promotion, and what it is facing.

    This is the check that would have caught the stuck-curriculum bug immediately. A gate is only
    as fast as the evidence feeding it: the eval slice rotates over 4 (role x opponent) pairs, so
    each pair earns one graded sample every 4 x evalEvery matches. A window that looks reasonable in
    isolation can still need thousands of matches to fill.
    """
    m = role_view(df)
    ev = m[(m["is_eval"] == 1) & (m["opponent_true"].isin(GRADED_OPPONENTS))]
    print("\nCurriculum gate:")
    if ev.empty:
        print("  NO graded eval matches at all - the curriculum cannot promote and never will.")
        return
    for role, g in ev.groupby("role"):
        parts = []
        for opp in GRADED_OPPONENTS:
            n = int((g["opponent_true"] == opp).sum())
            rate = g.loc[g["opponent_true"] == opp, "role_win"].mean() if n else float("nan")
            parts.append(f"vs {opp}: {n}/{ADVANCE_WINDOW} evals, WR {rate:.3f}")
        print(f"  {role:<10} {'   '.join(parts)}")
    n_ev = int(ev.groupby(["role", "opponent_true"]).ngroups)
    print(f"  (gate needs {ADVANCE_WINDOW} graded evals per opponent per role;"
          f" {len(ev) // max(n_ev, 1)} avg per cell so far)")

    # What the curriculum actually produced. A run where this is 100% random never promoted.
    tr = m[(m["is_eval"] == 0) & (m["is_learner"] == 1)] if "is_learner" in m else m[m["is_eval"] == 0]
    print("\nOpponents faced by the LEARNER (curriculum draws, eval matches excluded):")
    for role, g in tr.groupby("role"):
        counts = g["opponent_true"].value_counts().to_dict()
        share = {k: round(v / max(len(g), 1), 3) for k, v in counts.items()}
        print(f"  {role:<10} {share}")
    # Only random here means no curriculum draw ever returned rule or live, i.e. no promotion.
    if tr.groupby("role")["opponent_true"].apply(lambda s: set(s.unique()) <= {"random"}).all():
        print("  ^ every draw was 'random': the curriculum never left phase 1/3.")


def plot_eval_winrate(df: pd.DataFrame, out: Path, window: int,
                      adv_random: float, adv_rule: float, regress_rule: float) -> None:
    """The chart that matters: win rate against the two FIXED baselines, per role.

    Aggregate win rate is meaningless once a curriculum is running - it mixes random, rule-based and
    live opponents, and the role being measured is only on one side of it. These two curves are the
    only uncontaminated measurements in the run, and they are also exactly what the promotion gate
    reads, so a curve that stops climbing is a promotion that will not happen.
    """
    m = role_view(df)
    ev = m[(m["is_eval"] == 1) & (m["opponent_true"].isin(GRADED_OPPONENTS))]
    fig, axes = plt.subplots(1, len(GRADED_OPPONENTS), figsize=(13, 4.2), sharey=True)
    for ax, opp in zip(np.atleast_1d(axes), GRADED_OPPONENTS):
        sub = ev[ev["opponent_true"] == opp]
        for role, g in sub.groupby("role"):
            g = g.sort_values("match")
            ax.plot(g["match"],
                    rolling(g["role_win"].astype(float), window),
                    marker="o", ms=3, lw=1.4, label=f"{role}  (n={len(g)})")
        thr = adv_random if opp == "random" else adv_rule
        ax.axhline(thr, ls="--", lw=1, color="crimson",
                   label=f"promote at {thr:.2f}")
        if opp == "rule":
            ax.axhline(regress_rule, ls=":", lw=1, color="darkorange",
                       label=f"regress below {regress_rule:.2f}")
        ax.set(xlabel="match", ylabel=f"win rate (rolling {window})",
               title=f"eval win rate vs {opp}", ylim=(-0.05, 1.05))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Uncontaminated eval win rate (these are the promotion-gate signals)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "eval_winrate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_opponent_mix(df: pd.DataFrame, out: Path) -> None:
    """Share of each opponent the LEARNER faced, over time: the curriculum's phase progression.

    Uses learner rows only. Every fighter gets a row every match, so the scripted ones would
    otherwise dominate and make the mix look constant regardless of what the curriculum did.
    """
    src = df[df["is_learner"] == 1] if "is_learner" in df else df
    src = src[src["is_eval"] == 0]
    roles = list(src["role"].dropna().unique())
    if not roles:
        return
    fig, axes = plt.subplots(1, len(roles), figsize=(7 * len(roles), 4), squeeze=False)
    for ax, role in zip(axes[0], roles):
        g = src[src["role"] == role]
        if g.empty:
            continue
        nbins = 12
        bins = np.linspace(g["match"].min(), g["match"].max() + 1, nbins + 1)
        binned = g.assign(bin=pd.cut(g["match"], bins, labels=False, include_lowest=True))
        # Reindex to the full bin range: pd.crosstab drops empty bins, and stackplot then gets an
        # x longer than its y. Empty bins are real information here - they are matches with no
        # learner rows - so they are filled with zero rather than discarded.
        mix = (pd.crosstab(binned["bin"], binned["opponent_true"], normalize="index")
               .reindex(range(nbins), fill_value=0)
               .fillna(0))
        ax.stackplot(np.arange(nbins), mix.values, labels=list(mix.columns))
        centres = [(bins[i] + bins[i + 1]) / 2 for i in range(nbins)]
        ax.set_xticks(range(nbins))
        ax.set_xticklabels([f"{c:.0f}" for c in centres], fontsize=7, rotation=45)
        ax.set(xlabel="match", ylabel="share", ylim=(0, 1), title=f"{role}: opponent mix")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Opponent mix faced by the learner (a flat 100% random band = stuck in phase 1/3)")
    fig.tight_layout()
    fig.savefig(out / "opponent_mix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_winrate(df: pd.DataFrame, out: Path, window: int) -> None:
    if "result" not in df or "match" not in df:
        return
    per_match = df.groupby(["source", "match"], as_index=False).first()
    fig, ax = plt.subplots(figsize=(10, 4))
    for src, g in per_match.groupby("source"):
        wins = (g["result"] == "AlliesWin").astype(float)
        losses = (g["result"] == "AlliesLose").astype(float)
        if len(g) == 0:
            continue
        ax.plot(g["match"], rolling(wins, window), label=f"{src} win")
        ax.plot(g["match"], rolling(losses, window), label=f"{src} loss", alpha=0.6)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set(xlabel="match", ylabel=f"rate (rolling {window})", ylim=(0, 1),
           title="Ally win / loss rate  (AGGREGATE - mixes all opponents, see eval_winrate.png)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "winrate.png", dpi=120)
    plt.close(fig)


def plot_metrics(df: pd.DataFrame, out: Path, window: int) -> None:
    """One chart per metric, learner rows only.

    Restricted to learner rows because every fighter gets a row in every match, including the ones
    acting from a scripted source. Mixing them attributes a random sparring partner's behaviour to
    the learned policy and flattens every curve - which is exactly how a run where nothing is
    learned still looks like it has trends.
    """
    src = df[df["is_learner"] == 1] if "is_learner" in df else df
    if "is_learner" not in df:
        print("  note: this CSV predates is_learner; metric charts include scripted rows.")
    if src.empty:
        src = df
    for col, label, fname in METRICS:
        if col not in src or "match" not in src or src[col].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for srcname, g in src.groupby("source"):
            for role, gg in g.groupby("role") if "role" in g else [("all", g)]:
                gg = gg.sort_values("match")
                ax.plot(gg["match"], rolling(gg[col], window), label=f"{srcname} {role}", lw=1.2)
        ax.set(xlabel="match", ylabel=label, title=f"{label}  (learner rows only)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", type=Path, help="CSV file(s); default = newest in the log dir")
    ap.add_argument("--out", type=Path, default=None, help="directory for PNGs")
    ap.add_argument("--window", type=int, default=25, help="rolling-mean window in matches")
    ap.add_argument("--adv-random", type=float, default=ADV_RANDOM,
                    help="promotion threshold vs random (mirror -mix.advanceRandom)")
    ap.add_argument("--adv-rule", type=float, default=ADV_RULE,
                    help="promotion threshold vs rule (mirror -mix.advanceRule)")
    args = ap.parse_args()

    paths = args.csv or [newest_csv()]
    df = load(paths)
    summarize(df)
    gate_status(df, args.adv_random, args.adv_rule)

    out = args.out or (paths[0].parent / "plots")
    out.mkdir(parents=True, exist_ok=True)
    plot_eval_winrate(df, out, args.window, args.adv_random, args.adv_rule, REGRESS_RULE)
    plot_opponent_mix(df, out)
    plot_winrate(df, out, args.window)
    plot_metrics(df, out, args.window)
    print(f"\nwrote plots to {out}")


if __name__ == "__main__":
    main()
