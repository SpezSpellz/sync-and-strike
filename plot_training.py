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


def newest_csv() -> Path:
    files = sorted(LOG_DIR.glob("training-*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No training-*.csv found in {LOG_DIR}")
    return files[-1]


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
        if c in ("fighter", "role", "team", "result", "source"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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
    ax.set(xlabel="match", ylabel=f"rate (rolling {window})", ylim=(0, 1), title="Ally win / loss rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "winrate.png", dpi=120)
    plt.close(fig)


def plot_metrics(df: pd.DataFrame, out: Path, window: int) -> None:
    for col, label, fname in METRICS:
        if col not in df or "match" not in df or df[col].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for src, g in df.groupby("source"):
            for role, gg in g.groupby("role") if "role" in g else [("all", g)]:
                gg = gg.sort_values("match")
                ax.plot(gg["match"], rolling(gg[col], window), label=f"{src} {role}", lw=1.2)
        ax.set(xlabel="match", ylabel=label, title=label)
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
    args = ap.parse_args()

    paths = args.csv or [newest_csv()]
    df = load(paths)
    summarize(df)

    out = args.out or (paths[0].parent / "plots")
    out.mkdir(parents=True, exist_ok=True)
    plot_winrate(df, out, args.window)
    plot_metrics(df, out, args.window)
    print(f"\nwrote plots to {out}")


if __name__ == "__main__":
    main()
