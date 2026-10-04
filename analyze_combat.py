"""Per-fighter action breakdown: what are the fighters actually choosing?

The health/power summary hides the question that matters when a policy "looks dumb": WHICH actions is
it picking? damage_dealt_to_enemy is the same whether you land one heavy hit or five light ones, and the
same whether a fighter is winning by pressing attack or standing still and absorbing.

This groups the telemetry by fighter and reports, per fighter, the share of matches in which each move
was the last one submitted - which is the decision the policy actually made. Move choice is read from
the console heartbeat rather than the CSV (the CSV has no move column), so this works on any run that
logged a heartbeat.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

HEARTBEAT = re.compile(
    r"heartbeat:.*?\[(?P<state>.*)\]\s*$"
)
FIGHTER = re.compile(
    r"\[(?P<name>[A-Za-z_0-9]+) hp=(?P<hp>-?[\d.]+) pos=\((?P<px>-?[\d.]+), (?P<py>-?[\d.]+)\) "
    r"state=(?P<state>\w+) move='(?P<move>[^']*)'"
)


def parse(path: Path):
    """Yield (match_no, fighter, move, hp, state, pos_x, pos_y) per heartbeat."""
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"heartbeat: (?P<matches>\d+) matches done", line)
        if m:
            cur = int(m.group("matches"))
        for f in FIGHTER.finditer(line):
            yield (
                cur,
                f.group("name"),
                f.group("move"),
                float(f.group("hp")),
                f.group("state"),
                float(f.group("px")),
                float(f.group("py")),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="Editor.log")
    args = ap.parse_args()
    if not args.log.exists():
        raise SystemExit(f"not found: {args.log}")

    rows = list(parse(args.log))
    if not rows:
        raise SystemExit("no heartbeat fighter lines found")

    per_fighter_moves = defaultdict(Counter)
    per_fighter_state = defaultdict(Counter)
    hp_samples = defaultdict(list)
    gap_samples = []

    for match_no, name, move, hp, state, px, py in rows:
        if match_no is None:
            continue
        per_fighter_moves[name][move] += 1
        per_fighter_state[name][state] += 1
        hp_samples[name].append(hp)

    # Separation: for each heartbeat, the max pairwise x-gap between fighters.
    by_match = defaultdict(list)
    for match_no, name, move, hp, state, px, py in rows:
        if match_no is not None:
            by_match[match_no].append((name, px))
    for _, fighters in by_match.items():
        xs = sorted(px for _, px in fighters)
        if len(xs) >= 2:
            gap_samples.append(max(xs) - min(xs))

    print("=" * 70)
    print(f"heartbeat samples: {len(rows)}")
    print("=" * 70)
    for name in sorted(per_fighter_moves):
        total = sum(per_fighter_moves[name].values())
        print(f"\n{name}  ({total} samples)")
        print("  move chosen:")
        for move, n in per_fighter_moves[name].most_common():
            print(f"    {move or '<none>':<20} {100.0*n/total:5.1f}%")
        print("  state:")
        for st, n in per_fighter_state[name].most_common(5):
            print(f"    {st:<20} {100.0*n/total:5.1f}%")
        hs = sorted(hp_samples[name])
        if hs:
            print(f"  health: min {hs[0]:.0f}  median {hs[len(hs)//2]:.0f}  max {hs[-1]:.0f}")

    if gap_samples:
        gap_samples.sort()
        print(f"\nseparation (x-gap between outermost fighters), n={len(gap_samples)}")
        print(f"  min {gap_samples[0]:.1f}  p25 {gap_samples[len(gap_samples)//4]:.1f} "
              f"median {gap_samples[len(gap_samples)//2]:.1f} "
              f"p75 {gap_samples[3*len(gap_samples)//4]:.1f}  max {gap_samples[-1]:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()