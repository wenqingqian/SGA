#!/usr/bin/env python3
"""Pick the replay argmin and print shell assignments.

Reads a report produced by ``python -m sga.replay --replay-mode cycle`` with
multiple cycle lengths. Cells are keyed ``<score>@T<cycle_len>`` under
``branch_results[<t0>]["per_ratio"][<ratio>]["scores"]``; distortion values
live under each cell's ``distortion[<horizon>]``.

The selected horizon is the largest one with a finite distortion for every
cell. Ties break deterministically by (distortion, score, ratio, T).
"""
import argparse
import json
import math
import re
import sys

_CELL_RE = re.compile(r"^(?P<score>[A-Za-z0-9_]+)(?:@T(?P<t>\d+))?$")


def _cells(branch):
    cells = []
    for ratio_key, block in branch["per_ratio"].items():
        for key, cell in block["scores"].items():
            m = _CELL_RE.match(key)
            if not m:
                raise SystemExit(f"unexpected replay cell key: {key!r}")
            dist = cell.get("distortion") or {}
            values = {}
            for h, v in dist.items():
                if v is not None and math.isfinite(float(v)):
                    values[int(h)] = float(v)
            if not values:
                continue
            cells.append({
                "key": key,
                "score": m.group("score"),
                "ratio": float(ratio_key),
                "cycle": int(m.group("t")) if m.group("t") else None,
                "distortion": values,
            })
    if not cells:
        raise SystemExit("report contains no finite distortion cells")
    return cells


def _common_horizon(cells):
    common = set(cells[0]["distortion"])
    for cell in cells[1:]:
        common &= set(cell["distortion"])
    if not common:
        raise SystemExit("cells share no finite distortion horizon")
    return max(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--horizon", default="auto",
                    help="horizon step, or 'auto' for the largest common one")
    args = ap.parse_args()

    report = json.load(open(args.report))
    branches = report.get("branch_results") or {}
    if not branches:
        raise SystemExit("report has no branch_results")
    branch = branches[sorted(branches)[0]]
    cells = _cells(branch)
    if args.horizon == "auto":
        horizon = _common_horizon(cells)
    else:
        horizon = int(args.horizon)

    ranked = []
    for cell in cells:
        value = cell["distortion"].get(horizon)
        if value is None:
            continue
        ranked.append((value, cell))
    if not ranked:
        raise SystemExit(f"no cell has distortion at horizon {horizon}")
    ranked.sort(key=lambda item: (
        item[0], item[1]["score"], item[1]["ratio"],
        -1 if item[1]["cycle"] is None else item[1]["cycle"]))

    best_value, best = ranked[0]
    for value, cell in ranked[:10]:
        print(f"{value:.9g}\t{cell['score']}\tratio={cell['ratio']:g}"
              f"\tT={cell['cycle']}", file=sys.stderr)
    print(f"BEST_SCORE={best['score']}")
    print(f"BEST_RATIO={best['ratio']:g}")
    if best["cycle"] is not None:
        print(f"BEST_CYCLE={best['cycle']}")
    print(f"BEST_HORIZON={horizon}")
    print(f"BEST_DISTORTION={best_value:.12g}")
    print(f"BEST_CELL={best['key']}")


if __name__ == "__main__":
    main()
