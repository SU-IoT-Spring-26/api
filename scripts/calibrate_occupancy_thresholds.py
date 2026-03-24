#!/usr/bin/env python3
"""
Grid-search detection parameters by replaying stored thermal frames and scoring
against a ground-truth occupancy CSV.

Uses the same alignment and metrics as ``compare_occupancy_accuracy.py``. Mutates
``main`` module globals for each trial (``ROOM_TEMP_THRESHOLD``, ``MIN_CLUSTER_SIZE``, …)
then runs ``replay_thermal_occupancy.replay_compact_frames``. Intended for offline
experimentation with small grids.

Example (from repository root):

    python scripts/calibrate_occupancy_thresholds.py \\
        path/to/ground_truth.csv \\
        --data-dir ./thermal_data \\
        --room-threshold-grid 0.3 0.45 0.6 \\
        --field occupancy_effective_raw
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent


def _paths() -> None:
    for p in (REPO_ROOT, SCRIPTS_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def rows_to_history(rows: List[Dict[str, Any]]) -> List[dict]:
    history = []
    for r in rows:
        ts = r.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        e = dict(r)
        e["_dt"] = dt
        history.append(e)
    return history


def main() -> int:
    _paths()
    import main as m  # noqa: PLC0415
    from compare_occupancy_accuracy import (  # noqa: PLC0415
        align_and_compare,
        load_ground_truth,
        metrics_from_align_results,
    )
    from replay_thermal_occupancy import replay_compact_frames  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Replay compact thermal files under varying main.* thresholds; score vs ground-truth CSV."
    )
    parser.add_argument("ground_truth", type=Path, help="CSV: timestamp,sensor_id,actual_count")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--sensor-id", default=None, help="Replay only this sensor's compact files")
    parser.add_argument(
        "--field",
        default="occupancy",
        help="Replay output field to compare (occupancy, occupancy_effective_raw, occupancy_raw_instant)",
    )
    parser.add_argument(
        "--pipeline",
        choices=("full", "estimate_only"),
        default="full",
        help="Replay pipeline mode (default: full)",
    )
    parser.add_argument(
        "--room-threshold-grid",
        type=float,
        nargs="*",
        default=None,
        help="Values for main.ROOM_TEMP_THRESHOLD (default: single run with current code constants)",
    )
    parser.add_argument(
        "--min-cluster-grid",
        type=int,
        nargs="*",
        default=None,
        help="Values for main.MIN_CLUSTER_SIZE",
    )
    parser.add_argument(
        "--load-background",
        action="store_true",
        help="Load background_*.npy from data-dir when replaying (see replay script)",
    )
    args = parser.parse_args()

    import os

    data_dir = args.data_dir or Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if not args.ground_truth.is_file():
        print(f"Error: ground truth not found: {args.ground_truth}", file=sys.stderr)
        return 1

    gt = load_ground_truth(args.ground_truth)
    if not gt:
        print("Error: no valid ground-truth rows", file=sys.stderr)
        return 1

    rt_grid: List[Optional[float]]
    if args.room_threshold_grid is not None and len(args.room_threshold_grid) > 0:
        rt_grid = [float(x) for x in args.room_threshold_grid]
    else:
        rt_grid = [None]

    mc_grid: List[Optional[int]]
    if args.min_cluster_grid is not None and len(args.min_cluster_grid) > 0:
        mc_grid = [int(x) for x in args.min_cluster_grid]
    else:
        mc_grid = [None]

    baseline_rt = m.ROOM_TEMP_THRESHOLD
    baseline_mc = m.MIN_CLUSTER_SIZE

    best: Tuple[float, str, Optional[Tuple[float, int]]] = (float("inf"), "", None)

    for rt in rt_grid:
        for mc in mc_grid:
            if rt is not None:
                m.ROOM_TEMP_THRESHOLD = float(rt)
            if mc is not None:
                m.MIN_CLUSTER_SIZE = int(mc)
            label_parts = []
            if rt is not None:
                label_parts.append(f"ROOM_TEMP_THRESHOLD={rt}")
            if mc is not None:
                label_parts.append(f"MIN_CLUSTER_SIZE={mc}")
            label = "; ".join(label_parts) if label_parts else "defaults"

            rows = replay_compact_frames(
                data_dir,
                sensor_id=args.sensor_id,
                fresh=True,
                load_saved_background=args.load_background,
                pipeline=args.pipeline,
                persist_background=False,
                quiet_import=True,
            )
            history = rows_to_history(rows)
            results = align_and_compare(gt, history, args.window, field=args.field)
            met = metrics_from_align_results(results)
            if met is None:
                print(f"[{label}] no aligned readings")
                continue
            n_m, n_gt, mae, acc1 = met
            print(f"[{label}] matched={n_m}/{n_gt}  MAE={mae:.4f}  acc_within_1={acc1:.2%}")
            if mae < best[0]:
                best = (mae, label, (rt, mc))

    m.ROOM_TEMP_THRESHOLD = baseline_rt
    m.MIN_CLUSTER_SIZE = baseline_mc

    if best[2] is not None:
        print(f"\nBest MAE {best[0]:.4f} ({best[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
