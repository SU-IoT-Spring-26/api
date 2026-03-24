#!/usr/bin/env python3
"""
Compare API occupancy estimates to ground truth.

Reads a ground-truth file (CSV: timestamp, sensor_id, actual_count) and occupancy
history from local JSONL under THERMAL_DATA_DIR (or --data-dir). Aligns each
ground-truth point to the nearest API readings within a time window and computes
MAE, accuracy within 1, etc.

Ground-truth CSV format (header optional):
  timestamp,sensor_id,actual_count
  2025-02-07T14:30:00, living-room, 2
  ...

Timestamps are parsed with datetime.fromisoformat() (e.g. ISO 8601).

Use ``--field`` or ``--compare-fields`` to score ``occupancy`` (smoothed),
``occupancy_effective_raw``, or ``occupancy_raw_instant`` from JSONL lines.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def parse_ts(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def load_ground_truth(path: Path) -> List[Tuple[datetime, str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return rows
        # If first row looks like header, skip it
        if first and len(first) >= 3 and str(first[0]).lower().startswith("timestamp"):
            pass
        else:
            # First row is data
            ts = parse_ts(first[0] if len(first) > 0 else "")
            sid = (first[1] or "").strip() if len(first) > 1 else ""
            try:
                count = int(first[2]) if len(first) > 2 else 0
            except ValueError:
                count = 0
            if ts and sid is not None:
                rows.append((ts, sid, count))
        for row in reader:
            if len(row) < 3:
                continue
            ts = parse_ts(row[0])
            sid = (row[1] or "").strip()
            try:
                count = int(row[2])
            except ValueError:
                continue
            if ts is None:
                continue
            rows.append((ts, sid, count))
    return rows


def occupancy_from_entry(entry: dict, field: str) -> int:
    """Read occupancy-like integer from a JSONL entry; fall back to ``occupancy``."""
    if field in entry and entry[field] is not None:
        try:
            return int(entry[field])
        except (TypeError, ValueError):
            pass
    return int(entry.get("occupancy", 0))


def load_occupancy_history(data_dir: Path, sensor_id: Optional[str]) -> List[dict]:
    """Load all occupancy entries from JSONL files under data_dir, optionally filter by sensor_id."""
    entries = []
    for path in sorted(data_dir.glob("occupancy_*.jsonl")):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if sensor_id is not None and entry.get("sensor_id") != sensor_id:
                    continue
                ts = entry.get("timestamp")
                if ts:
                    try:
                        entry["_dt"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    entries.append(entry)
    return entries


def align_and_compare(
    ground_truth: list[tuple[datetime, str, int]],
    history: list[dict],
    window_seconds: int,
    field: str = "occupancy",
) -> List[Tuple[float, float, int, int]]:
    """For each ground-truth (ts, sensor_id, actual), find API estimate within window.

    ``field`` selects which logged column to compare (e.g. ``occupancy`` smoothed,
    ``occupancy_effective_raw``, ``occupancy_raw_instant``).
    """
    results = []
    for ts, sid, actual in ground_truth:
        candidates = [
            e for e in history
            if e.get("sensor_id") == sid
            and abs((e["_dt"] - ts).total_seconds()) <= window_seconds
        ]
        if not candidates:
            results.append((float("nan"), float("nan"), actual, -1))
            continue
        avg_est = sum(occupancy_from_entry(e, field) for e in candidates) / len(candidates)
        results.append((actual, avg_est, actual, int(round(avg_est))))
    return results


def metrics_from_align_results(results: List[Tuple[float, float, int, int]]) -> Optional[Tuple[int, int, float, float]]:
    """Return (n_matched, n_gt, mae, acc_within_1) or None if no matches."""
    with_estimate = [(a, e) for (_, _, a, e) in results if e >= 0]
    if not with_estimate:
        return None
    n_gt = len(results)
    errors = [e - a for a, e in with_estimate]
    mae = sum(abs(x) for x in errors) / len(errors)
    accuracy_within_1 = sum(1 for a, e in with_estimate if abs(e - a) <= 1) / len(with_estimate)
    return len(with_estimate), n_gt, mae, accuracy_within_1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare occupancy API estimates to ground truth.")
    parser.add_argument("ground_truth", type=Path, help="CSV file: timestamp,sensor_id,actual_count")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with occupancy_YYYYMMDD.jsonl (default: THERMAL_DATA_DIR or ./thermal_data)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Time window in seconds to match API reading to ground truth (default: 120)",
    )
    parser.add_argument(
        "--field",
        default="occupancy",
        help="Occupancy JSONL field to compare (default: occupancy). "
        "Common: occupancy_effective_raw, occupancy_raw_instant",
    )
    parser.add_argument(
        "--compare-fields",
        default=None,
        metavar="LIST",
        help="Comma-separated fields to report in one run (overrides --field for metrics), e.g. "
        "occupancy,occupancy_effective_raw,occupancy_raw_instant",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-row comparison")
    args = parser.parse_args()

    data_dir = args.data_dir or Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if not args.ground_truth.is_file():
        print(f"Error: ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    gt = load_ground_truth(args.ground_truth)
    if not gt:
        print("Error: no valid ground truth rows", file=sys.stderr)
        return 1

    history = load_occupancy_history(data_dir, sensor_id=None)

    if args.compare_fields:
        fields = [f.strip() for f in args.compare_fields.split(",") if f.strip()]
        if not fields:
            fields = ["occupancy"]
        print(f"Ground truth rows: {len(gt)}")
        any_ok = False
        for field in fields:
            results = align_and_compare(gt, history, args.window, field=field)
            m = metrics_from_align_results(results)
            if m is None:
                print(f"\n[{field}] No API readings in window.")
                continue
            any_ok = True
            n_m, n_gt, mae, acc1 = m
            print(f"\n[{field}] matched={n_m}/{n_gt}  MAE={mae:.3f}  accuracy_within_1={acc1:.2%}")
        if not any_ok:
            return 1
    else:
        results = align_and_compare(gt, history, args.window, field=args.field)
        m = metrics_from_align_results(results)
        if m is None:
            print("No API readings found within time window of any ground truth row.")
            return 1
        n_m, n_gt, mae, accuracy_within_1 = m
        print(f"Ground truth rows: {len(gt)}")
        print(f"Field: {args.field}")
        print(f"Rows with API estimate in window: {n_m}")
        print(f"MAE: {mae:.3f}")
        print(f"Accuracy (within 1): {accuracy_within_1:.2%}")

    if args.verbose:
        fld = args.field
        results = align_and_compare(gt, history, args.window, field=fld)
        print(f"\nPer-row (actual, estimated) [{fld}]:")
        for (_, _, actual, est) in results:
            if est >= 0:
                print(f"  {actual} -> {est}  (error {est - actual})")
            else:
                print(f"  {actual} -> (no match)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
