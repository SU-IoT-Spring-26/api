#!/usr/bin/env python3
"""
Compare fever-related flags from occupancy JSONL to binary ground truth.

Ground-truth CSV (header optional):
  timestamp,sensor_id,fever
  2025-02-07T14:30:00,living-room,1

''fever'' is 0/1 (no fever / fever). Compared fields (see --field):
  any_fever (consecutive-frame gated), any_fever_raw, fever_count > 0.
"""

from __future__ import annotations

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


def load_fever_gt(path: Path) -> List[Tuple[datetime, str, bool]]:
    rows: List[Tuple[datetime, str, bool]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return rows
        if first and len(first) >= 3 and str(first[0]).lower().startswith("timestamp"):
            pass
        else:
            ts = parse_ts(first[0] if len(first) > 0 else "")
            sid = (first[1] or "").strip() if len(first) > 1 else ""
            try:
                fv = int(first[2]) if len(first) > 2 else 0
            except ValueError:
                fv = 0
            if ts and sid:
                rows.append((ts, sid, fv != 0))
        for row in reader:
            if len(row) < 3:
                continue
            ts = parse_ts(row[0])
            sid = (row[1] or "").strip()
            try:
                fv = int(row[2])
            except ValueError:
                continue
            if ts is None:
                continue
            rows.append((ts, sid, fv != 0))
    return rows


def fever_from_entry(entry: dict, field: str) -> bool:
    if field == "fever_count_positive":
        try:
            return int(entry.get("fever_count", 0)) > 0
        except (TypeError, ValueError):
            return False
    v = entry.get(field, False)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return bool(v)


def load_occupancy_history(data_dir: Path, sensor_id: Optional[str]) -> List[dict]:
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


def align_and_score(
    gt: List[Tuple[datetime, str, bool]],
    history: List[dict],
    window_seconds: int,
    field: str,
) -> Optional[Tuple[int, float, float]]:
    """Return (matched, accuracy, balanced_accuracy) or None."""
    tp = fp = tn = fn = 0
    matched = 0
    for ts, sid, actual_hot in gt:
        candidates = [
            e
            for e in history
            if e.get("sensor_id") == sid and abs((e["_dt"] - ts).total_seconds()) <= window_seconds
        ]
        if not candidates:
            continue
        matched += 1
        pred_hot = any(fever_from_entry(e, field) for e in candidates)
        if actual_hot and pred_hot:
            tp += 1
        elif actual_hot and not pred_hot:
            fn += 1
        elif not actual_hot and pred_hot:
            fp += 1
        else:
            tn += 1
    if matched == 0:
        return None
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    bal = (sens + spec) / 2.0
    return matched, acc, bal


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fever flags from occupancy JSONL to binary ground truth.")
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument(
        "--field",
        default="any_fever",
        choices=("any_fever", "any_fever_raw", "fever_count_positive"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1
    if not args.ground_truth.is_file():
        print(f"Error: ground truth not found: {args.ground_truth}", file=sys.stderr)
        return 1

    gt = load_fever_gt(args.ground_truth)
    if not gt:
        print("Error: no valid ground-truth rows", file=sys.stderr)
        return 1

    history = load_occupancy_history(data_dir, sensor_id=None)
    out = align_and_score(gt, history, args.window, args.field)
    if out is None:
        print("No occupancy readings aligned to ground truth in window.", file=sys.stderr)
        return 1
    matched, acc, bal = out
    print(f"Ground truth rows: {len(gt)}")
    print(f"Aligned rows: {matched}")
    print(f"Field: {args.field}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Balanced accuracy (sensitivity/spec): {bal:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
