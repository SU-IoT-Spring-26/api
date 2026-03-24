#!/usr/bin/env python3
"""
Replay stored compact thermal frames through the occupancy pipeline offline.

Reads ``thermal_*_compact.json`` files (same shape as written by the API) in
chronological order and runs ``estimate_occupancy`` plus optional signal
processing and background updates, without writing thermal/occupancy files.

Typical use: regenerate predictions after changing detection constants, or
build synthetic JSONL for calibration (see ``calibrate_occupancy_thresholds.py``).

Run from repository root (so ``import main`` resolves):

    python scripts/replay_thermal_occupancy.py --data-dir ./thermal_data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def reset_pipeline_state(m: Any) -> None:
    for name in (
        "thermal_background_by_sensor",
        "empty_frame_count_by_sensor",
        "last_empty_frame_thermal_by_sensor",
        "occupancy_raw_history_by_sensor",
        "last_frame_median_by_sensor",
        "last_raw_occupancy_by_sensor",
        "last_smoothed_occupancy_by_sensor",
        "fever_consecutive_by_sensor",
    ):
        getattr(m, name).clear()


def list_compact_paths(data_dir: Path, sensor_id: Optional[str]) -> List[Path]:
    paths = [p for p in data_dir.glob("thermal_*_compact.json") if p.is_file()]
    if sensor_id is not None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sensor_id))[:64]
        paths = [p for p in paths if f"thermal_{safe}_" in p.name]
    return sorted(paths, key=lambda p: p.name)


def load_payload_timestamp(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return str(payload.get("timestamp") or "")
    except Exception:
        return ""


def replay_compact_frames(
    data_dir: Path,
    *,
    sensor_id: Optional[str] = None,
    fresh: bool = True,
    load_saved_background: bool = False,
    pipeline: str = "full",
    persist_background: bool = False,
    quiet_import: bool = True,
) -> List[Dict[str, Any]]:
    """
    ``pipeline``:
      - ``estimate_only``: clustering only (no smoothing, hysteresis, or background).
      - ``full``: estimate + ``apply_occupancy_signal_processing`` + ``_maybe_update_thermal_background``.
    """
    _ensure_repo_path()
    import main as m  # noqa: PLC0415

    data_dir = data_dir.resolve()
    m.DATA_DIR = data_dir

    old_save = m.SAVE_DATA
    old_save_bg = m._save_thermal_background
    m.SAVE_DATA = False
    if not persist_background:
        m._save_thermal_background = lambda _sid: None  # type: ignore[assignment]

    if fresh:
        reset_pipeline_state(m)

    rows_out: List[Dict[str, Any]] = []
    paths = list_compact_paths(data_dir, sensor_id)
    if not paths and not quiet_import:
        print("No thermal_*_compact.json files found.", file=sys.stderr)

    loaded_bg: set[str] = set()

    try:
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                continue
            if payload.get("format") != "compact":
                continue
            data = payload.get("data") or {}
            sid = payload.get("sensor_id") or data.get("sensor_id") or "unknown"
            ts = str(payload.get("timestamp") or "")

            if load_saved_background and sid not in loaded_bg:
                m._load_thermal_background(sid)
                loaded_bg.add(sid)

            occ = m.estimate_occupancy(data, sensor_id=sid)
            occ["sensor_id"] = sid

            if pipeline == "full":
                try:
                    arr = m.thermal_data_to_array(data)
                    m.apply_occupancy_signal_processing(sid, occ, arr)
                    m._maybe_update_thermal_background(
                        sid, arr, int(occ.get("occupancy_effective_raw", occ.get("occupancy", 0)))
                    )
                except Exception:
                    pass
            cls_occ = int(occ.get("occupancy", 0))
            if pipeline != "full":
                occ["occupancy_raw_instant"] = cls_occ
                occ["occupancy_effective_raw"] = cls_occ
                occ["frame_valid"] = True
                occ["fever_consecutive_frames"] = 0
                occ["any_fever_raw"] = bool(occ.get("any_fever", False))

            row: Dict[str, Any] = {
                "timestamp": ts or load_payload_timestamp(path),
                "sensor_id": sid,
                "source_file": path.name,
                "occupancy": int(occ.get("occupancy", 0)),
                "occupancy_raw_instant": occ.get("occupancy_raw_instant", cls_occ),
                "occupancy_effective_raw": occ.get("occupancy_effective_raw", cls_occ),
                "frame_valid": occ.get("frame_valid", True),
                "room_temperature": occ.get("room_temperature"),
                "fever_count": int(occ.get("fever_count", 0)),
                "elevated_count": int(occ.get("elevated_count", 0)),
                "any_fever": bool(occ.get("any_fever", False)),
                "any_fever_raw": bool(occ.get("any_fever_raw", occ.get("any_fever", False))),
                "any_elevated": bool(occ.get("any_elevated", False)),
                "fever_consecutive_frames": int(occ.get("fever_consecutive_frames", 0)),
            }
            rows_out.append(row)
    finally:
        m.SAVE_DATA = old_save
        m._save_thermal_background = old_save_bg

    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay compact thermal JSON files through the occupancy pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with thermal_*_compact.json (default: THERMAL_DATA_DIR or ./thermal_data)",
    )
    parser.add_argument("--sensor-id", default=None, help="Only replay files for this sensor_id")
    parser.add_argument(
        "--pipeline",
        choices=("full", "estimate_only"),
        default="full",
        help="full = smoothing + background updates; estimate_only = clustering only",
    )
    parser.add_argument(
        "--load-background",
        action="store_true",
        help="When not using --fresh, load background_*.npy from data-dir before each sensor's frames",
    )
    parser.add_argument(
        "--no-fresh",
        dest="fresh",
        action="store_false",
        help="Do not reset pipeline state (smoothers, backgrounds in memory); default is a clean replay",
    )
    parser.set_defaults(fresh=True)
    parser.add_argument(
        "--persist-background",
        action="store_true",
        help="Allow writing background_*.npy during replay (default: off)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write one row per frame to this CSV path",
    )
    args = parser.parse_args()

    import os

    data_dir = args.data_dir or Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    rows = replay_compact_frames(
        data_dir,
        sensor_id=args.sensor_id,
        fresh=args.fresh,
        load_saved_background=args.load_background,
        pipeline=args.pipeline,
        persist_background=args.persist_background,
        quiet_import=False,
    )
    print(f"Replayed {len(rows)} compact frame(s).")
    if args.output_csv:
        if not rows:
            print("No rows to write.", file=sys.stderr)
            return 1
        fieldnames = list(rows[0].keys())
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
