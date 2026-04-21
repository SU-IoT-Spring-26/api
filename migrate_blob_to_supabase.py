#!/usr/bin/env python3
"""
One-time migration: backfill historical thermal readings from blob storage
(and local disk) into the Supabase `readings` table.

Re-run safe — uses upsert on (sensor_id, received_at). Run a second time
after deploying new code to fill the gap window during deployment.

Usage:
    # Dry run first to see what would happen
    python migrate_blob_to_supabase.py --dry-run

    # Then for real
    python migrate_blob_to_supabase.py
"""
import argparse
import sys
from pathlib import Path

# Ensure imports resolve relative to this script's directory.
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Importing main.py runs its module-level code, including create_client().
# So your SUPABASE_URL and SUPABASE_SECRET_KEY env vars must be set first.
from main import (
    supa_client,
    _iter_thermal_files,
    _list_blob_thermal_names,
    _ensure_local_copy,
    _read_json_payload,
    _parse_thermal_blob_meta,
    estimate_occupancy,
    convert_numpy_types,
    _load_thermal_background,
    _load_stationary_background,
    thermal_background_by_sensor,
    stationary_thermal_background_by_sensor,
)

BATCH_SIZE = 100  # small batches; you only have a few hundred readings total


def get_valid_sensor_ids() -> set:
    """Pull current sensor IDs from the sensors table — anything not in this
    set is from a deprecated/renamed sensor and gets skipped."""
    result = supa_client.table("sensors").select("sensor_id").execute()
    return {row["sensor_id"] for row in (result.data or [])}


def gather_all_files() -> list:
    """Collect every thermal file from local disk and blob storage, deduped
    by filename. Returns a list of (source_type, ref) tuples."""
    seen = set()
    out = []

    for path in _iter_thermal_files():
        if path.name not in seen:
            seen.add(path.name)
            out.append(("local", path))

    for filename in _list_blob_thermal_names():
        if filename not in seen:
            seen.add(filename)
            out.append(("blob", filename))

    return out


def load_payload(source: str, ref) -> dict:
    """Load a JSON payload from either a local Path or a blob filename."""
    if source == "local":
        return _read_json_payload(ref)
    local_path = _ensure_local_copy(ref)
    if local_path is None:
        raise FileNotFoundError(f"Could not retrieve blob {ref}")
    return _read_json_payload(local_path)


def build_row(payload: dict, filename: str) -> dict | None:
    """Turn a payload into a readings table row. Returns None if invalid."""
    # sensor_id can be at top level or inside `data`
    sid = payload.get("sensor_id") or payload.get("data", {}).get("sensor_id")
    if not sid:
        return None

    # Prefer the timestamp from the payload; fall back to parsing the filename
    ts = payload.get("timestamp")
    if not ts and filename:
        _, ts = _parse_thermal_blob_meta(filename)
    if not ts:
        return None

    data = payload.get("data") or {}
    if "t" not in data or not data["t"]:
        return None

    # estimate_occupancy needs sensor_id on the data dict and pre-loaded
    # background frames per sensor (cached after first load).
    data_for_estimation = dict(data)
    data_for_estimation["sensor_id"] = sid

    if sid not in thermal_background_by_sensor:
        try:
            _load_thermal_background(sid)
        except Exception:
            pass  # background is optional; estimate_occupancy will cope
    if sid not in stationary_thermal_background_by_sensor:
        try:
            _load_stationary_background(sid)
        except Exception:
            pass

    try:
        occ = convert_numpy_types(
            estimate_occupancy(data_for_estimation, sensor_id=sid)
        )
    except Exception as e:
        print(f"  estimate_occupancy failed for {sid} @ {ts}: {e}")
        return None

    return {
        "sensor_id": sid,
        "recorded_at": ts,  # rename to "recorded_at" if your column uses that
        "avg_temp": float(np.mean(data["t"])),
        "occupancy_count": int(occ.get("occupancy", 0)),
        "pixel_data": list(data["t"]),  # ensure plain list, not numpy
    }


def flush_batch(batch: list, dry_run: bool) -> int:
    """Upsert a batch. On error, retry one-by-one to surface bad rows.
    Returns the number of rows successfully written."""
    if not batch:
        return 0
    if dry_run:
        return len(batch)
    try:
        supa_client.table("readings").upsert(
            batch, on_conflict="sensor_id,received_at"
        ).execute()
        return len(batch)
    except Exception as e:
        print(f"Batch upsert failed ({e}); retrying one-by-one...")
        ok = 0
        for row in batch:
            try:
                supa_client.table("readings").upsert(
                    row, on_conflict="sensor_id,received_at"
                ).execute()
                ok += 1
            except Exception as e2:
                print(f"  Row failed ({row['sensor_id']} @ {row['received_at']}): {e2}")
        return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process files but skip the actual database writes.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files (sanity check).")
    args = parser.parse_args()

    print("Fetching valid sensor IDs from Supabase...")
    valid_ids = get_valid_sensor_ids()
    if not valid_ids:
        print("ERROR: No sensors in the sensors table. Insert rows there first.")
        sys.exit(1)
    print(f"  {len(valid_ids)} sensor(s) registered: {sorted(valid_ids)}")

    print("Gathering thermal files (local + blob)...")
    all_files = gather_all_files()
    print(f"  {len(all_files)} unique files found")

    if args.limit:
        all_files = all_files[: args.limit]
        print(f"  --limit set: processing first {len(all_files)} only")

    counts = {"written": 0, "skipped_stale_sensor": 0, "skipped_invalid": 0, "load_error": 0}
    batch = []

    for i, (source, ref) in enumerate(all_files, 1):
        if i % 50 == 0:
            print(f"  [{i}/{len(all_files)}] written={counts['written']} batch={len(batch)}")

        try:
            payload = load_payload(source, ref)
        except Exception as e:
            counts["load_error"] += 1
            print(f"  Could not load {ref}: {e}")
            continue

        # Cheap pre-filter: skip stale sensor IDs without doing the heavy
        # estimate_occupancy work. Saves time and avoids FK violations.
        sid = payload.get("sensor_id") or payload.get("data", {}).get("sensor_id")
        if sid not in valid_ids:
            counts["skipped_stale_sensor"] += 1
            continue

        filename = ref.name if source == "local" else ref
        row = build_row(payload, filename)
        if row is None:
            counts["skipped_invalid"] += 1
            continue

        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            counts["written"] += flush_batch(batch, args.dry_run)
            batch = []

    counts["written"] += flush_batch(batch, args.dry_run)

    print("\n=== Migration complete ===")
    if args.dry_run:
        print("  (DRY RUN — no rows actually written)")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Verify by counting rows per sensor
    if not args.dry_run:
        print("\nFinal row counts in readings table:")
        for sid in sorted(valid_ids):
            r = supa_client.table("readings") \
                .select("*", count="exact", head=True) \
                .eq("sensor_id", sid).execute()
            print(f"  {sid}: {r.count}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    from main import _get_blob_container, _list_blob_thermal_names
    container = _get_blob_container()

    prefixes = set()
    for blob in container.list_blobs():
        prefix = blob.name.split("/")[0] if "/" in blob.name else "(root)"
        prefixes.add(prefix)
    print("Prefixes in container:", prefixes)
    print("container:", container)
    print("container name:", container.container_name if container else "n/a")

    files = _list_blob_thermal_names()
    print(f"files matching thermal/ prefix: {len(files)}")
    if files:
        print("first 5:", files[:5])
    else:
        # If no thermal/ prefix matches, list everything to see what IS there
        if container:
            print("\nNo thermal/ matches. Listing ALL blobs in container (first 20):")
            for i, blob in enumerate(container.list_blobs()):
                if i >= 20:
                    print("  ...")
                    break
                print(f"  {blob.name}")
    # main()