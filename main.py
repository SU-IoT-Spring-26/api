#!/usr/bin/env python3
"""
Occupancy API – FastAPI data reception and storage for thermal/occupancy data.
Designed for Azure App Service (resource group: occupancy-rg, app: occupancy-api).
Stores data locally and optionally to Azure Blob Storage when configured.
"""

import bisect
import json
import os
import gzip
import threading
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from scipy.ndimage import label

# Optional ML inference engine (gracefully absent if onnxruntime not installed)
try:
    from ml import MLInferenceEngine as _MLInferenceEngine
    _ml_engine = _MLInferenceEngine()
except Exception:
    _ml_engine = None  # type: ignore[assignment]

# Optional Azure Blob Storage (only used if AZURE_STORAGE_CONNECTION_STRING is set)
# _blob_container_client: container client if connected, False if init failed, None if not tried
_blob_container_client: Any = None

# Optional Azure SQL connection cache.
# None = not yet attempted, False = permanently disabled (misconfiguration or import error), object = live connection
_sql_connection: Any = None


def _get_blob_container():
    """Lazy-init Azure Blob container client from env. Returns None if not configured or init failed."""
    global _blob_container_client
    if _blob_container_client is False:
        return None  # already tried and failed
    if _blob_container_client is not None:
        return _blob_container_client
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str or not conn_str.strip():
        return None
    try:
        from azure.storage.blob import BlobServiceClient

        container_name = os.environ.get("AZURE_STORAGE_CONTAINER_NAME", "iotoccupancydata")
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        _blob_container_client = blob_service.get_container_client(container_name)
        _blob_container_client.get_container_properties()  # verify access
        return _blob_container_client
    except Exception as e:
        print(f"Azure Blob Storage init failed (blob uploads disabled): {e}")
        _blob_container_client = False
        # If local storage was suppressed because blob was configured, re-enable it as fallback
        # so sensor uploads are not silently dropped.
        global SAVE_LOCAL_DATA
        if not SAVE_LOCAL_DATA:
            print("WARNING: Blob storage unavailable and local storage was disabled — enabling local storage as fallback.")
            SAVE_LOCAL_DATA = True
        return None


def _upload_blob(blob_name: str, data: bytes, content_type: str = "application/json") -> None:
    """Upload a block blob. No-op if Blob Storage is not configured."""
    container = _get_blob_container()
    if container is None:
        return
    try:
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True, content_settings={"content_type": content_type})
    except Exception as e:
        print(f"Azure Blob upload failed ({blob_name}): {e}")


def _append_to_blob(blob_name: str, line: str) -> None:
    """Append a line to an append blob (create if not exists). No-op if Blob Storage is not configured."""
    container = _get_blob_container()
    if container is None:
        return
    try:
        blob_client = container.get_blob_client(blob_name)
        if not blob_client.exists():
            blob_client.create_append_blob()
        blob_client.append_block(line.encode("utf-8"))
    except Exception as e:
        print(f"Azure Blob append failed ({blob_name}): {e}")

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _restore_state_from_disk()
    if _ml_engine is not None:
        _ml_engine.load(blob_container=_get_blob_container())
    yield


app = FastAPI(
    title="Occupancy API",
    description="Receive and store thermal camera data; estimate and query occupancy.",
    version="1.0.0",
    lifespan=_lifespan,
)


def _restore_state_from_disk() -> None:
    """Reload the most recent thermal frame and occupancy result per sensor from disk or Blob after a restart."""
    global latest_thermal_data, last_update_time, latest_occupancy

    # --- Thermal frames: latest frame per sensor from local files ---
    best_thermal: Dict[str, Tuple[str, dict]] = {}  # sensor_id -> (timestamp_str, data)
    for p in _iter_thermal_files():
        try:
            payload = _read_json_payload(p)
            sid = payload.get("sensor_id") or (payload.get("data") or {}).get("sensor_id")
            if not sid or sid in best_thermal:
                continue
            ts = payload.get("timestamp", "")
            data = payload.get("data") or {}
            if payload.get("format") == "compact" and "t" in data:
                expanded = expand_thermal_data(data)
                expanded["sensor_id"] = sid
                data = expanded
            if not data.get("pixels"):
                continue
            best_thermal[sid] = (ts, data)
        except Exception:
            continue

    # Blob fallback for thermal: pull the newest frame per sensor not found locally.
    seen_safe_ids = {_sanitize_sensor_id_for_filename(s) for s in best_thermal}
    for filename in _list_blob_thermal_names():
        safe_id, ts_iso = _parse_thermal_blob_meta(filename)
        if not safe_id or safe_id in seen_safe_ids:
            continue
        seen_safe_ids.add(safe_id)
        local_path = _ensure_local_copy(filename)
        if local_path is None:
            continue
        try:
            payload = _read_json_payload(local_path)
            sid = payload.get("sensor_id") or (payload.get("data") or {}).get("sensor_id") or safe_id
            if sid in best_thermal:
                continue
            ts = payload.get("timestamp", "") or ts_iso
            data = payload.get("data") or {}
            if payload.get("format") == "compact" and "t" in data:
                expanded = expand_thermal_data(data)
                expanded["sensor_id"] = sid
                data = expanded
            if not data.get("pixels"):
                continue
            best_thermal[sid] = (ts, data)
        except Exception:
            continue

    for sid, (ts, data) in best_thermal.items():
        latest_thermal_by_sensor[sid] = data
        last_update_time_by_sensor[sid] = ts
        if latest_thermal_data is None or ts > (last_update_time or ""):
            latest_thermal_data = data
            last_update_time = ts

    # --- Occupancy: latest entry per sensor from local daily jsonl files ---
    best_occ: Dict[str, Tuple[str, dict]] = {}  # sensor_id -> (timestamp_str, entry)
    if DATA_DIR.exists():
        for p in DATA_DIR.glob("occupancy_*.jsonl"):
            try:
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    sid = entry.get("sensor_id")
                    ts = entry.get("timestamp", "")
                    if not sid:
                        continue
                    if sid not in best_occ or ts > best_occ[sid][0]:
                        best_occ[sid] = (ts, entry)
            except Exception:
                continue

    # Blob fallback for occupancy: scan newest daily files until all thermal sensors are covered.
    sensors_needing_occ = set(best_thermal) - set(best_occ)
    if sensors_needing_occ:
        container = _get_blob_container()
        if container is not None:
            try:
                occ_blob_names = sorted(
                    [b.name for b in container.list_blobs(name_starts_with="occupancy/")
                     if b.name.endswith(".jsonl")],
                    reverse=True,
                )
                for blob_name in occ_blob_names[:5]:  # scan at most 5 daily files
                    if not sensors_needing_occ:
                        break
                    try:
                        raw = container.get_blob_client(blob_name).download_blob().readall()
                        for line in raw.decode("utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            sid = entry.get("sensor_id")
                            ts = entry.get("timestamp", "")
                            if not sid or sid not in sensors_needing_occ:
                                continue
                            if sid not in best_occ or ts > best_occ[sid][0]:
                                best_occ[sid] = (ts, entry)
                        sensors_needing_occ -= set(best_occ)
                    except Exception:
                        continue
            except Exception as e:
                print(f"Blob occupancy restore failed: {e}")

    for sid, (ts, entry) in best_occ.items():
        latest_occupancy_by_sensor[sid] = entry
        if sid in latest_thermal_by_sensor and ts > (last_update_time_by_sensor.get(sid) or ""):
            last_update_time_by_sensor[sid] = ts
        if latest_occupancy is None or ts > (last_update_time or ""):
            latest_occupancy = entry

    if best_thermal or best_occ:
        sensors = set(best_thermal) | set(best_occ)
        print(f"Restored state for {len(sensors)} sensor(s): {sorted(sensors)}")

    _load_ml_labels()  # called unconditionally — handles missing DATA_DIR gracefully
    _load_sensor_metadata()
    _load_ground_truth()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
_blob_conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
_save_to_blob_raw = os.environ.get("SAVE_TO_BLOB")
if _save_to_blob_raw is None:
    SAVE_TO_BLOB = bool(_blob_conn_str)
else:
    SAVE_TO_BLOB = _save_to_blob_raw.lower() in ("1", "true", "yes")

_save_local_data_raw = os.environ.get("SAVE_LOCAL_DATA")
if _save_local_data_raw is None:
    # Default to Blob persistence when Blob is configured.
    SAVE_LOCAL_DATA = not SAVE_TO_BLOB
else:
    SAVE_LOCAL_DATA = _save_local_data_raw.lower() in ("1", "true", "yes")

# Backward-compatible alias for local writes.
if "SAVE_THERMAL_DATA" in os.environ and _save_local_data_raw is None:
    SAVE_LOCAL_DATA = os.environ.get("SAVE_THERMAL_DATA", "true").lower() in ("1", "true", "yes")
# Backward-compatible alias expected by offline tooling.
SAVE_DATA = SAVE_LOCAL_DATA
SQL_CONNECTION_STRING = os.environ.get("SQL_CONNECTION_STRING", "").strip()
SAVE_TO_SQL = os.environ.get("SAVE_TO_SQL", "true").lower() in ("1", "true", "yes")

# Webhook alert: POST to this URL on fever detection (rising edge) or room overcapacity.
# Leave empty to disable. Fires in a background thread so it never delays the API response.
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
# Fraction of capacity that triggers an overcapacity alert (default 1.0 = at or above capacity).
ALERT_OVERCAPACITY_PCT = float(os.environ.get("ALERT_OVERCAPACITY_PCT", "1.0"))

# Optional API key protecting ground truth mutation endpoints (POST/DELETE).
# When set, callers must supply the matching X-API-Key header.
GROUND_TRUTH_API_KEY: Optional[str] = os.environ.get("GROUND_TRUTH_API_KEY") or None

# Occupancy detection parameters
MIN_HUMAN_TEMP = 30.0
MAX_HUMAN_TEMP = 45.0
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 200
ROOM_TEMP_THRESHOLD = float(os.environ.get("ROOM_TEMP_THRESHOLD", "0.5"))
FEVER_THRESHOLD_C = float(os.environ.get("FEVER_THRESHOLD_C", "37.5"))
FEVER_ELEVATED_THRESHOLD_C = float(os.environ.get("FEVER_ELEVATED_THRESHOLD_C", "37.0"))
FEVER_MIN_CONSECUTIVE_FRAMES = max(1, int(os.environ.get("FEVER_MIN_CONSECUTIVE_FRAMES", "2")))
# Cluster shape filters: reject elongated or sparse blobs (radiators, hot pipes, warm walls).
# Aspect ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h); >4 → likely not a person.
# Fill ratio = cluster pixels / bbox area; <0.15 → too diffuse to be a person.
MAX_CLUSTER_ASPECT_RATIO = float(os.environ.get("MAX_CLUSTER_ASPECT_RATIO", "4.0"))
MIN_CLUSTER_FILL_RATIO = float(os.environ.get("MIN_CLUSTER_FILL_RATIO", "0.15"))
# Ambient-temperature fever compensation: the MLX90640 reads skin surface temperature, which
# rises with room temperature.  For every 1°C the room is above FEVER_REFERENCE_ROOM_C the
# effective fever/elevated thresholds increase by FEVER_AMBIENT_COMP_C to reduce false positives.
FEVER_REFERENCE_ROOM_C = float(os.environ.get("FEVER_REFERENCE_ROOM_C", "22.0"))
FEVER_AMBIENT_COMP_C = float(os.environ.get("FEVER_AMBIENT_COMP_C", "0.1"))
# Sustained elevated temperature alert: flag a cluster when it has been in the elevated-temp
# range for this many consecutive frames (independent of the full-fever gate).
ELEVATED_MIN_CONSECUTIVE_FRAMES = max(1, int(os.environ.get("ELEVATED_MIN_CONSECUTIVE_FRAMES", "5")))

# MLX90640 subpage artifact detection
# The sensor alternates subpages; a moving body can produce a checkerboard double-image.
# Frames flagged as subpage-corrupted fall back to the previous valid frame (no rejection,
# so occupancy is never dropped to 0 due to sensor read timing).
SUBPAGE_ARTIFACT_ENABLED = os.environ.get("SUBPAGE_ARTIFACT_ENABLED", "true").lower() in ("1", "true", "yes")
# Minimum row-mean delta (°C) for a row-diff pair to be counted toward the checkerboard score.
# A clean frame has near-zero even/odd asymmetry; a subpage frame shows alternating-row offsets.
SUBPAGE_ROW_DIFF_THRESHOLD_C = float(os.environ.get("SUBPAGE_ROW_DIFF_THRESHOLD_C", "0.8"))
# Minimum fraction of row-diff pairs that must show the alternating sign pattern to flag as corrupted.
SUBPAGE_CHECKERBOARD_FRAC = float(os.environ.get("SUBPAGE_CHECKERBOARD_FRAC", "0.25"))

# Temporal occupancy smoothing and frame sanity (no ground truth required)
OCCUPANCY_SMOOTH_WINDOW = max(1, int(os.environ.get("OCCUPANCY_SMOOTH_WINDOW", "5")))
OCCUPANCY_HYSTERESIS_DELTA = max(0, int(os.environ.get("OCCUPANCY_HYSTERESIS_DELTA", "1")))
FRAME_ROOM_MEDIAN_MAX_JUMP_C = float(os.environ.get("FRAME_ROOM_MEDIAN_MAX_JUMP_C", "4.0"))
BACKGROUND_MAX_MEAN_ABS_DELTA_C = float(os.environ.get("BACKGROUND_MAX_MEAN_ABS_DELTA_C", "2.5"))

# Background subtraction (thermal): per-sensor background, updated when room empty
BACKGROUND_ALPHA = float(os.environ.get("BACKGROUND_ALPHA", "0.95"))  # EMA weight for existing background
BACKGROUND_MIN_FRAMES_EMPTY = int(os.environ.get("BACKGROUND_MIN_FRAMES_EMPTY", "3"))  # Consecutive empty frames to update
thermal_background_by_sensor: Dict[str, np.ndarray] = {}
empty_frame_count_by_sensor: Dict[str, int] = {}
last_empty_frame_thermal_by_sensor: Dict[str, np.ndarray] = {}

# Long-term stationary-source background: updated on EVERY frame regardless of occupancy.
# Uses a very slow EMA so heat sources present all day (radiators, monitors, warm walls)
# are gradually absorbed and no longer trigger false-positive people detections.
# Default α=0.9995 → time constant ≈ 2000 frames.  At 15 s/frame that is ~8 hours,
# meaning a source that has been on all day reaches ~98% absorption after 24 hours.
STATIONARY_BG_ALPHA = float(os.environ.get("STATIONARY_BG_ALPHA", "0.9995"))
# Persist to disk every N frames to avoid I/O on every upload (~15 min at default).
STATIONARY_BG_SAVE_INTERVAL = max(1, int(os.environ.get("STATIONARY_BG_SAVE_INTERVAL", "60")))
stationary_thermal_background_by_sensor: Dict[str, np.ndarray] = {}
_stationary_bg_frame_count: Dict[str, int] = {}

occupancy_raw_history_by_sensor: Dict[str, deque] = {}
last_frame_median_by_sensor: Dict[str, float] = {}
last_raw_occupancy_by_sensor: Dict[str, int] = {}
last_smoothed_occupancy_by_sensor: Dict[str, int] = {}
fever_consecutive_by_sensor: Dict[str, int] = {}
elevated_consecutive_by_sensor: Dict[str, int] = {}
# Per-sensor last clean (non-subpage-corrupted) frame for fallback interpolation
_last_clean_frame_by_sensor: Dict[str, np.ndarray] = {}

# In-memory latest state
latest_thermal_data: Optional[dict] = None
last_update_time: Optional[str] = None
latest_occupancy: Optional[dict] = None
latest_thermal_by_sensor: Dict[str, dict] = {}
last_update_time_by_sensor: Dict[str, str] = {}
latest_occupancy_by_sensor: Dict[str, dict] = {}
_data_counter = 0

if SAVE_LOCAL_DATA:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# ML labelling and training state
_ml_labels: Dict[str, dict] = {}  # filename → {file, occupancy, fever, ts}

# Ground truth labels for accuracy evaluation.
# Keyed by (timestamp_iso, sensor_id) as a string "ts|sid" for dedup.
_ground_truth: Dict[str, dict] = {}  # key → {timestamp, sensor_id, actual_occupancy, actual_fever, ts_added}
_ground_truth_lock = threading.Lock()
_ground_truth_persist_lock = threading.Lock()  # serialises file/blob writes
_ml_training_status: dict = {"state": "idle", "message": "", "ts": None, "log": []}
_ml_training_lock = threading.Lock()

# Per-sensor room metadata (room_name, capacity, building).
_sensor_metadata: Dict[str, dict] = {}  # sensor_id → {room_name, capacity, building, ts_updated}
_sensor_metadata_lock = threading.Lock()

# Previous alert state per sensor — used to detect fever/overcapacity rising edges.
_last_alert_state_by_sensor: Dict[str, dict] = {}  # sensor_id → {fever: bool, overcapacity: bool}
_alert_state_lock = threading.Lock()


def _load_ml_labels() -> None:
    """Load persisted ML frame labels from disk (or Blob fallback) into _ml_labels.

    On a fresh container the local file won't exist, so we fall back to
    downloading ml/labels.jsonl from Azure Blob Storage when configured.
    When SAVE_LOCAL_DATA is true, the downloaded content is cached to DATA_DIR
    so subsequent reads within the same container lifetime stay local.
    """
    global _ml_labels
    path = DATA_DIR / "ml_labels.jsonl"
    loaded: Dict[str, dict] = {}

    if path.exists():
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if "file" in entry:
                            loaded[entry["file"]] = entry
            _ml_labels = loaded
            print(f"Loaded {len(_ml_labels)} ML label(s)")
            return
        except Exception as e:
            print(f"Could not read ML labels from disk: {e}")

    container = _get_blob_container()
    if container is None:
        return
    try:
        blob_client = container.get_blob_client("ml/labels.jsonl")
        raw = blob_client.download_blob().readall().decode("utf-8")
        print("Restored ML labels from Azure Blob Storage")
        for line in raw.splitlines():
            line = line.strip()
            if line:
                entry = json.loads(line)
                if "file" in entry:
                    loaded[entry["file"]] = entry
        _ml_labels = loaded
        print(f"Loaded {len(_ml_labels)} ML label(s)")
        if SAVE_LOCAL_DATA:
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(raw)
            except Exception as e:
                print(f"Could not cache ML labels locally: {e}")
    except Exception as e:
        print(f"Could not restore ML labels from Blob: {e}")


def _persist_ml_labels() -> None:
    """Rewrite ml_labels.jsonl from the in-memory dict and sync to Blob.

    Local write and Blob upload are independent: Blob-only deployments
    (SAVE_LOCAL_DATA=false) still persist labels to Azure Blob.
    A consistent snapshot is taken under lock so concurrent label POSTs
    cannot cause RuntimeError or partial writes.
    """
    with _ml_training_lock:
        snapshot = list(_ml_labels.values())
    if SAVE_LOCAL_DATA:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "ml_labels.jsonl"
        try:
            with open(path, "w") as f:
                for entry in snapshot:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"Error saving ML labels locally: {e}")
    if SAVE_TO_BLOB:
        try:
            content = "".join(
                json.dumps(e, separators=(",", ":")) + "\n" for e in snapshot
            )
            _upload_blob("ml/labels.jsonl", content.encode("utf-8"), content_type="application/jsonlines")
        except Exception as e:
            print(f"Error uploading ML labels to Blob: {e}")


def _load_sensor_metadata() -> None:
    """Load persisted sensor metadata (room_name, capacity, building) from disk or Blob."""
    global _sensor_metadata
    raw: Optional[bytes] = None
    if DATA_DIR.exists():
        path = DATA_DIR / "sensor_metadata.json"
        if path.exists():
            try:
                raw = path.read_bytes()
            except Exception as e:
                print(f"Could not read sensor_metadata.json: {e}")
    if raw is None:
        container = _get_blob_container()
        if container is not None:
            try:
                raw = container.get_blob_client("sensor_metadata.json").download_blob().readall()
            except Exception:
                pass
    if raw:
        try:
            loaded = json.loads(raw.decode("utf-8"))
            if isinstance(loaded, dict):
                normalized: Dict[str, dict] = {}
                for sid, entry in loaded.items():
                    if not isinstance(entry, dict):
                        continue
                    clean: dict = {}
                    for field in ("room_name", "building", "ts_updated"):
                        if field in entry:
                            clean[field] = entry[field]
                    if "capacity" in entry:
                        try:
                            clean["capacity"] = int(entry["capacity"])
                        except (TypeError, ValueError):
                            pass  # drop unparseable capacity
                    normalized[sid] = clean
                with _sensor_metadata_lock:
                    _sensor_metadata.clear()
                    _sensor_metadata.update(normalized)
                print(f"Loaded metadata for {len(normalized)} sensor(s)")
        except Exception as e:
            print(f"Could not parse sensor_metadata.json: {e}")


def _persist_sensor_metadata() -> None:
    """Write sensor metadata to disk and Blob."""
    with _sensor_metadata_lock:
        snapshot = dict(_sensor_metadata)
    content = json.dumps(snapshot, indent=2).encode("utf-8")
    if SAVE_LOCAL_DATA:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            (DATA_DIR / "sensor_metadata.json").write_bytes(content)
        except Exception as e:
            print(f"Error saving sensor_metadata.json: {e}")
    if SAVE_TO_BLOB:
        try:
            _upload_blob("sensor_metadata.json", content, content_type="application/json")
        except Exception as e:
            print(f"Error uploading sensor_metadata.json: {e}")


def _fire_webhook(payload: dict) -> None:
    """POST alert payload to ALERT_WEBHOOK_URL. Called in a daemon thread."""
    if not ALERT_WEBHOOK_URL:
        return
    import urllib.request  # noqa: PLC0415
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[Alert] webhook OK ({resp.status}): {payload.get('alert_type')} for {payload.get('sensor_id')}")
    except Exception as e:
        print(f"[Alert] webhook failed: {e}")


def _maybe_fire_alerts(sensor_id: str, occupancy_result: dict, now_iso: str) -> None:
    """Detect rising edges on fever and overcapacity; fire webhook once per transition."""
    if not ALERT_WEBHOOK_URL:
        return

    curr_fever = bool(occupancy_result.get("any_fever", False))
    with _sensor_metadata_lock:
        meta = dict(_sensor_metadata.get(sensor_id, {}))
    capacity = meta.get("capacity")
    occ = int(occupancy_result.get("occupancy", 0))
    curr_overcapacity = (
        capacity is not None and capacity > 0 and occ / capacity >= ALERT_OVERCAPACITY_PCT
    )

    with _alert_state_lock:
        prev = _last_alert_state_by_sensor.get(sensor_id, {"fever": False, "overcapacity": False})
        new_fever = curr_fever and not prev["fever"]
        new_overcapacity = curr_overcapacity and not prev["overcapacity"]
        _last_alert_state_by_sensor[sensor_id] = {"fever": curr_fever, "overcapacity": curr_overcapacity}

    alerts = []
    if new_fever:
        alerts.append({
            "alert_type": "fever",
            "sensor_id": sensor_id,
            "room_name": meta.get("room_name"),
            "building": meta.get("building"),
            "timestamp": now_iso,
            "occupancy": occ,
            "fever_count": int(occupancy_result.get("fever_count", 0)),
            "effective_fever_threshold": occupancy_result.get("effective_fever_threshold"),
            "message": "Fever detected — room may require disinfection.",
        })
    if new_overcapacity:
        alerts.append({
            "alert_type": "overcapacity",
            "sensor_id": sensor_id,
            "room_name": meta.get("room_name"),
            "building": meta.get("building"),
            "timestamp": now_iso,
            "occupancy": occ,
            "capacity": capacity,
            "occupancy_pct": round(occ / capacity, 3),
            "message": f"Room at or above capacity ({occ}/{capacity}).",
        })

    for alert in alerts:
        threading.Thread(target=_fire_webhook, args=(alert,), daemon=True).start()


def _load_ground_truth() -> None:
    """Load persisted ground truth entries into _ground_truth from disk or Blob."""
    global _ground_truth
    loaded: Dict[str, dict] = {}

    def _ingest(line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except Exception:
            return
        ts = entry.get("timestamp", "")
        sid = entry.get("sensor_id", "")
        if ts and isinstance(sid, str):
            sid = sid.strip()
            if sid:
                loaded[f"{ts}|{sid}"] = entry

    if DATA_DIR.exists():
        path = DATA_DIR / "ground_truth.jsonl"
        if path.exists():
            try:
                for line in path.read_text().splitlines():
                    _ingest(line)
            except Exception as e:
                print(f"Could not load ground truth locally: {e}")

    if not loaded:
        container = _get_blob_container()
        if container is not None:
            try:
                raw = container.get_blob_client("occupancy/ground_truth.jsonl").download_blob().readall()
                for line in raw.decode("utf-8").splitlines():
                    _ingest(line)
            except Exception as e:
                print(f"Could not load ground truth from Blob: {e}")

    _ground_truth = loaded
    if loaded:
        print(f"Loaded {len(loaded)} ground truth entry/entries")


def _persist_ground_truth() -> None:
    """Rewrite ground_truth.jsonl from in-memory dict and sync to Blob."""
    with _ground_truth_lock:
        snapshot = list(_ground_truth.values())
    with _ground_truth_persist_lock:
        if SAVE_LOCAL_DATA:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / "ground_truth.jsonl"
            try:
                with open(path, "w") as f:
                    for entry in snapshot:
                        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            except Exception as e:
                print(f"Error saving ground truth locally: {e}")
        if SAVE_TO_BLOB:
            try:
                content = "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in snapshot)
                _upload_blob("occupancy/ground_truth.jsonl", content.encode("utf-8"), content_type="application/jsonlines")
            except Exception as e:
                print(f"Error uploading ground truth to Blob: {e}")


def _load_thermal_background(sensor_id: str) -> None:
    """Load persisted thermal background for a sensor from disk if present."""
    safe_id = _sanitize_sensor_id_for_filename(sensor_id)
    path = DATA_DIR / f"background_{safe_id}.npy"
    if path.exists():
        try:
            thermal_background_by_sensor[sensor_id] = np.load(path)
        except Exception:
            pass


def _save_thermal_background(sensor_id: str) -> None:
    """Persist thermal background for a sensor to disk."""
    if not SAVE_LOCAL_DATA:
        return
    arr = thermal_background_by_sensor.get(sensor_id)
    if arr is None:
        return
    safe_id = _sanitize_sensor_id_for_filename(sensor_id)
    path = DATA_DIR / f"background_{safe_id}.npy"
    try:
        np.save(path, arr)
    except Exception as e:
        print(f"Error saving thermal background ({sensor_id}): {e}")


def _load_stationary_background(sensor_id: str) -> None:
    """Load persisted stationary background for a sensor from disk if present."""
    safe_id = _sanitize_sensor_id_for_filename(sensor_id)
    path = DATA_DIR / f"background_stationary_{safe_id}.npy"
    if path.exists():
        try:
            stationary_thermal_background_by_sensor[sensor_id] = np.load(path)
        except Exception:
            pass


def _save_stationary_background(sensor_id: str) -> None:
    """Persist stationary background for a sensor to disk."""
    if not SAVE_LOCAL_DATA:
        return
    arr = stationary_thermal_background_by_sensor.get(sensor_id)
    if arr is None:
        return
    safe_id = _sanitize_sensor_id_for_filename(sensor_id)
    path = DATA_DIR / f"background_stationary_{safe_id}.npy"
    try:
        np.save(path, arr)
    except Exception as e:
        print(f"Error saving stationary background ({sensor_id}): {e}")


def _update_stationary_background(sensor_id: str, temp_array_2d: np.ndarray) -> None:
    """Slow EMA update of the stationary background — runs on every frame.

    Unlike the empty-room background, this accumulates regardless of occupancy.
    Heat sources that are present all day (radiators, computers, warm walls) will
    converge toward their actual temperature in this background over ~8 hours,
    so the delta used for people detection approaches zero for those pixels.
    People produce a large delta because they are only transiently present at any
    given pixel, so their contribution averages out much more slowly than a full day.
    """
    global stationary_thermal_background_by_sensor, _stationary_bg_frame_count
    arr = temp_array_2d.astype(np.float64)
    if sensor_id not in stationary_thermal_background_by_sensor:
        stationary_thermal_background_by_sensor[sensor_id] = arr.copy()
        _stationary_bg_frame_count[sensor_id] = 1
    else:
        bg = stationary_thermal_background_by_sensor[sensor_id]
        stationary_thermal_background_by_sensor[sensor_id] = (
            STATIONARY_BG_ALPHA * bg + (1.0 - STATIONARY_BG_ALPHA) * arr
        )
        _stationary_bg_frame_count[sensor_id] = _stationary_bg_frame_count.get(sensor_id, 0) + 1
    if _stationary_bg_frame_count.get(sensor_id, 0) % STATIONARY_BG_SAVE_INTERVAL == 0:
        _save_stationary_background(sensor_id)


def _parse_temperature(value: Any) -> float:
    """Convert a temperature value to float. Accepts numbers or strings like '21.3' or '21.3°C'."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Strip trailing °C, °F, or other non-numeric suffix
    for suffix in ("°C", "°F", "C", "F", "°"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    return float(s)


def temperature_to_color(temp: float, min_temp: float, max_temp: float) -> Tuple[int, int, int]:
    temp = max(min_temp, min(max_temp, temp))
    if max_temp == min_temp:
        return (128, 128, 128)
    normalized = (temp - min_temp) / (max_temp - min_temp)
    if normalized < 0.25:
        r, g, b = 0, int(normalized * 4 * 255), 255
    elif normalized < 0.5:
        r, g, b = 0, 255, int((1 - (normalized - 0.25) * 4) * 255)
    elif normalized < 0.75:
        r, g, b = int((normalized - 0.5) * 4 * 255), 255, 0
    else:
        r, g, b = 255, int((1 - (normalized - 0.75) * 4) * 255), 0
    return (r, g, b)


def expand_thermal_data(compact_data: dict) -> dict:
    width = compact_data["w"]
    height = compact_data["h"]
    min_temp = _parse_temperature(compact_data["min"])
    max_temp = _parse_temperature(compact_data["max"])
    temps = compact_data["t"]
    pixels = []
    for i, temp in enumerate(temps):
        t_float = _parse_temperature(temp)
        row = i // width
        col = i % width
        r, g, b = temperature_to_color(t_float, min_temp, max_temp)
        pixels.append({"row": row, "col": col, "temp": t_float, "r": r, "g": g, "b": b})
    return {
        "width": width,
        "height": height,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "pixels": pixels,
    }


def collapse_to_compact(expanded_data: dict) -> dict:
    """Convert expanded thermal data (pixels) to compact format (w, h, min, max, t)."""
    width = expanded_data["width"]
    height = expanded_data["height"]
    pixels = expanded_data.get("pixels") or []
    if not pixels:
        raise ValueError("expanded thermal data has no pixels")
    # Row-major order (same as expand_thermal_data: index = row * width + col)
    sorted_pixels = sorted(pixels, key=lambda p: (p["row"], p["col"]))
    t = [_parse_temperature(p["temp"]) for p in sorted_pixels]
    out = {
        "w": width,
        "h": height,
        "min": round(min(t), 1),
        "max": round(max(t), 1),
        "t": [round(x, 1) for x in t],
    }
    if expanded_data.get("sensor_id") is not None:
        out["sensor_id"] = expanded_data["sensor_id"]
    return out


def detect_subpage_artifact(temp_array: np.ndarray) -> Tuple[bool, float]:
    """Detect the MLX90640 subpage chessboard artifact.

    The MLX90640 captures two interleaved subpages (even rows then odd rows, or
    vice-versa) separated by ~40 ms. When a person is moving, the two subpages
    disagree and produce a checkerboard pattern: alternating rows are systematically
    warmer or cooler than their neighbors.

    Detection approach:
      1. Compute per-row means, then compute differences between consecutive row
         means. A clean frame has small, sign-random differences; a subpage-corrupted
         frame shows large differences whose signs tend to alternate every row.
      2. Step through diff pairs (d[0],d[1]), (d[2],d[3]), … and count pairs where
         both magnitudes exceed the threshold AND the signs are opposite.

    Returns (is_corrupted, checkerboard_fraction).
    """
    if not SUBPAGE_ARTIFACT_ENABLED:
        return False, 0.0
    h, w = temp_array.shape
    if h < 4:
        return False, 0.0

    row_means = temp_array.mean(axis=1)  # shape (h,)
    # Differences between consecutive rows
    diffs = row_means[1:] - row_means[:-1]  # shape (h-1,)

    # Look for sign alternation in pairs: diff[i] and diff[i+1] should have opposite signs
    n_alternating = 0
    total_pairs = 0
    for i in range(0, len(diffs) - 1, 2):
        d0, d1 = diffs[i], diffs[i + 1]
        total_pairs += 1
        if abs(d0) > SUBPAGE_ROW_DIFF_THRESHOLD_C and abs(d1) > SUBPAGE_ROW_DIFF_THRESHOLD_C:
            if d0 * d1 < 0:  # opposite signs
                n_alternating += 1

    checkerboard_frac = n_alternating / max(1, total_pairs)
    is_corrupted = checkerboard_frac >= SUBPAGE_CHECKERBOARD_FRAC
    return is_corrupted, checkerboard_frac


def interpolate_subpages(corrupted: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Blend a corrupted frame with the previous clean frame to suppress the artifact.

    Rather than discarding the corrupted frame entirely (which would stall occupancy
    updates), we average the two frames. This smooths out the chessboard offset while
    retaining any genuine thermal changes that occurred since the previous frame.
    """
    return (corrupted.astype(float) + previous.astype(float)) / 2.0


def prepare_thermal_frame_for_analysis(
    thermal_data: dict, sensor_id: Optional[str] = None
) -> Tuple[np.ndarray, bool, float]:
    """Convert thermal data to a 2-D array and apply subpage artifact correction.

    Returns (corrected_array, subpage_corrupted, checkerboard_frac). Updates
    _last_clean_frame_by_sensor so the same corrected frame is used by all
    downstream steps (occupancy, signal processing, background update).
    """
    temp_array_2d = thermal_data_to_array(thermal_data)
    subpage_corrupted = False
    subpage_frac = 0.0
    if sensor_id is not None:
        subpage_corrupted, subpage_frac = detect_subpage_artifact(temp_array_2d)
        cached_clean_frame = _last_clean_frame_by_sensor.get(sensor_id)
        if subpage_corrupted and cached_clean_frame is not None:
            if cached_clean_frame.shape == temp_array_2d.shape:
                temp_array_2d = interpolate_subpages(temp_array_2d, cached_clean_frame)
            else:
                _last_clean_frame_by_sensor.pop(sensor_id, None)
        if not subpage_corrupted:
            _last_clean_frame_by_sensor[sensor_id] = temp_array_2d.copy()
    return temp_array_2d, subpage_corrupted, subpage_frac


def _validate_thermal_payload(data: dict) -> None:
    """Reject malformed frames before processing (avoids crashes on empty or length-mismatch data)."""
    if "t" in data:
        if "w" not in data or "h" not in data:
            raise ValueError("compact thermal data must include w and h")
        w, h = int(data["w"]), int(data["h"])
        t = data.get("t")
        if t is None:
            raise ValueError("compact thermal data must include t")
        if len(t) == 0:
            raise ValueError("compact thermal data has an empty temperature array")
        if w <= 0 or h <= 0:
            raise ValueError("invalid thermal dimensions")
        if len(t) != w * h:
            raise ValueError("compact thermal data length does not match w*h")
        return
    if "pixels" in data:
        if "width" not in data or "height" not in data:
            raise ValueError("expanded thermal data must include width and height")
        w, h = int(data["width"]), int(data["height"])
        pixels = data.get("pixels") or []
        if w <= 0 or h <= 0:
            raise ValueError("invalid thermal dimensions")
        if len(pixels) == 0:
            raise ValueError("expanded thermal data has an empty pixels array")
        if len(pixels) != w * h:
            raise ValueError("expanded thermal pixel count does not match width*height")
        return
    raise ValueError("unknown thermal data format")


def thermal_data_to_array(data: dict) -> np.ndarray:
    if "t" in data:
        width, height = data["w"], data["h"]
        temps = [_parse_temperature(t) for t in data["t"]]
    elif "pixels" in data:
        width, height = data["width"], data["height"]
        temps = [_parse_temperature(p["temp"]) for p in data["pixels"]]
    else:
        raise ValueError("Unknown thermal data format")
    return np.array(temps, dtype=float).reshape((height, width))


def estimate_room_temperature(temp_array: np.ndarray) -> float:
    return float(np.median(temp_array))


def detect_human_heat(
    temp_array: np.ndarray,
    room_temp: float,
    use_delta: bool = False,
    absolute_temp_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Human heat mask.

    If ``use_delta`` is True, ``temp_array`` is delta above background. The upper bound
    must apply to **absolute** temperature (human bodies are not hotter than ``MAX_HUMAN_TEMP``);
    otherwise a hot object (e.g. 60 °C) can produce a delta below 45 °C and be mistaken for a person.
    """
    if use_delta:
        abs_arr = absolute_temp_array if absolute_temp_array is not None else temp_array
        human_mask = (temp_array >= ROOM_TEMP_THRESHOLD) & (abs_arr <= MAX_HUMAN_TEMP)
    else:
        human_mask = (temp_array >= MIN_HUMAN_TEMP) & (temp_array <= MAX_HUMAN_TEMP)
        relative_mask = (temp_array - room_temp) >= ROOM_TEMP_THRESHOLD
        human_mask = human_mask & relative_mask
    return human_mask.astype(int)


def find_people_clusters(
    human_mask: np.ndarray,
    temp_array: np.ndarray,
    fever_threshold: Optional[float] = None,
    elevated_threshold: Optional[float] = None,
) -> List[Dict]:
    """Detect human clusters with shape and temperature filtering.

    ``fever_threshold`` / ``elevated_threshold`` allow the caller to pass
    ambient-compensated values; fall back to global constants when absent.
    """
    eff_fever = fever_threshold if fever_threshold is not None else FEVER_THRESHOLD_C
    eff_elevated = elevated_threshold if elevated_threshold is not None else FEVER_ELEVATED_THRESHOLD_C

    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(human_mask, structure=structure)
    people_clusters = []
    for i in range(1, num_features + 1):
        cluster_size = int(np.sum(labeled_array == i))
        if not (MIN_CLUSTER_SIZE <= cluster_size <= MAX_CLUSTER_SIZE):
            continue
        cluster_pixels = np.where(labeled_array == i)
        rows, cols = cluster_pixels
        # Bounding-box shape filters: reject elongated and very sparse blobs.
        bbox_h = int(rows.max() - rows.min()) + 1
        bbox_w = int(cols.max() - cols.min()) + 1
        long_side = max(bbox_h, bbox_w)
        short_side = min(bbox_h, bbox_w)
        aspect_ratio = long_side / max(short_side, 1)
        fill_ratio = cluster_size / max(bbox_h * bbox_w, 1)
        if aspect_ratio > MAX_CLUSTER_ASPECT_RATIO or fill_ratio < MIN_CLUSTER_FILL_RATIO:
            continue
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))
        cluster_temps = temp_array[rows, cols]
        representative_temp_c = float(np.percentile(cluster_temps, 90)) if cluster_temps.size else 0.0
        fever_detected = representative_temp_c >= eff_fever
        elevated_temp = (
            eff_elevated > 0
            and representative_temp_c >= eff_elevated
            and representative_temp_c < eff_fever
        )
        people_clusters.append({
            "id": i,
            "size": cluster_size,
            "center": (center_row, center_col),
            "representative_temp_c": round(representative_temp_c, 2),
            "elevated_temp": elevated_temp,
            "fever_detected": fever_detected,
            "aspect_ratio": round(aspect_ratio, 2),
            "fill_ratio": round(fill_ratio, 2),
        })
    return people_clusters


def estimate_occupancy(
    thermal_data: dict,
    sensor_id: Optional[str] = None,
    *,
    _prepared: Optional[Tuple[np.ndarray, bool, float]] = None,
) -> dict:
    subpage_corrupted = False
    subpage_frac = 0.0
    try:
        if _prepared is not None:
            temp_array_2d, subpage_corrupted, subpage_frac = _prepared
        else:
            temp_array_2d, subpage_corrupted, subpage_frac = prepare_thermal_frame_for_analysis(
                thermal_data, sensor_id
            )

        room_temp = estimate_room_temperature(temp_array_2d)
        array_for_detection = temp_array_2d
        use_delta = False
        if sensor_id:
            # Combine the empty-room background (fast, only updates when unoccupied) with
            # the stationary background (slow, always updates) by taking the per-pixel
            # maximum.  This cancels out both ambient shifts and always-on heat sources
            # (radiators, monitors, warm walls) without requiring the room to be empty.
            empty_bg = thermal_background_by_sensor.get(sensor_id)
            stationary_bg = stationary_thermal_background_by_sensor.get(sensor_id)
            combined_bg: Optional[np.ndarray] = None
            if empty_bg is not None and empty_bg.shape == temp_array_2d.shape:
                combined_bg = empty_bg
            if stationary_bg is not None and stationary_bg.shape == temp_array_2d.shape:
                combined_bg = (
                    np.maximum(combined_bg, stationary_bg)
                    if combined_bg is not None
                    else stationary_bg
                )
            if combined_bg is not None:
                delta = np.maximum(0.0, temp_array_2d - combined_bg)
                array_for_detection = delta
                use_delta = True
        human_mask = detect_human_heat(
            array_for_detection,
            room_temp,
            use_delta=use_delta,
            absolute_temp_array=temp_array_2d if use_delta else None,
        )
        # Ambient-compensated fever thresholds: skin surface temperature tracks room
        # temperature, so the absolute fever threshold needs to shift with ambient.
        ambient_delta = room_temp - FEVER_REFERENCE_ROOM_C
        comp = FEVER_AMBIENT_COMP_C * ambient_delta
        eff_fever = FEVER_THRESHOLD_C + comp
        eff_elevated = FEVER_ELEVATED_THRESHOLD_C + comp if FEVER_ELEVATED_THRESHOLD_C > 0 else 0.0
        people_clusters = find_people_clusters(
            human_mask, temp_array_2d,
            fever_threshold=eff_fever,
            elevated_threshold=eff_elevated,
        )
        fever_count = sum(1 for c in people_clusters if c.get("fever_detected"))
        elevated_count = sum(1 for c in people_clusters if c.get("elevated_temp"))
        return {
            "occupancy": len(people_clusters),
            "room_temperature": room_temp,
            "people_clusters": people_clusters,
            "fever_count": fever_count,
            "elevated_count": elevated_count,
            "any_fever": fever_count > 0,
            "any_elevated": elevated_count > 0,
            "effective_fever_threshold": round(eff_fever, 2),
            "subpage_corrupted": subpage_corrupted,
            "subpage_checkerboard_frac": round(subpage_frac, 3),
        }
    except Exception as e:
        return {
            "occupancy": 0,
            "room_temperature": None,
            "people_clusters": [],
            "fever_count": 0,
            "elevated_count": 0,
            "any_fever": False,
            "any_elevated": False,
            "subpage_corrupted": subpage_corrupted,
            "subpage_checkerboard_frac": round(subpage_frac, 3),
            "error": str(e),
        }


def apply_occupancy_signal_processing(
    sensor_id: str, occupancy_result: dict, temp_array_2d: np.ndarray
) -> None:
    """
    Mutates occupancy_result in place: frame sanity (median jump), temporal median smoothing,
    hysteresis, and consecutive-frame fever gating. Does not re-run clustering.
    """
    instant_from_model = int(occupancy_result["occupancy"])
    frame_median = float(np.median(temp_array_2d))
    frame_valid = True
    if FRAME_ROOM_MEDIAN_MAX_JUMP_C > 0 and sensor_id in last_frame_median_by_sensor:
        if abs(frame_median - last_frame_median_by_sensor[sensor_id]) > FRAME_ROOM_MEDIAN_MAX_JUMP_C:
            frame_valid = False

    last_frame_median_by_sensor[sensor_id] = frame_median

    if frame_valid:
        effective_raw = instant_from_model
        last_raw_occupancy_by_sensor[sensor_id] = instant_from_model
    else:
        effective_raw = last_raw_occupancy_by_sensor.get(sensor_id, instant_from_model)

    occupancy_result["occupancy_raw_instant"] = instant_from_model
    occupancy_result["occupancy_effective_raw"] = effective_raw
    occupancy_result["frame_valid"] = frame_valid

    maxlen = OCCUPANCY_SMOOTH_WINDOW
    if sensor_id not in occupancy_raw_history_by_sensor:
        occupancy_raw_history_by_sensor[sensor_id] = deque(maxlen=maxlen)
    occupancy_raw_history_by_sensor[sensor_id].append(effective_raw)

    window = list(occupancy_raw_history_by_sensor[sensor_id])
    cand = int(round(float(np.median(window))))

    prev_s = last_smoothed_occupancy_by_sensor.get(sensor_id)
    if prev_s is not None and OCCUPANCY_HYSTERESIS_DELTA > 0:
        if abs(cand - prev_s) <= OCCUPANCY_HYSTERESIS_DELTA and cand != prev_s:
            smoothed = prev_s
        else:
            smoothed = cand
    else:
        smoothed = cand
    last_smoothed_occupancy_by_sensor[sensor_id] = smoothed
    occupancy_result["occupancy"] = smoothed

    any_fever_raw = bool(occupancy_result.get("any_fever", False))
    if any_fever_raw:
        fever_consecutive_by_sensor[sensor_id] = fever_consecutive_by_sensor.get(sensor_id, 0) + 1
    else:
        fever_consecutive_by_sensor[sensor_id] = 0
    streak = fever_consecutive_by_sensor[sensor_id]
    occupancy_result["fever_consecutive_frames"] = streak
    occupancy_result["any_fever_raw"] = any_fever_raw
    if streak >= FEVER_MIN_CONSECUTIVE_FRAMES:
        occupancy_result["any_fever"] = True
    else:
        occupancy_result["any_fever"] = False

    # Sustained elevated-temperature gate (independent of full fever).
    any_elevated_raw = bool(occupancy_result.get("any_elevated", False))
    if any_elevated_raw:
        elevated_consecutive_by_sensor[sensor_id] = elevated_consecutive_by_sensor.get(sensor_id, 0) + 1
    else:
        elevated_consecutive_by_sensor[sensor_id] = 0
    elevated_streak = elevated_consecutive_by_sensor[sensor_id]
    occupancy_result["elevated_consecutive_frames"] = elevated_streak
    occupancy_result["sustained_elevated"] = elevated_streak >= ELEVATED_MIN_CONSECUTIVE_FRAMES


def _sanitize_sensor_id_for_filename(sensor_id: Optional[str]) -> str:
    if not sensor_id:
        return "unknown"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sensor_id))[:64]


def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(x) for x in obj)
    return obj


def _read_json_payload(path: Path) -> dict:
    """Read JSON payload from .json or .json.gz files."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())

def _get_sql_connection():
    """Return a cached Azure SQL connection. Returns None if not configured or permanently disabled."""
    global _sql_connection
    if _sql_connection is False:
        return None  # permanently disabled after earlier failure
    if _sql_connection is not None:
        try:
            # Lightweight connectivity check at the ODBC driver level (no query sent to server).
            _sql_connection.getinfo(2)  # SQL_DATA_SOURCE_NAME
            return _sql_connection
        except Exception:
            _sql_connection = None  # stale, attempt to reconnect below

    if not SQL_CONNECTION_STRING:
        return None

    try:
        import pyodbc  # noqa: PLC0415 – intentionally deferred for optional dependency
    except ImportError:
        print("pyodbc is not installed; Azure SQL saving disabled.")
        _sql_connection = False
        return None

    try:
        conn = pyodbc.connect(SQL_CONNECTION_STRING, timeout=10)
        _sql_connection = conn
        return _sql_connection
    except Exception as e:
        print(f"Azure SQL connection failed ({type(e).__name__}); will retry on next upload: {e}")
        _sql_connection = None
        return None


def save_occupancy_data_sql(occupancy_result: dict, timestamp_iso: Optional[str] = None) -> None:
    """Save occupancy estimation to Azure SQL."""
    global _sql_connection
    if not SAVE_TO_SQL:
        return

    conn = _get_sql_connection()
    if conn is None:
        return

    cursor = None
    try:
        sid = occupancy_result.get("sensor_id") or "unknown"
        ts = timestamp_iso or datetime.now().isoformat()

        entry = {
            "timestamp": ts,
            "sensor_id": sid,
            "occupancy": int(occupancy_result["occupancy"]),
            "room_temperature": (
                float(occupancy_result["room_temperature"])
                if occupancy_result.get("room_temperature") is not None
                else None
            ),
            "people_clusters": json.dumps(
                convert_numpy_types(occupancy_result.get("people_clusters", []))
            ),
            "fever_count": int(occupancy_result.get("fever_count", 0)),
            "any_fever": bool(occupancy_result.get("any_fever", False)),
        }

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO occupancy_data
            ([timestamp], sensor_id, occupancy, room_temperature,
             people_clusters, fever_count, any_fever)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            entry["timestamp"],
            entry["sensor_id"],
            entry["occupancy"],
            entry["room_temperature"],
            entry["people_clusters"],
            entry["fever_count"],
            1 if entry["any_fever"] else 0,
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving occupancy data to Azure SQL ({type(e).__name__}); will retry on next call.")
        try:
            conn.rollback()
        except Exception:
            pass
        # Invalidate the cached connection so the next call will reconnect
        _sql_connection = None
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass


def save_thermal_data(compact_data: dict, sensor_id: Optional[str] = None) -> None:
    global _data_counter
    if not SAVE_LOCAL_DATA and not SAVE_TO_BLOB:
        return
    sid = sensor_id or compact_data.get("sensor_id") or "unknown"
    safe_id = _sanitize_sensor_id_for_filename(sid)
    try:
        timestamp = datetime.now(timezone.utc)
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        payload = {
            "timestamp": timestamp.isoformat(),
            "format": "compact",
            "sensor_id": sid,
            "data": compact_data,
        }
        compressed_bytes = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        path = DATA_DIR / f"thermal_{safe_id}_{ts}_compact.json.gz"
        if SAVE_LOCAL_DATA:
            with open(path, "wb") as f:
                f.write(compressed_bytes)
        if SAVE_TO_BLOB:
            # Azure Blob: compressed compact payload for long-term retention.
            _upload_blob(f"thermal/{path.name}", compressed_bytes, content_type="application/gzip")
        _data_counter += 1
    except Exception as e:
        print(f"Error saving thermal data (sensor={sid}): {e}")


def save_occupancy_data(occupancy_result: dict) -> None:
    if not SAVE_LOCAL_DATA and not SAVE_TO_BLOB:
        return
    try:
        timestamp = datetime.now(timezone.utc)
        date_str = timestamp.strftime("%Y%m%d")
        path = DATA_DIR / f"occupancy_{date_str}.jsonl"
        sid = occupancy_result.get("sensor_id") or "unknown"
        entry = {
            "timestamp": timestamp.isoformat(),
            "sensor_id": sid,
            "occupancy": int(occupancy_result["occupancy"]),
            "occupancy_raw_instant": int(occupancy_result.get("occupancy_raw_instant", occupancy_result["occupancy"])),
            "occupancy_effective_raw": int(
                occupancy_result.get("occupancy_effective_raw", occupancy_result["occupancy"])
            ),
            "room_temperature": (
                float(occupancy_result["room_temperature"])
                if occupancy_result.get("room_temperature") is not None
                else None
            ),
            "people_clusters": convert_numpy_types(occupancy_result.get("people_clusters", [])),
            "fever_count": int(occupancy_result.get("fever_count", 0)),
            "elevated_count": int(occupancy_result.get("elevated_count", 0)),
            "any_fever": bool(occupancy_result.get("any_fever", False)),
            "any_fever_raw": bool(occupancy_result.get("any_fever_raw", occupancy_result.get("any_fever", False))),
            "any_elevated": bool(occupancy_result.get("any_elevated", False)),
            "fever_consecutive_frames": int(occupancy_result.get("fever_consecutive_frames", 0)),
            "elevated_consecutive_frames": int(occupancy_result.get("elevated_consecutive_frames", 0)),
            "sustained_elevated": bool(occupancy_result.get("sustained_elevated", False)),
            "effective_fever_threshold": occupancy_result.get("effective_fever_threshold"),
            "frame_valid": bool(occupancy_result.get("frame_valid", True)),
        }
        line = json.dumps(entry) + "\n"
        if SAVE_LOCAL_DATA:
            with open(path, "a") as f:
                f.write(line)
        if SAVE_TO_BLOB:
            # Azure Blob: append to daily append blob
            _append_to_blob(f"occupancy/occupancy_{date_str}.jsonl", line)
    except Exception as e:
        print(f"Error saving occupancy data (sensor={sid}): {e}")


def _iter_thermal_files() -> List[Path]:
    """Return all locally stored thermal frame files (legacy json + compressed json.gz)."""
    if not DATA_DIR.exists():
        return []
    files = [p for p in DATA_DIR.glob("thermal_*.json") if p.is_file()]
    files.extend([p for p in DATA_DIR.glob("thermal_*.json.gz") if p.is_file()])
    # Newest first (filenames include timestamp)
    files.sort(key=lambda p: p.name, reverse=True)
    return files


def _parse_thermal_blob_meta(filename: str) -> tuple:
    """Parse (safe_id, timestamp_iso) from a thermal frame filename.

    Pattern: thermal_{safe_id}_{YYYYMMDD}_{HHMMSS}_{ms}_compact.json(.gz)
    Returns ("", "") when the filename does not match.
    """
    stem = filename
    for suffix in ("_compact.json.gz", "_compact.json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if (
            len(p) == 8
            and p.isdigit()
            and i + 1 < len(parts)
            and len(parts[i + 1]) == 6
            and parts[i + 1].isdigit()
        ):
            date_p, time_p = p, parts[i + 1]
            ts_iso = (
                f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:]}"
                f"T{time_p[:2]}:{time_p[2:4]}:{time_p[4:]}"
            )
            safe_id = "_".join(parts[1:i])
            return safe_id, ts_iso
    return "", ""


def _list_blob_thermal_names(
    sensor_id_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
) -> List[str]:
    """List thermal frame filenames stored in Azure Blob under the thermal/ prefix.

    Returns bare filenames (no path prefix), sorted newest-first.
    Results are pre-filtered by date and sensor_id when supplied to avoid
    downloading any blob content.
    """
    container = _get_blob_container()
    if container is None:
        return []
    safe_filter = _sanitize_sensor_id_for_filename(sensor_id_filter) if sensor_id_filter else None
    try:
        blobs = container.list_blobs(name_starts_with="thermal/")
        names: List[str] = []
        for blob in blobs:
            filename = blob.name[len("thermal/"):]
            if not filename.startswith("thermal_"):
                continue
            if not (filename.endswith("_compact.json.gz") or filename.endswith("_compact.json")):
                continue
            if date_filter and date_filter not in filename:
                continue
            if safe_filter:
                blob_safe_id, _ = _parse_thermal_blob_meta(filename)
                if blob_safe_id != safe_filter:
                    continue
            names.append(filename)
        names.sort(reverse=True)
        return names
    except Exception as e:
        print(f"Azure Blob list failed: {e}")
        return []


def _ensure_local_copy(filename: str) -> Optional[Path]:
    """Return a local Path to the thermal frame, downloading from Blob if needed.

    Returns None when the file is not found locally or in Blob.
    """
    local_path = DATA_DIR / filename
    if local_path.exists():
        return local_path
    container = _get_blob_container()
    if container is None:
        return None
    try:
        blob_client = container.get_blob_client(f"thermal/{filename}")
        data = blob_client.download_blob().readall()
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return local_path
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, delete=False
            ) as tf:
                tf.write(data)
                return Path(tf.name)
    except Exception:
        return None


def _safe_int(value: Optional[int], default: int, min_value: int, max_value: int) -> int:
    try:
        v = int(value) if value is not None else int(default)
    except Exception:
        v = int(default)
    return max(min_value, min(max_value, v))


@app.get("/api/sensors")
def list_sensors() -> dict:
    """List known sensor IDs with any stored room metadata."""
    sensors = set(latest_thermal_by_sensor.keys()) | set(latest_occupancy_by_sensor.keys())
    for p in _iter_thermal_files():
        try:
            payload = _read_json_payload(p)
            sid = payload.get("sensor_id")
            if sid:
                sensors.add(str(sid))
        except Exception:
            continue
    out = sorted(sensors)
    return {
        "count": len(out),
        "sensors": out,
        "metadata": {sid: _sensor_metadata.get(sid, {}) for sid in out},
    }


class _SensorMetadataUpdate(BaseModel):
    room_name: Optional[str] = None
    capacity: Optional[int] = None
    building: Optional[str] = None


@app.patch("/api/sensors/{sensor_id}")
def update_sensor_metadata(sensor_id: str, update: _SensorMetadataUpdate) -> dict:
    """Set room name, capacity, and/or building for a sensor.

    Only supplied fields are updated; omitted fields are left unchanged.
    Capacity is used for occupancy percentage reporting and overcapacity alerts.
    """
    if update.capacity is not None and update.capacity < 0:
        raise HTTPException(status_code=400, detail="capacity must be >= 0")
    with _sensor_metadata_lock:
        meta = dict(_sensor_metadata.get(sensor_id, {}))
        if update.room_name is not None:
            meta["room_name"] = update.room_name
        if update.capacity is not None:
            meta["capacity"] = update.capacity
        if update.building is not None:
            meta["building"] = update.building
        meta["ts_updated"] = datetime.now(timezone.utc).isoformat()
        _sensor_metadata[sensor_id] = meta
    _persist_sensor_metadata()
    return {"sensor_id": sensor_id, "metadata": meta}


@app.get("/api/thermal/history")
def get_thermal_history(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
    date: Optional[str] = Query(default=None, description="YYYYMMDD (optional)"),
    limit: int = Query(default=100, description="Max frames to return (1..500)"),
    offset: int = Query(default=0, description="Number of matching frames to skip"),
    include_data: bool = Query(default=False, description="If true, include full frame payload; else metadata only"),
) -> dict:
    """
    Return stored thermal frames (all sensors by default).
    Sources: local THERMAL_DATA_DIR files first, then Azure Blob Storage when configured.
    Blob-only entries return metadata derived from the filename; full pixel data is
    downloaded on demand when include_data=true.
    """
    limit_i = _safe_int(limit, 100, 1, 500)
    offset_i = _safe_int(offset, 0, 0, 1_000_000_000)
    # Fetch one extra row so has_more is false when the page is full but no next page exists.
    fetch_limit = limit_i + 1

    # Build a unified candidate list: local files first, then Blob-only entries.
    # Each candidate is either a Path (local) or a str filename (Blob-only).
    local_files = _iter_thermal_files()
    local_names = {p.name for p in local_files}

    candidates: List[Any] = list(local_files)
    for blob_name in _list_blob_thermal_names(sensor_id_filter=sensor_id, date_filter=date):
        if blob_name not in local_names:
            candidates.append(blob_name)  # str = Blob-only

    def _build_entry(source: Any) -> Optional[dict]:
        """Build a history entry dict from a local Path or a Blob filename string."""
        is_blob_only = isinstance(source, str)
        filename = source if is_blob_only else source.name

        if is_blob_only:
            # Derive metadata from filename without downloading.
            blob_safe_id, ts_iso = _parse_thermal_blob_meta(filename)
            sid = blob_safe_id or None
            ts = ts_iso or None
            fmt = "compact"
        else:
            try:
                payload = _read_json_payload(source)
            except Exception:
                return None
            sid = payload.get("sensor_id")
            ts = payload.get("timestamp")
            fmt = payload.get("format")

        # Apply filters (redundant for Blob entries pre-filtered by _list_blob_thermal_names,
        # but required for local entries).
        if sensor_id is not None and str(sid) != str(sensor_id):
            return None
        if date:
            try:
                ymd = str(ts)[:10].replace("-", "")
            except Exception:
                ymd = ""
            if ymd != date:
                return None

        entry: dict = {
            "file": filename,
            "timestamp": ts,
            "sensor_id": sid,
            "format": fmt,
        }

        if include_data:
            # For Blob-only entries we need to download first.
            if is_blob_only:
                local_path = _ensure_local_copy(filename)
                if local_path is None:
                    return entry  # return metadata-only if download fails
                try:
                    payload = _read_json_payload(local_path)
                except Exception:
                    return entry
            payload_data = payload.get("data")
            source_format = fmt
            returned_format = fmt
            if payload.get("format") == "compact" and isinstance(payload_data, dict) and "t" in payload_data:
                try:
                    expanded = expand_thermal_data(payload_data)
                    expanded["sensor_id"] = sid
                    entry["data"] = expanded
                    returned_format = "expanded"
                except Exception:
                    entry["data"] = payload_data
                    returned_format = "compact"
            else:
                entry["data"] = payload_data
            entry["source_format"] = source_format
            entry["returned_format"] = returned_format
            entry["format"] = returned_format

        return entry

    matches: List[dict] = []
    seen = 0
    for candidate in candidates:
        entry = _build_entry(candidate)
        if entry is None:
            continue
        if seen < offset_i:
            seen += 1
            continue
        matches.append(entry)
        if len(matches) >= fetch_limit:
            break

    has_more = len(matches) > limit_i
    page = matches[:limit_i]
    next_offset = (offset_i + len(page)) if has_more else None

    return {
        "sensor_id": sensor_id,
        "date": date,
        "limit": limit_i,
        "offset": offset_i,
        "count": len(page),
        "has_more": has_more,
        "next_offset": next_offset,
        "data": page,
    }


@app.get("/api/test")
def test() -> dict:
    """Health check for Azure and clients."""
    return {"status": "server is running", "time": datetime.now().isoformat()}


def _maybe_update_thermal_background(sensor_id: str, temp_array_2d: np.ndarray, occupancy_effective_raw: int) -> None:
    """
    Update per-sensor thermal background when room is empty (effective raw occupancy 0).
    Requires BACKGROUND_MIN_FRAMES_EMPTY consecutive stable empty frames (low frame-to-frame change).
    """
    global thermal_background_by_sensor, empty_frame_count_by_sensor
    if sensor_id not in empty_frame_count_by_sensor:
        empty_frame_count_by_sensor[sensor_id] = 0
    if occupancy_effective_raw > 0:
        empty_frame_count_by_sensor[sensor_id] = 0
        return

    if BACKGROUND_MAX_MEAN_ABS_DELTA_C > 0 and sensor_id in last_empty_frame_thermal_by_sensor:
        prev = last_empty_frame_thermal_by_sensor[sensor_id]
        if prev.shape == temp_array_2d.shape:
            mad = float(np.mean(np.abs(temp_array_2d - prev)))
            if mad > BACKGROUND_MAX_MEAN_ABS_DELTA_C:
                empty_frame_count_by_sensor[sensor_id] = 1
                last_empty_frame_thermal_by_sensor[sensor_id] = temp_array_2d.copy().astype(np.float64)
                return

    last_empty_frame_thermal_by_sensor[sensor_id] = temp_array_2d.copy().astype(np.float64)
    empty_frame_count_by_sensor[sensor_id] += 1
    if empty_frame_count_by_sensor[sensor_id] < BACKGROUND_MIN_FRAMES_EMPTY:
        return
    empty_frame_count_by_sensor[sensor_id] = 0
    alpha = BACKGROUND_ALPHA
    if sensor_id not in thermal_background_by_sensor:
        thermal_background_by_sensor[sensor_id] = temp_array_2d.copy().astype(np.float64)
    else:
        thermal_background_by_sensor[sensor_id] = (
            alpha * thermal_background_by_sensor[sensor_id] + (1.0 - alpha) * temp_array_2d
        ).astype(np.float64)
    _save_thermal_background(sensor_id)


@app.post("/api/thermal")
def receive_thermal_data(data: dict) -> dict:
    """Receive thermal data from ESP32 (compact or expanded format)."""
    global latest_thermal_data, last_update_time, latest_occupancy
    if not data:
        raise HTTPException(status_code=400, detail="No data received")
    try:
        _validate_thermal_payload(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    sensor_id = data.get("sensor_id") or "unknown"
    if sensor_id not in thermal_background_by_sensor:
        _load_thermal_background(sensor_id)
    if sensor_id not in stationary_thermal_background_by_sensor:
        _load_stationary_background(sensor_id)
    if "t" in data:
        compact_data = dict(data)
        try:
            expanded_data = expand_thermal_data(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data expansion failed: {e}") from e
        expanded_data["sensor_id"] = sensor_id
        latest_thermal_data = expanded_data
    else:
        latest_thermal_data = data if data.get("sensor_id") else {**data, "sensor_id": sensor_id}
        try:
            compact_data = collapse_to_compact(latest_thermal_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    prepared = prepare_thermal_frame_for_analysis(data, sensor_id=sensor_id)
    corrected_frame = prepared[0]
    occupancy_result = estimate_occupancy(data, sensor_id=sensor_id, _prepared=prepared)
    occupancy_result["sensor_id"] = sensor_id
    try:
        apply_occupancy_signal_processing(sensor_id, occupancy_result, corrected_frame)
        _maybe_update_thermal_background(
            sensor_id, corrected_frame, int(occupancy_result.get("occupancy_effective_raw", 0))
        )
        _update_stationary_background(sensor_id, corrected_frame)
        if _ml_engine is not None and _ml_engine.available:
            empty_bg = thermal_background_by_sensor.get(sensor_id)
            stat_bg = stationary_thermal_background_by_sensor.get(sensor_id)
            ml_bg = (
                np.maximum(empty_bg, stat_bg)
                if empty_bg is not None and stat_bg is not None and empty_bg.shape == stat_bg.shape
                else (empty_bg if empty_bg is not None else stat_bg)
            )
            ml_result = _ml_engine.predict(corrected_frame, background=ml_bg)
            if ml_result:
                occupancy_result["ml"] = ml_result
    except Exception as e:
        print(f"Signal processing error (sensor={sensor_id}): {e}")
    latest_occupancy = occupancy_result
    now_iso = datetime.now(timezone.utc).isoformat()
    last_update_time = now_iso
    # Per-sensor latest state
    latest_thermal_by_sensor[sensor_id] = dict(latest_thermal_data) if latest_thermal_data else {}
    latest_occupancy_by_sensor[sensor_id] = dict(occupancy_result)
    last_update_time_by_sensor[sensor_id] = now_iso
    save_thermal_data(compact_data, sensor_id)
    save_occupancy_data(occupancy_result)
    save_occupancy_data_sql(occupancy_result, timestamp_iso=now_iso)
    _maybe_fire_alerts(sensor_id, occupancy_result, now_iso)
    pixel_count = len(latest_thermal_data.get("pixels", []))
    return {
        "status": "success",
        "received": pixel_count,
        "occupancy": occupancy_result["occupancy"],
        "occupancy_raw_instant": occupancy_result.get("occupancy_raw_instant", occupancy_result["occupancy"]),
        "occupancy_effective_raw": occupancy_result.get("occupancy_effective_raw", occupancy_result["occupancy"]),
        "frame_valid": occupancy_result.get("frame_valid", True),
        "fever_count": occupancy_result.get("fever_count", 0),
        "elevated_count": occupancy_result.get("elevated_count", 0),
        "any_fever": occupancy_result.get("any_fever", False),
        "any_fever_raw": occupancy_result.get("any_fever_raw", False),
        "any_elevated": occupancy_result.get("any_elevated", False),
        "fever_consecutive_frames": occupancy_result.get("fever_consecutive_frames", 0),
        "elevated_consecutive_frames": occupancy_result.get("elevated_consecutive_frames", 0),
        "sustained_elevated": occupancy_result.get("sustained_elevated", False),
        "effective_fever_threshold": occupancy_result.get("effective_fever_threshold"),
    }


@app.get("/api/thermal/current/poll")
def get_thermal_data(
    sensor_id: Optional[str] = Query(default=None, description="If set, return latest for this sensor_id"),
) -> dict:
    """Return latest thermal data (expanded format with occupancy)."""
    if sensor_id:
        data = latest_thermal_by_sensor.get(sensor_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"No data available for sensor_id={sensor_id}")
        out = dict(data)
        out["last_update"] = last_update_time_by_sensor.get(sensor_id)
        occ = latest_occupancy_by_sensor.get(sensor_id)
        if occ:
            out["occupancy"] = occ.get("occupancy")
            out["occupancy_raw_instant"] = occ.get("occupancy_raw_instant", occ.get("occupancy"))
            out["occupancy_effective_raw"] = occ.get("occupancy_effective_raw", occ.get("occupancy"))
            out["frame_valid"] = occ.get("frame_valid", True)
            out["room_temperature"] = occ.get("room_temperature")
            out["people_clusters"] = convert_numpy_types(occ.get("people_clusters", []))
            out["fever_count"] = occ.get("fever_count", 0)
            out["elevated_count"] = occ.get("elevated_count", 0)
            out["any_fever"] = occ.get("any_fever", False)
            out["any_fever_raw"] = occ.get("any_fever_raw", False)
            out["any_elevated"] = occ.get("any_elevated", False)
            out["fever_consecutive_frames"] = occ.get("fever_consecutive_frames", 0)
            out["elevated_consecutive_frames"] = occ.get("elevated_consecutive_frames", 0)
            out["sustained_elevated"] = occ.get("sustained_elevated", False)
            out["effective_fever_threshold"] = occ.get("effective_fever_threshold")
            ml = occ.get("ml") or {}
            out["ml_occupancy"] = ml.get("ml_occupancy")
            out["ml_occupancy_confidence"] = ml.get("ml_occupancy_confidence")
            out["ml_fever"] = ml.get("ml_fever")
            out["ml_fever_confidence"] = ml.get("ml_fever_confidence")
        return out

    if latest_thermal_data is None:
        raise HTTPException(status_code=404, detail="No data available")
    out = dict(latest_thermal_data)
    out["last_update"] = last_update_time
    if latest_occupancy:
        out["occupancy"] = latest_occupancy["occupancy"]
        out["occupancy_raw_instant"] = latest_occupancy.get("occupancy_raw_instant", latest_occupancy["occupancy"])
        out["occupancy_effective_raw"] = latest_occupancy.get(
            "occupancy_effective_raw", latest_occupancy["occupancy"]
        )
        out["frame_valid"] = latest_occupancy.get("frame_valid", True)
        out["room_temperature"] = latest_occupancy.get("room_temperature")
        out["people_clusters"] = convert_numpy_types(latest_occupancy.get("people_clusters", []))
        out["fever_count"] = latest_occupancy.get("fever_count", 0)
        out["elevated_count"] = latest_occupancy.get("elevated_count", 0)
        out["any_fever"] = latest_occupancy.get("any_fever", False)
        out["any_fever_raw"] = latest_occupancy.get("any_fever_raw", False)
        out["any_elevated"] = latest_occupancy.get("any_elevated", False)
        out["fever_consecutive_frames"] = latest_occupancy.get("fever_consecutive_frames", 0)
        out["elevated_consecutive_frames"] = latest_occupancy.get("elevated_consecutive_frames", 0)
        out["sustained_elevated"] = latest_occupancy.get("sustained_elevated", False)
        out["effective_fever_threshold"] = latest_occupancy.get("effective_fever_threshold")
        ml = latest_occupancy.get("ml") or {}
        out["ml_occupancy"] = ml.get("ml_occupancy")
        out["ml_occupancy_confidence"] = ml.get("ml_occupancy_confidence")
        out["ml_fever"] = ml.get("ml_fever")
        out["ml_fever_confidence"] = ml.get("ml_fever_confidence")
    return out


@app.get("/api/thermal/current/all")
def get_all_thermal_data() -> dict:
    """Return latest thermal data for all sensors."""
    result = {}
    for sensor_id, data in latest_thermal_by_sensor.items():
        out = dict()
        meta = _sensor_metadata.get(sensor_id, {})
        out["building"] = meta.get("building") or data.get("building", "Other")
        out["room_name"] = meta.get("room_name")
        out["capacity"] = meta.get("capacity")
        out["last_update"] = last_update_time_by_sensor.get(sensor_id)
        occ = latest_occupancy_by_sensor.get(sensor_id)
        if occ:
            out["occupancy"] = occ.get("occupancy")
            out["room_temperature"] = occ.get("room_temperature")
            out["any_fever"] = occ.get("any_fever", False)
            capacity = meta.get("capacity")
            if capacity:
                occ_count = occ.get("occupancy") or 0
                out["occupancy_pct"] = round(occ_count / capacity, 3)
        result[sensor_id] = out
    return result


@app.get("/api/thermal/predicted/poll")
def get_predicted_thermal_data_poll(
    sensor_id: Optional[str] = Query(default=None, description="If set, return latest for this sensor_id"),
) -> dict:
    """Backward-compatible alias for ``GET /api/thermal/current/poll`` (latest reading, not a thermal forecast)."""
    return get_thermal_data(sensor_id)


@app.get("/api/thermal/predicted/all")
def get_predicted_thermal_data() -> dict:
    """Backward-compatible alias for ``GET /api/thermal/current/all`` (latest readings per sensor)."""
    return get_all_thermal_data()


@app.get("/api/occupancy/history")
def get_occupancy_history(
    date: str = Query(default=None, description="YYYYMMDD (default: today)"),
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
) -> dict:
    """Historical occupancy log entries for a date."""
    date_str = date or datetime.now().strftime("%Y%m%d")
    path = DATA_DIR / f"occupancy_{date_str}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No occupancy data found for date {date_str}")
    entries = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if sensor_id is None or entry.get("sensor_id") == sensor_id:
                entries.append(entry)
    return {"date": date_str, "sensor_id": sensor_id, "count": len(entries), "data": entries}


@app.get("/api/occupancy/stats")
def get_occupancy_stats(
    date: str = Query(default=None, description="YYYYMMDD (default: today)"),
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
) -> dict:
    """Occupancy statistics for a date."""
    date_str = date or datetime.now().strftime("%Y%m%d")
    path = DATA_DIR / f"occupancy_{date_str}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No occupancy data found for date {date_str}")
    values = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if sensor_id is None or entry.get("sensor_id") == sensor_id:
                values.append(entry["occupancy"])
    if not values:
        raise HTTPException(status_code=404, detail="No occupancy data available")
    avg_raw = sum(values) / len(values)
    current = values[-1]
    meta = _sensor_metadata.get(sensor_id or "", {}) if sensor_id else {}
    capacity = meta.get("capacity")
    result: dict = {
        "date": date_str,
        "sensor_id": sensor_id,
        "room_name": meta.get("room_name"),
        "building": meta.get("building"),
        "capacity": capacity,
        "total_readings": len(values),
        "min_occupancy": min(values),
        "max_occupancy": max(values),
        "avg_occupancy": round(avg_raw, 2),
        "current_occupancy": current,
        "occupancy_distribution": dict(Counter(values)),
    }
    if capacity:
        result["current_occupancy_pct"] = round(current / capacity, 3)
        result["avg_occupancy_pct"] = round(avg_raw / capacity, 3)
        result["max_occupancy_pct"] = round(max(values) / capacity, 3)
    return result


def _compute_occupancy_trends(
    date_str: str, sensor_id: Optional[str], bucket: str
) -> List[dict]:
    """Read occupancy log for date_str, bucket by hour or day, return list of bucket stats."""
    path = DATA_DIR / f"occupancy_{date_str}.jsonl"
    if not path.exists():
        return []
    buckets: Dict[Tuple, List[dict]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if sensor_id is not None and entry.get("sensor_id") != sensor_id:
                continue
            ts_str = entry.get("timestamp")
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if bucket == "hour":
                key = (dt.year, dt.month, dt.day, dt.hour)
            else:
                key = (dt.year, dt.month, dt.day)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(entry)
    out = []
    for key in sorted(buckets.keys()):
        entries = buckets[key]
        occupancies = [e["occupancy"] for e in entries]
        room_temps = [e["room_temperature"] for e in entries if e.get("room_temperature") is not None]
        if bucket == "hour":
            y, m, d, h = key
            bucket_start_dt = datetime(y, m, d, h, 0, 0)
            bucket_end_dt = bucket_start_dt + timedelta(hours=1)
            bucket_start = bucket_start_dt.isoformat()
            bucket_end = bucket_end_dt.isoformat()
        else:
            y, m, d = key
            bucket_start_dt = datetime(y, m, d, 0, 0, 0)
            bucket_end_dt = bucket_start_dt + timedelta(days=1)
            bucket_start = bucket_start_dt.isoformat()
            bucket_end = bucket_end_dt.isoformat()
        avg_room = round(sum(room_temps) / len(room_temps), 2) if room_temps else None
        out.append({
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "avg_occupancy": round(sum(occupancies) / len(occupancies), 2),
            "max_occupancy": max(occupancies),
            "sample_count": len(entries),
            "avg_room_temperature": avg_room,
        })
    return out


@app.get("/api/occupancy/trends")
def get_occupancy_trends(
    date: str = Query(default=None, description="YYYYMMDD (default: today)"),
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
    bucket: str = Query(default="hour", description="Aggregation bucket: hour | day"),
) -> dict:
    """Occupancy and room temperature aggregated by time bucket (hour or day)."""
    date_str = date or datetime.now().strftime("%Y%m%d")
    if bucket not in ("hour", "day"):
        raise HTTPException(status_code=400, detail="bucket must be 'hour' or 'day'")
    buckets_list = _compute_occupancy_trends(date_str, sensor_id, bucket)
    return {
        "date": date_str,
        "sensor_id": sensor_id,
        "bucket": bucket,
        "count": len(buckets_list),
        "data": buckets_list,
    }


def _predict_occupancy_heuristic(sensor_id: Optional[str], horizon_hours: int) -> List[dict]:
    """Predict occupancy using day-of-week + hour-of-day pattern from up to 28 days of history.

    For each target bucket we collect historical readings that share both the same
    weekday and the same hour (primary).  When fewer than 3 samples exist for that
    specific weekday+hour combination we fall back to using all readings for that
    hour across any weekday (secondary).  This produces better predictions for
    rooms with strong weekly schedules (e.g. a lecture hall empty on weekends).
    """
    now = datetime.now()
    # Collect readings keyed by (weekday 0=Mon, hour) and by hour alone (fallback).
    by_dow_hour: Dict[Tuple[int, int], List[int]] = {}
    by_hour: Dict[int, List[int]] = {h: [] for h in range(24)}
    for day_offset in range(1, 29):  # up to 4 weeks back
        d = now - timedelta(days=day_offset)
        date_str = d.strftime("%Y%m%d")
        path = DATA_DIR / f"occupancy_{date_str}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if sensor_id is not None and entry.get("sensor_id") != sensor_id:
                    continue
                ts_str = entry.get("timestamp")
                if not ts_str:
                    continue
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    continue
                val = entry["occupancy"]
                key = (dt.weekday(), dt.hour)
                if key not in by_dow_hour:
                    by_dow_hour[key] = []
                by_dow_hour[key].append(val)
                by_hour[dt.hour].append(val)

    def _stats(vals: List[int]) -> dict:
        avg = sum(vals) / len(vals)
        variance = sum((v - avg) ** 2 for v in vals) / len(vals)
        return {
            "expected_occupancy": round(avg, 2),
            "min_occupancy": min(vals),
            "max_occupancy": max(vals),
            "std_occupancy": round(variance ** 0.5, 2),
            "sample_count": len(vals),
        }

    out = []
    for i in range(horizon_hours):
        t = now + timedelta(hours=i)
        bucket_start = t.replace(minute=0, second=0, microsecond=0)
        bucket_end = bucket_start + timedelta(hours=1)
        key = (t.weekday(), t.hour)
        dow_vals = by_dow_hour.get(key, [])
        hour_vals = by_hour[t.hour]
        if len(dow_vals) >= 3:
            s = _stats(dow_vals)
            method = "weekday+hour"
        elif hour_vals:
            s = _stats(hour_vals)
            method = "hour"
        else:
            s = {"expected_occupancy": 0.0, "min_occupancy": 0, "max_occupancy": 0,
                 "std_occupancy": 0.0, "sample_count": 0}
            method = "no_data"
        out.append({
            "bucket_start": bucket_start.isoformat(),
            "bucket_end": bucket_end.isoformat(),
            "method": method,
            **s,
        })
    return out


@app.get("/api/occupancy/predict")
def get_occupancy_predict(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
    horizon_hours: int = Query(default=24, ge=1, le=168, description="Number of hours to predict (1..168)"),
) -> dict:
    """Predicted occupancy using day-of-week + hour pattern from up to 28 days of history."""
    predictions = _predict_occupancy_heuristic(sensor_id, horizon_hours)
    return {
        "sensor_id": sensor_id,
        "horizon_hours": horizon_hours,
        "count": len(predictions),
        "data": predictions,
    }


# ---------------------------------------------------------------------------
# Ground truth & accuracy endpoints
# ---------------------------------------------------------------------------

class _GroundTruthEntry(BaseModel):
    timestamp: str           # ISO 8601, e.g. "2026-04-17T14:30:00"
    sensor_id: str
    actual_occupancy: int    # true headcount at that moment
    actual_fever: Optional[bool] = None  # True if a fever was present


def _load_occupancy_history_for_accuracy(
    sensor_id: Optional[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[dict]:
    """Read all occupancy JSONL files, parse timestamps, optionally filter."""
    entries: List[dict] = []
    if not DATA_DIR.exists():
        return entries
    for path in sorted(DATA_DIR.glob("occupancy_*.jsonl")):
        if not path.is_file():
            continue
        # Early-prune by date embedded in filename (occupancy_YYYYMMDD.jsonl)
        if since is not None or until is not None:
            try:
                day_str = path.stem[len("occupancy_"):]
                file_day = datetime.strptime(day_str, "%Y%m%d").date()
                file_start = datetime.combine(file_day, datetime.min.time()).replace(tzinfo=timezone.utc)
                file_end = file_start + timedelta(days=1)
                if since is not None and file_end <= since:
                    continue
                if until is not None and file_start > until:
                    continue
            except ValueError:
                pass  # non-standard filename — don't skip, fall through to per-line filtering
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if sensor_id is not None and entry.get("sensor_id") != sensor_id:
                        continue
                    ts_str = entry.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        continue
                    if since is not None and dt < since:
                        continue
                    if until is not None and dt > until:
                        continue
                    entry["_dt"] = dt
                    entries.append(entry)
        except Exception:
            continue
    return entries


def _compute_accuracy_metrics(
    sensor_id: Optional[str],
    window_seconds: int,
    since: Optional[datetime],
    until: Optional[datetime],
) -> dict:
    """Compute RMSE, MAE, accuracy-within-1, binary precision/recall/F1 and fever metrics.

    Aligns each stored ground truth point to the nearest API readings within
    ``window_seconds``, averages multiple estimates in the window, then accumulates
    error statistics. Only ground truth entries matching ``sensor_id`` (when set)
    and inside the optional date range are included.
    """
    with _ground_truth_lock:
        all_gt = list(_ground_truth.values())
    gt_entries = [
        e for e in all_gt
        if (sensor_id is None or e.get("sensor_id") == sensor_id)
    ]
    if since or until:
        filtered = []
        for e in gt_entries:
            try:
                dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if since and dt < since:
                continue
            if until and dt > until:
                continue
            filtered.append(e)
        gt_entries = filtered

    if not gt_entries:
        return {"error": "no_ground_truth", "n_ground_truth": 0}

    history = _load_occupancy_history_for_accuracy(sensor_id, since, until)

    # Index history by sensor_id with sorted _dt lists for O(n log m) window lookups.
    history_by_sensor: Dict[str, List[dict]] = defaultdict(list)
    for h in history:
        history_by_sensor[h.get("sensor_id", "")].append(h)
    for entries_for_sensor in history_by_sensor.values():
        entries_for_sensor.sort(key=lambda x: x["_dt"])
    # Pre-extract sorted timestamps per sensor for bisect operations.
    sensor_timestamps: Dict[str, List[datetime]] = {
        sid: [h["_dt"] for h in entries_for_sensor]
        for sid, entries_for_sensor in history_by_sensor.items()
    }

    # --- Occupancy regression metrics ---
    sq_errors: List[float] = []
    abs_errors: List[float] = []
    within_1: List[bool] = []
    # Binary occupancy (occupied vs empty)
    occ_tp = occ_fp = occ_tn = occ_fn = 0
    # Fever binary metrics
    fv_tp = fv_fp = fv_tn = fv_fn = 0
    fv_gt_count = 0
    unmatched = 0

    for e in gt_entries:
        try:
            gt_dt = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            # Treat naive timestamps as UTC so comparison with tz-aware history works.
            if gt_dt.tzinfo is None:
                gt_dt = gt_dt.replace(tzinfo=timezone.utc)
        except Exception:
            unmatched += 1
            continue
        gt_sid = e.get("sensor_id", "")
        sensor_hist = history_by_sensor.get(gt_sid, [])
        if sensor_hist:
            window_delta = timedelta(seconds=window_seconds)
            ts_list = sensor_timestamps[gt_sid]
            lo = bisect.bisect_left(ts_list, gt_dt - window_delta)
            hi = bisect.bisect_right(ts_list, gt_dt + window_delta)
            candidates = sensor_hist[lo:hi]
        else:
            candidates = []
        if not candidates:
            unmatched += 1
            continue

        valid_occupancies: List[int] = []
        for h in candidates:
            try:
                valid_occupancies.append(int(h.get("occupancy", 0)))
            except (TypeError, ValueError):
                continue
        if not valid_occupancies:
            unmatched += 1
            continue

        actual = int(e["actual_occupancy"])
        est_avg = sum(valid_occupancies) / len(valid_occupancies)
        est = int(round(est_avg))

        err = est - actual
        sq_errors.append(err ** 2)
        abs_errors.append(abs(err))
        within_1.append(abs(err) <= 1)

        # Binary occupied/empty
        actual_occ = actual > 0
        pred_occ = est > 0
        if actual_occ and pred_occ:
            occ_tp += 1
        elif actual_occ and not pred_occ:
            occ_fn += 1
        elif not actual_occ and pred_occ:
            occ_fp += 1
        else:
            occ_tn += 1

        # Fever
        if e.get("actual_fever") is not None:
            fv_gt_count += 1
            actual_fv = bool(e["actual_fever"])
            pred_fv = any(bool(h.get("any_fever", False)) for h in candidates)
            if actual_fv and pred_fv:
                fv_tp += 1
            elif actual_fv and not pred_fv:
                fv_fn += 1
            elif not actual_fv and pred_fv:
                fv_fp += 1
            else:
                fv_tn += 1

    n = len(sq_errors)
    if n == 0:
        return {
            "error": "no_matches",
            "n_ground_truth": len(gt_entries),
            "n_unmatched": unmatched,
            "window_seconds": window_seconds,
        }

    rmse = (sum(sq_errors) / n) ** 0.5
    mae = sum(abs_errors) / n
    acc1 = sum(within_1) / n

    def _safe_div(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den > 0 else None

    occ_precision = _safe_div(occ_tp, occ_tp + occ_fp)
    occ_recall    = _safe_div(occ_tp, occ_tp + occ_fn)
    occ_f1 = (
        round(2 * occ_precision * occ_recall / (occ_precision + occ_recall), 4)
        if occ_precision is not None and occ_recall is not None and (occ_precision + occ_recall) > 0
        else None
    )

    result: dict = {
        "sensor_id": sensor_id,
        "n_ground_truth": len(gt_entries),
        "n_matched": n,
        "n_unmatched": unmatched,
        "window_seconds": window_seconds,
        "occupancy": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "accuracy_within_1": round(acc1, 4),
            "binary_occupied": {
                "tp": occ_tp, "fp": occ_fp, "tn": occ_tn, "fn": occ_fn,
                "precision": occ_precision,
                "recall": occ_recall,
                "f1": occ_f1,
            },
        },
    }

    if fv_gt_count > 0:
        fv_precision  = _safe_div(fv_tp, fv_tp + fv_fp)
        fv_recall     = _safe_div(fv_tp, fv_tp + fv_fn)
        fv_f1 = (
            round(2 * fv_precision * fv_recall / (fv_precision + fv_recall), 4)
            if fv_precision is not None and fv_recall is not None and (fv_precision + fv_recall) > 0
            else None
        )
        fv_sens = _safe_div(fv_tp, fv_tp + fv_fn)
        fv_spec = _safe_div(fv_tn, fv_tn + fv_fp)
        fv_bal = round((fv_sens + fv_spec) / 2, 4) if fv_sens is not None and fv_spec is not None else None
        result["fever"] = {
            "n_labeled": fv_gt_count,
            "tp": fv_tp, "fp": fv_fp, "tn": fv_tn, "fn": fv_fn,
            "precision": fv_precision,
            "recall": fv_recall,
            "f1": fv_f1,
            "balanced_accuracy": fv_bal,
        }

    return result


def _normalize_gt_timestamp(ts: str) -> str:
    """Parse an ISO 8601 timestamp string and return a UTC-normalised isoformat string.

    Raises ``ValueError`` for unparseable inputs so callers can return HTTP 400.
    """
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _require_gt_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Reject requests when GROUND_TRUTH_API_KEY is set and the header doesn't match."""
    if GROUND_TRUTH_API_KEY and x_api_key != GROUND_TRUTH_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key")


@app.post("/api/occupancy/groundtruth", dependencies=[Depends(_require_gt_api_key)])
def post_ground_truth(entry: _GroundTruthEntry) -> dict:
    """Submit a ground truth observation (actual headcount at a given timestamp).

    Re-submitting the same (timestamp, sensor_id) pair overwrites the earlier entry.
    Used to populate accuracy metrics via GET /api/occupancy/accuracy.
    """
    try:
        ts_norm = _normalize_gt_timestamp(entry.timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="timestamp must be a valid ISO 8601 string")
    if not 0 <= entry.actual_occupancy <= 50:
        raise HTTPException(status_code=400, detail="actual_occupancy must be 0–50")
    sid = entry.sensor_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="sensor_id must not be empty or whitespace")
    key = f"{ts_norm}|{sid}"
    record = {
        "timestamp": ts_norm,
        "sensor_id": sid,
        "actual_occupancy": entry.actual_occupancy,
        "actual_fever": entry.actual_fever,
        "ts_added": datetime.now(timezone.utc).isoformat(),
    }
    with _ground_truth_lock:
        _ground_truth[key] = record
    _persist_ground_truth()
    return {"status": "ok", "key": key, "entry": record}


@app.delete("/api/occupancy/groundtruth", dependencies=[Depends(_require_gt_api_key)])
def delete_ground_truth(
    timestamp: str = Query(..., description="ISO 8601 timestamp of the entry to remove"),
    sensor_id: str = Query(..., description="sensor_id of the entry to remove"),
) -> dict:
    """Remove a single ground truth entry by (timestamp, sensor_id)."""
    sid = sensor_id.strip()
    try:
        ts_norm = _normalize_gt_timestamp(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be a valid ISO 8601 string")
    key = f"{ts_norm}|{sid}"
    with _ground_truth_lock:
        if key not in _ground_truth:
            raise HTTPException(status_code=404, detail=f"No ground truth entry found for key: {key}")
        del _ground_truth[key]
    _persist_ground_truth()
    return {"status": "deleted", "key": key}


@app.get("/api/occupancy/groundtruth")
def get_ground_truth(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
) -> dict:
    """List stored ground truth entries."""
    with _ground_truth_lock:
        all_entries = list(_ground_truth.values())
    entries = [
        e for e in all_entries
        if sensor_id is None or e.get("sensor_id") == sensor_id
    ]
    entries.sort(key=lambda e: e.get("timestamp", ""))
    return {"count": len(entries), "sensor_id": sensor_id, "data": entries}


ACCURACY_DEFAULT_LOOKBACK_DAYS = int(os.environ.get("ACCURACY_DEFAULT_LOOKBACK_DAYS", "30"))


@app.get("/api/occupancy/accuracy")
def get_occupancy_accuracy(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id (default: all)"),
    window_seconds: int = Query(default=120, ge=10, le=900, description="Match window in seconds (default: 120)"),
    since: Optional[str] = Query(default=None, description="ISO 8601 start of evaluation range (inclusive); defaults to 30 days ago"),
    until: Optional[str] = Query(default=None, description="ISO 8601 end of evaluation range (inclusive)"),
) -> dict:
    """Compute occupancy accuracy metrics (RMSE, MAE, precision, recall, F1) against stored ground truth.

    Each ground truth point is matched to API readings that fall within
    ``window_seconds`` of the ground-truth timestamp, and the matched readings
    are averaged to produce the predicted occupancy value. The proposal
    evaluation metrics — RMSE between predicted and actual occupancy, and
    precision/recall for binary occupancy detection — are all returned here.
    Fever metrics are included when ground truth entries carry ``actual_fever``.

    When ``since`` is omitted the scan is limited to the last
    ``ACCURACY_DEFAULT_LOOKBACK_DAYS`` days (default 30) to bound I/O.
    Pass an explicit ``since`` value to extend the window.
    """
    since_dt = until_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="since must be a valid ISO 8601 string")
    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="until must be a valid ISO 8601 string")

    if since_dt is None:
        since_dt = datetime.now(timezone.utc) - timedelta(days=ACCURACY_DEFAULT_LOOKBACK_DAYS)

    result = _compute_accuracy_metrics(sensor_id, window_seconds, since_dt, until_dt)

    # Per-sensor breakdown when no sensor filter is set
    if sensor_id is None:
        with _ground_truth_lock:
            sensors_with_gt = {e.get("sensor_id") for e in _ground_truth.values() if e.get("sensor_id")}
        if len(sensors_with_gt) > 1:
            per_sensor = {}
            for sid in sorted(sensors_with_gt):
                per_sensor[sid] = _compute_accuracy_metrics(sid, window_seconds, since_dt, until_dt)
            result["per_sensor"] = per_sensor

    return result


# ---------------------------------------------------------------------------
# ML API endpoints
# ---------------------------------------------------------------------------

class _MLLabelRequest(BaseModel):
    file: str
    occupancy: int
    fever: bool


@app.get("/api/ml/status")
def get_ml_status() -> dict:
    """ML model status, label counts, and training progress."""
    occ_loaded = bool(_ml_engine and _ml_engine.occupancy_model_loaded)
    fever_loaded = bool(_ml_engine and _ml_engine.fever_model_loaded)
    with _ml_training_lock:
        labels_snap = list(_ml_labels.values())
        training_snap = dict(_ml_training_status)
    occ_dist: Dict[str, int] = {}
    for v in labels_snap:
        k = str(v.get("occupancy", 0))
        occ_dist[k] = occ_dist.get(k, 0) + 1
    return {
        "occupancy_model": occ_loaded,
        "fever_model": fever_loaded,
        "n_labels": len(labels_snap),
        "fever_label_count": sum(1 for v in labels_snap if v.get("fever")),
        "occupancy_distribution": occ_dist,
        "training": training_snap,
        "eval": training_snap.get("eval"),
        "split": training_snap.get("split"),
    }


@app.get("/api/ml/labels")
def get_ml_labels_api() -> dict:
    """Return all stored frame labels."""
    with _ml_training_lock:
        labels_snap = list(_ml_labels.values())
    return {"labels": labels_snap, "count": len(labels_snap)}


def _is_valid_thermal_filename(filename: str) -> bool:
    return filename.startswith("thermal_") and (
        filename.endswith("_compact.json") or filename.endswith("_compact.json.gz")
    )


@app.post("/api/ml/label")
def post_ml_label(req: _MLLabelRequest) -> dict:
    """Save or update a ground-truth label for a thermal frame file."""
    safe_file = Path(req.file).name
    if not safe_file:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not _is_valid_thermal_filename(safe_file):
        raise HTTPException(status_code=400, detail="file must match thermal_*_compact.json(.gz)")
    local_exists = (DATA_DIR / safe_file).is_file()
    if not local_exists:
        # Check Blob existence without downloading the full frame.
        container = _get_blob_container()
        blob_exists = False
        if container is not None:
            try:
                blob_exists = container.get_blob_client(f"thermal/{safe_file}").exists()
            except Exception:
                pass
        if not blob_exists:
            raise HTTPException(status_code=404, detail=f"Frame not found: {safe_file}")
    if not 0 <= req.occupancy <= 20:
        raise HTTPException(status_code=400, detail="occupancy must be 0–20")
    label_entry = {
        "file": safe_file,
        "occupancy": req.occupancy,
        "fever": req.fever,
        "ts": datetime.now().isoformat(),
    }
    with _ml_training_lock:
        _ml_labels[safe_file] = label_entry
    _persist_ml_labels()
    return {"status": "ok", "label": label_entry}


@app.get("/api/ml/infer")
def get_ml_infer(file: str = Query(..., description="Thermal frame filename")) -> dict:
    """Run heuristic and ML inference on a stored thermal frame and return both results."""
    safe_file = Path(file).name
    local_path = _ensure_local_copy(safe_file)
    if local_path is None:
        raise HTTPException(status_code=404, detail=f"Frame not found: {safe_file}")
    try:
        payload = _read_json_payload(local_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse frame: {e}")

    compact = payload.get("data") or payload
    sensor_id = payload.get("sensor_id")

    try:
        expanded = expand_thermal_data(compact)
        expanded["sensor_id"] = sensor_id
    except Exception:
        expanded = None

    try:
        heuristic = convert_numpy_types(estimate_occupancy(compact, sensor_id=sensor_id))
    except Exception as e:
        heuristic = {"error": str(e)}

    ml_result = None
    if _ml_engine and _ml_engine.available:
        try:
            arr = thermal_data_to_array(compact)
            bg = thermal_background_by_sensor.get(sensor_id) if sensor_id else None
            ml_result = _ml_engine.predict(arr, bg)
        except Exception as e:
            ml_result = {"error": str(e)}

    return {
        "file": safe_file,
        "sensor_id": sensor_id,
        "timestamp": payload.get("timestamp"),
        "frame": expanded,
        "heuristic": heuristic,
        "ml": ml_result,
        "existing_label": _ml_labels.get(safe_file),
    }


@app.post("/api/ml/train")
def post_ml_train() -> dict:
    """Trigger a background model training run from all labelled frames."""
    global _ml_training_status
    with _ml_training_lock:
        if _ml_training_status.get("state") in {"starting", "running"}:
            return {"status": "already_running", "training": dict(_ml_training_status)}
        label_count = len(_ml_labels)
        if label_count < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 10 labelled frames to train; have {label_count}.",
            )
        # Mark as starting inside the lock so a concurrent request sees it
        # before the thread has a chance to update it itself.
        _ml_training_status = {
            "state": "starting",
            "message": "Starting training thread…",
            "ts": datetime.now().isoformat(),
            "log": [],
        }
    t = threading.Thread(target=_run_training_thread_wrapper, daemon=True)
    t.start()
    return {"status": "started"}


def _run_training_thread() -> None:
    """Background thread: build features → train → export ONNX → reload engine."""
    global _ml_training_status
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        with _ml_training_lock:
            _ml_training_status = {**_ml_training_status, "message": msg, "log": list(log_lines)}
        print(f"[ML Train] {msg}")

    with _ml_training_lock:
        _ml_training_status = {"state": "running", "message": "Starting…", "ts": datetime.now().isoformat(), "log": []}

    try:
        from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415
        from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415
    except ImportError:
        with _ml_training_lock:
            _ml_training_status = {
                "state": "error",
                "message": "scikit-learn not installed. Add it to requirements.txt and redeploy.",
                "ts": datetime.now().isoformat(),
                "log": log_lines,
            }
        return

    try:
        from ml.features import extract as _ml_extract  # noqa: PLC0415
    except ImportError as e:
        with _ml_training_lock:
            _ml_training_status = {"state": "error", "message": f"ml package unavailable: {e}", "ts": datetime.now().isoformat(), "log": log_lines}
        return

    # Snapshot labels under lock so concurrent POST /api/ml/label can't mutate
    # the dict during iteration or produce inconsistent training data.
    with _ml_training_lock:
        labels_snap = list(_ml_labels.values())
    log(f"Building features for {len(labels_snap)} labelled frames…")

    # Each row is (frame_timestamp_str, feature_vec, occ_label, fever_label).
    # Sorting by frame timestamp gives a time-ordered dataset for the
    # 70/15/15 train/val/test split required by the project proposal.
    rows: List[Tuple[str, Any, int, int]] = []
    _tmp_paths: List[Path] = []

    for lbl in labels_snap:
        try:
            safe_file = Path(lbl["file"]).name
            path = _ensure_local_copy(safe_file)
            if path is None:
                log(f"  Skipping missing file: {lbl['file']}")
                continue
            if path.parent != DATA_DIR:
                _tmp_paths.append(path)
            payload = _read_json_payload(path)
            raw_ts = payload.get("timestamp", "")
            try:
                frame_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                # Fall back to the filename's embedded date (YYYYMMDD_HHMMSS) so the
                # split remains time-ordered even when the payload timestamp is absent.
                stem = Path(lbl["file"]).stem
                try:
                    frame_ts = datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")
                except ValueError:
                    frame_ts = datetime.min
            compact = payload.get("data") or payload
            arr = thermal_data_to_array(compact)
            bg = thermal_background_by_sensor.get(payload.get("sensor_id"))
            rows.append((frame_ts, _ml_extract(arr, bg), int(lbl["occupancy"]), 1 if lbl.get("fever") else 0))
        except Exception as exc:
            log(f"  Warning: {lbl['file']}: {exc}")

    for _p in _tmp_paths:
        try:
            _p.unlink()
        except Exception:
            pass

    # Sort by frame timestamp (oldest first) for time-based split.
    rows.sort(key=lambda r: r[0])
    n = len(rows)

    if n < 10:
        with _ml_training_lock:
            _ml_training_status = {
                "state": "error",
                "message": f"Only {n} usable feature vectors (need 10+). Check that labelled files still exist.",
                "ts": datetime.now().isoformat(),
                "log": log_lines,
            }
        return

    features   = [r[1] for r in rows]
    occ_targets   = [r[2] for r in rows]
    fever_targets = [r[3] for r in rows]

    X = np.array(features, dtype=np.float32)
    y_occ   = np.array(occ_targets)
    y_fever = np.array(fever_targets)
    out_dir = Path(os.environ.get("ML_MODEL_DIR", "ml_models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 70 / 15 / 15 time-based split (proposal specification).
    train_end = max(1, int(n * 0.70))
    val_end   = max(train_end + 1, int(n * 0.85))
    n_train, n_val, n_test = train_end, val_end - train_end, n - val_end
    log(f"Split: {n_train} train / {n_val} val / {n_test} test (time-ordered, 70/15/15)")

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_occ_train, y_occ_val, y_occ_test     = y_occ[:train_end], y_occ[train_end:val_end], y_occ[val_end:]
    y_fv_train,  y_fv_val,  y_fv_test      = y_fever[:train_end], y_fever[train_end:val_end], y_fever[val_end:]

    eval_results: dict = {}

    log(f"Training occupancy model on {n_train} samples (classes: {sorted(set(occ_targets[:train_end]))})…")
    occ_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    occ_clf.fit(X_train, y_occ_train)
    occ_train_acc = accuracy_score(y_occ_train, occ_clf.predict(X_train))
    log(f"  Train accuracy: {occ_train_acc:.3f}")
    occ_eval: dict = {"train_accuracy": round(occ_train_acc, 4)}
    if n_val > 0:
        occ_val_acc = accuracy_score(y_occ_val, occ_clf.predict(X_val))
        log(f"  Val   accuracy: {occ_val_acc:.3f}")
        occ_eval["val_accuracy"] = round(occ_val_acc, 4)
    if n_test > 0:
        occ_test_acc = accuracy_score(y_occ_test, occ_clf.predict(X_test))
        log(f"  Test  accuracy: {occ_test_acc:.3f}")
        occ_eval["test_accuracy"] = round(occ_test_acc, 4)
    eval_results["occupancy"] = occ_eval
    _export_onnx_model(occ_clf, X.shape[1], out_dir / "occupancy_model.onnx", "occupancy_model", log)

    fever_pos = int(y_fv_train.sum())
    if fever_pos >= 5:
        log(f"Training fever model ({fever_pos}/{n_train} positive training samples)…")
        fever_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        fever_clf.fit(X_train, y_fv_train)
        fv_train_f1 = f1_score(y_fv_train, fever_clf.predict(X_train), zero_division=0)
        log(f"  Train F1: {fv_train_f1:.3f}")
        fv_eval: dict = {"train_f1": round(fv_train_f1, 4)}
        if n_val > 0:
            fv_val_f1 = f1_score(y_fv_val, fever_clf.predict(X_val), zero_division=0)
            log(f"  Val   F1: {fv_val_f1:.3f}")
            fv_eval["val_f1"] = round(fv_val_f1, 4)
        if n_test > 0:
            fv_test_f1 = f1_score(y_fv_test, fever_clf.predict(X_test), zero_division=0)
            log(f"  Test  F1: {fv_test_f1:.3f}")
            fv_eval["test_f1"] = round(fv_test_f1, 4)
        eval_results["fever"] = fv_eval
        _export_onnx_model(fever_clf, X.shape[1], out_dir / "fever_model.onnx", "fever_model", log)
    else:
        log(f"Skipping fever model — need ≥5 positive training samples, have {fever_pos}")

    log("Reloading ML engine with new models…")
    if _ml_engine:
        _ml_engine.load(blob_container=_get_blob_container())

    with _ml_training_lock:
        _ml_training_status = {
            "state": "done",
            "message": f"Trained on {n_train} samples (val={n_val}, test={n_test}).",
            "ts": datetime.now().isoformat(),
            "log": log_lines,
            "eval": eval_results,
            "split": {"train": n_train, "val": n_val, "test": n_test},
        }


def _run_training_thread_wrapper() -> None:
    """Top-level wrapper so any uncaught exception in the training thread sets
    state to 'error' instead of silently leaving it stuck on 'running'."""
    global _ml_training_status
    try:
        _run_training_thread()
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        sanitized = f"{type(exc).__name__}: {exc}"
        print(f"[ML Train] Uncaught exception: {sanitized}\n{tb}")
        with _ml_training_lock:
            _ml_training_status = {
                "state": "error",
                "message": f"Unexpected error: {sanitized}",
                "ts": datetime.now().isoformat(),
                "log": _ml_training_status.get("log", []),
            }


def _export_onnx_model(clf, n_features: int, out_path: Path, name: str, log) -> None:
    try:
        from skl2onnx import convert_sklearn  # noqa: PLC0415
        from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415
    except ImportError:
        log(f"  skl2onnx not installed — skipping ONNX export for {name}")
        return
    try:
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_bytes = convert_sklearn(clf, name=name, initial_types=initial_type).SerializeToString()
        out_path.write_bytes(onnx_bytes)
        log(f"  Saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")
        if SAVE_TO_BLOB:
            _upload_blob(f"ml/{out_path.name}", onnx_bytes, content_type="application/octet-stream")
            log(f"  Uploaded to Azure Blob: ml/{out_path.name}")
    except Exception as exc:
        log(f"  ONNX export error for {name}: {exc}")


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def visualization_page() -> str:
    """Live thermal visualization dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermal Camera Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }
  .topnav { display:flex; align-items:center; gap:16px; margin-bottom:18px; }
  .topnav a { font-size:0.8rem; color:#6366f1; text-decoration:none; border:1px solid #6366f1; border-radius:5px; padding:4px 10px; }
  .topnav a:hover { background:#6366f1; color:#fff; }
  h1 { font-size: 1.25rem; font-weight: 600; color: #f8fafc; }
  .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  select { background: #1e2330; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 6px; padding: 6px 10px; font-size: 0.875rem; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
  .badge-occ  { background: #1e40af; color: #bfdbfe; }
  .badge-temp { background: #14532d; color: #bbf7d0; }
  .badge-fever { background: #7f1d1d; color: #fecaca; }
  .badge-elevated { background: #78350f; color: #fde68a; }
  .badge-ok { background: #1e293b; color: #94a3b8; }
  .badge-ml { background: #312e81; color: #c7d2fe; }
  .badge-ml-na { background: #1e293b; color: #475569; }
  .canvas-wrap { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  canvas { image-rendering: pixelated; border: 1px solid #2d3748; border-radius: 4px; }
  #legend { display: flex; flex-direction: column; gap: 4px; }
  .legend-bar { width: 20px; height: 160px; border-radius: 3px; background: linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff); }
  .legend-label { font-size: 0.7rem; color: #94a3b8; }
  #status { margin-top: 14px; font-size: 0.8rem; color: #64748b; }
  .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .stats-section { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .stats-divider { width: 1px; height: 20px; background: #2d3748; margin: 0 4px; }
  .stats-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: .05em; }
</style>
</head>
<body>
<div class="topnav">
  <h1>Thermal Camera Dashboard</h1>
  <a href="/ml">ML Studio</a>
</div>
<div class="toolbar">
  <label for="sensorSel" style="font-size:0.875rem;color:#94a3b8">Sensor:</label>
  <select id="sensorSel"><option value="">— any —</option></select>
  <label style="font-size:0.875rem;color:#94a3b8">
    <input type="checkbox" id="showClusters" checked style="margin-right:4px">
    Show people
  </label>
  <label style="font-size:0.875rem;color:#94a3b8">Scale:
    <select id="scaleSel">
      <option value="8">8×</option>
      <option value="12" selected>12×</option>
      <option value="16">16×</option>
      <option value="20">20×</option>
    </select>
  </label>
</div>
<div class="stats" id="stats"></div>
<div class="canvas-wrap">
  <canvas id="thermal"></canvas>
  <div id="legend">
    <div class="legend-label" id="maxLabel">—</div>
    <div class="legend-bar"></div>
    <div class="legend-label" id="minLabel">—</div>
  </div>
</div>
<div id="status">Connecting…</div>
<div id="accuracy-panel" style="margin-top:18px;display:none">
  <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">
    Accuracy vs Ground Truth
  </div>
  <div id="accuracy-content" style="display:flex;gap:10px;flex-wrap:wrap;font-size:0.8rem"></div>
  <div id="accuracy-footer" style="font-size:0.7rem;color:#475569;margin-top:4px"></div>
</div>

<script>
const canvas = document.getElementById('thermal');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const sensorSel = document.getElementById('sensorSel');
const scaleSel = document.getElementById('scaleSel');
const showClusters = document.getElementById('showClusters');

let pollTimer = null;

async function loadSensors() {
  try {
    const r = await fetch('/api/sensors');
    const j = await r.json();
    j.sensors.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      sensorSel.appendChild(opt);
    });
  } catch(e) {}
}

async function loadAccuracy() {
  const panel = document.getElementById('accuracy-panel');
  const content = document.getElementById('accuracy-content');
  const footer = document.getElementById('accuracy-footer');
  try {
    const sid = sensorSel.value || null;
    const url = '/api/occupancy/accuracy' + (sid ? '?sensor_id=' + encodeURIComponent(sid) : '');
    const r = await fetch(url);
    if (!r.ok) return;
    const j = await r.json();
    if (j.error || j.n_ground_truth === 0) { panel.style.display = 'none'; return; }
    panel.style.display = '';
    const occ = j.occupancy || {};
    const bin = occ.binary_occupied || {};
    const fv  = j.fever || null;
    const fmt = v => v != null ? (typeof v === 'number' ? v.toFixed(3) : v) : '—';
    const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';
    let html = '';
    const tile = (label, val, color) =>
      `<div style="background:#1e2330;border:1px solid #2d3748;border-radius:6px;padding:8px 14px;min-width:90px">
        <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em">${label}</div>
        <div style="font-size:1.1rem;font-weight:700;color:${color}">${val}</div>
      </div>`;
    html += tile('RMSE', fmt(occ.rmse), '#f8fafc');
    html += tile('MAE', fmt(occ.mae), '#f8fafc');
    html += tile('±1 Acc', pct(occ.accuracy_within_1), '#34d399');
    html += tile('Precision', pct(bin.precision), '#93c5fd');
    html += tile('Recall', pct(bin.recall), '#93c5fd');
    html += tile('F1', pct(bin.f1), '#a78bfa');
    if (fv) {
      html += tile('Fever F1', pct(fv.f1), '#f87171');
      html += tile('Fever Recall', pct(fv.recall), '#f87171');
    }
    content.innerHTML = html;
    footer.textContent = `${j.n_matched} of ${j.n_ground_truth} GT points matched (±${j.window_seconds}s window)`;
  } catch(e) { panel.style.display = 'none'; }
}

function badge(cls, text) {
  return `<span class="badge ${cls}">${text}</span>`;
}

function renderFrame(data) {
  const scale = parseInt(scaleSel.value, 10);
  const pixels = data.pixels || [];
  if (!pixels.length) return;

  const W = data.width, H = data.height;
  canvas.width  = W * scale;
  canvas.height = H * scale;

  // Draw thermal pixels
  pixels.forEach(p => {
    ctx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
    ctx.fillRect(p.col * scale, p.row * scale, scale, scale);
  });

  // Cluster overlays
  if (showClusters.checked) {
    const clusters = data.people_clusters || [];
    clusters.forEach(c => {
      const [row, col] = c.center;
      const cx = col * scale + scale / 2;
      const cy = row * scale + scale / 2;
      const r  = Math.sqrt(c.size) * scale * 0.55;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.strokeStyle = c.fever_detected ? '#ff2222' : c.elevated_temp ? '#ffaa00' : '#00ff88';
      ctx.lineWidth = 2;
      ctx.stroke();
      // temp label
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.max(10, scale - 2)}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(c.representative_temp_c.toFixed(1) + '°', cx, cy);
    });
  }

  // Legend
  document.getElementById('maxLabel').textContent = data.max_temp?.toFixed(1) + ' °C';
  document.getElementById('minLabel').textContent = data.min_temp?.toFixed(1) + ' °C';

  // Stats badges
  const occ  = data.occupancy ?? '—';
  const temp = data.room_temperature != null ? data.room_temperature.toFixed(1) + ' °C' : '—';
  const feverThreshStr = data.effective_fever_threshold != null
    ? ` ≥${data.effective_fever_threshold.toFixed(1)}°C` : '';
  const feverBadge = data.any_fever
    ? badge('badge-fever', `Fever detected${feverThreshStr}`)
    : data.sustained_elevated
      ? badge('badge-elevated', 'Sustained elevated temp')
      : data.any_elevated
        ? badge('badge-elevated', 'Elevated temp')
        : badge('badge-ok', 'No fever');

  // ML inference badges
  let mlBadge = '';
  if (data.ml_occupancy != null) {
    const conf = data.ml_occupancy_confidence != null
      ? ` (${(data.ml_occupancy_confidence * 100).toFixed(0)}%)`
      : '';
    mlBadge = badge('badge-ml', `ML: ${data.ml_occupancy} ${data.ml_occupancy === 1 ? 'person' : 'people'}${conf}`);
  } else {
    mlBadge = badge('badge-ml-na', 'ML: no model');
  }

  statsEl.innerHTML =
    '<span class="stats-label">Heuristic</span> ' +
    badge('badge-occ',  `${occ} ${occ === 1 ? 'person' : 'people'}`) + ' ' +
    badge('badge-temp', `Room ${temp}`) + ' ' +
    feverBadge +
    (data.frame_valid === false ? ' ' + badge('badge-elevated', 'Frame invalid') : '') +
    '<span class="stats-divider"></span>' +
    '<span class="stats-label">ML</span> ' +
    mlBadge +
    (data.ml_fever != null ? ' ' + (data.ml_fever ? badge('badge-fever', 'ML fever') : badge('badge-ok', 'ML no fever')) : '');

  const ts = data.last_update ? new Date(data.last_update).toLocaleTimeString('en-US', { timeZone: 'America/New_York' }) + ' EST' : '—';
  statusEl.textContent = `Last update: ${ts}  ·  ${W}×${H} px`;
}

async function poll() {
  const sid = sensorSel.value;
  const url = '/api/thermal/current/poll' + (sid ? `?sensor_id=${encodeURIComponent(sid)}` : '');
  try {
    const r = await fetch(url);
    if (r.status === 404) {
      statusEl.textContent = 'No data yet — waiting for a sensor to post.';
    } else if (!r.ok) {
      statusEl.textContent = `Error ${r.status}`;
    } else {
      renderFrame(await r.json());
    }
  } catch(e) {
    statusEl.textContent = `Fetch error: ${e.message}`;
  }
  pollTimer = setTimeout(poll, 10000);
}

function restart() {
  clearTimeout(pollTimer);
  poll();
}

sensorSel.addEventListener('change', () => { restart(); loadAccuracy(); });
scaleSel.addEventListener('change', () => { /* re-render on next poll */ });
showClusters.addEventListener('change', restart);

loadSensors();
loadAccuracy();
poll();
</script>
</body>
</html>"""


@app.get("/ml", response_class=HTMLResponse)
def ml_studio_page() -> str:
    """ML Studio — label frames, train models, run inference."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Studio</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
.topbar { display:flex; align-items:center; gap:14px; padding:14px 20px; border-bottom:1px solid #1e2330; }
.topbar h1 { font-size:1.1rem; font-weight:600; color:#f8fafc; flex:1; }
.topbar a { font-size:0.8rem; color:#6366f1; text-decoration:none; border:1px solid #6366f1; border-radius:5px; padding:4px 10px; }
.topbar a:hover { background:#6366f1; color:#fff; }
.tabs { display:flex; gap:0; border-bottom:1px solid #1e2330; padding:0 20px; }
.tab-btn { padding:10px 18px; font-size:0.875rem; color:#94a3b8; background:none; border:none; border-bottom:2px solid transparent; cursor:pointer; }
.tab-btn.active { color:#6366f1; border-bottom-color:#6366f1; }
.tab-btn:hover:not(.active) { color:#e2e8f0; }
.tab-panel { display:none; padding:20px; }
.tab-panel.active { display:flex; flex-direction:column; gap:16px; }
/* Status cards */
.cards { display:flex; gap:12px; flex-wrap:wrap; }
.card { background:#1a1f2e; border:1px solid #2d3748; border-radius:8px; padding:14px 18px; min-width:160px; }
.card-title { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }
.card-value { font-size:1.4rem; font-weight:700; }
.good { color:#4ade80; }
.warn { color:#facc15; }
.bad  { color:#f87171; }
.idle { color:#64748b; }
/* Toolbar */
.toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
select, input[type=date] { background:#1e2330; color:#e2e8f0; border:1px solid #2d3748; border-radius:6px; padding:6px 10px; font-size:0.875rem; }
button { border:none; border-radius:6px; padding:7px 14px; font-size:0.875rem; cursor:pointer; font-weight:500; }
.btn-primary { background:#6366f1; color:#fff; }
.btn-primary:hover { background:#4f46e5; }
.btn-primary:disabled { background:#374151; color:#6b7280; cursor:not-allowed; }
.btn-success { background:#16a34a; color:#fff; }
.btn-success:hover { background:#15803d; }
.btn-success:disabled { background:#374151; color:#6b7280; cursor:not-allowed; }
.btn-danger { background:#dc2626; color:#fff; }
.btn-danger:hover { background:#b91c1c; }
/* Split layout */
.split { display:flex; gap:16px; align-items:flex-start; }
.frame-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(88px,1fr)); gap:8px; max-height:65vh; overflow-y:auto; min-width:200px; }
.thumb-wrap { position:relative; cursor:pointer; border:2px solid transparent; border-radius:5px; overflow:hidden; }
.thumb-wrap:hover { border-color:#6366f1; }
.thumb-wrap.selected { border-color:#6366f1; }
.thumb-wrap canvas { display:block; width:100%; height:auto; image-rendering:pixelated; }
.label-dot { position:absolute; top:3px; right:3px; width:10px; height:10px; border-radius:50%; border:1px solid #0f1117; }
.dot-labelled { background:#4ade80; }
.dot-fever { background:#f87171; }
.detail-panel { flex:1; min-width:260px; background:#1a1f2e; border:1px solid #2d3748; border-radius:8px; padding:16px; display:flex; flex-direction:column; gap:12px; }
.detail-panel canvas { image-rendering:pixelated; border:1px solid #2d3748; border-radius:4px; max-width:100%; }
.field-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.field-row label { font-size:0.8rem; color:#94a3b8; }
input[type=number] { width:70px; background:#1e2330; color:#e2e8f0; border:1px solid #2d3748; border-radius:6px; padding:5px 8px; font-size:0.875rem; }
input[type=checkbox] { width:16px; height:16px; accent-color:#f87171; cursor:pointer; }
.result-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.result-table th { text-align:left; color:#64748b; padding:4px 8px; border-bottom:1px solid #2d3748; }
.result-table td { padding:5px 8px; border-bottom:1px solid #1e2330; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-yes { background:#7f1d1d; color:#fca5a5; }
.badge-no  { background:#1e293b; color:#94a3b8; }
.badge-na  { background:#292524; color:#78716c; }
/* Log */
.log-box { background:#0a0c12; border:1px solid #1e2330; border-radius:6px; padding:10px; font-family:monospace; font-size:0.75rem; color:#94a3b8; max-height:220px; overflow-y:auto; white-space:pre-wrap; }
.empty-msg { color:#475569; font-size:0.85rem; text-align:center; padding:24px; }
.section-title { font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
</style>
</head>
<body>
<div class="topbar">
  <h1>ML Studio</h1>
  <a href="/">Dashboard</a>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('status')">Status &amp; Train</button>
  <button class="tab-btn" onclick="switchTab('label')">Label Data</button>
  <button class="tab-btn" onclick="switchTab('infer')">Run Inference</button>
</div>

<!-- STATUS TAB -->
<div id="tab-status" class="tab-panel active">
  <div class="cards" id="statusCards">
    <div class="card"><div class="card-title">Occupancy Model</div><div class="card-value idle" id="cardOcc">—</div></div>
    <div class="card"><div class="card-title">Fever Model</div><div class="card-value idle" id="cardFever">—</div></div>
    <div class="card"><div class="card-title">Labelled Frames</div><div class="card-value idle" id="cardLabels">—</div></div>
    <div class="card"><div class="card-title">Fever Labels</div><div class="card-value idle" id="cardFeverLabels">—</div></div>
  </div>
  <div>
    <div class="section-title" style="margin-bottom:8px">Occupancy Distribution in Labels</div>
    <div id="occDist" style="font-size:0.8rem;color:#94a3b8">—</div>
  </div>
  <div class="toolbar">
    <button class="btn-primary" id="trainBtn" onclick="startTraining()" disabled>Train Model</button>
    <button class="btn-primary" onclick="refreshStatus()">Refresh Status</button>
    <span id="trainMsg" style="font-size:0.8rem;color:#94a3b8"></span>
  </div>
  <div>
    <div class="section-title" style="margin-bottom:6px">Training Log</div>
    <div class="log-box" id="trainLog">No training run yet.</div>
  </div>
</div>

<!-- LABEL TAB -->
<div id="tab-label" class="tab-panel">
  <div class="toolbar">
    <label style="font-size:0.875rem;color:#94a3b8">Sensor:</label>
    <select id="labelSensorSel"><option value="">— all —</option></select>
    <label style="font-size:0.875rem;color:#94a3b8">Date:</label>
    <input type="date" id="labelDateInput">
    <button class="btn-primary" onclick="loadFrames('label')">Load Frames</button>
    <span id="labelCount" style="font-size:0.8rem;color:#94a3b8"></span>
  </div>
  <div class="split">
    <div>
      <div class="frame-grid" id="labelGrid"><div class="empty-msg">Load frames to start labelling.</div></div>
    </div>
    <div class="detail-panel" id="labelDetail">
      <div class="empty-msg">Select a frame from the grid.</div>
    </div>
  </div>
</div>

<!-- INFER TAB -->
<div id="tab-infer" class="tab-panel">
  <div class="toolbar">
    <label style="font-size:0.875rem;color:#94a3b8">Sensor:</label>
    <select id="inferSensorSel"><option value="">— all —</option></select>
    <label style="font-size:0.875rem;color:#94a3b8">Date:</label>
    <input type="date" id="inferDateInput">
    <button class="btn-primary" onclick="loadFrames('infer')">Load Frames</button>
  </div>
  <div class="split">
    <div>
      <div class="frame-grid" id="inferGrid"><div class="empty-msg">Load frames to select one for inference.</div></div>
    </div>
    <div class="detail-panel" id="inferDetail">
      <div class="empty-msg">Select a frame from the grid.</div>
    </div>
  </div>
</div>

<script>
// ─── State ───────────────────────────────────────────────────────────────────
let labelFrames = [], inferFrames = [];
let labelSelected = null, inferSelected = null;
let labelMap = {};   // file → {occupancy, fever}
let statusPoll = null;

// ─── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['status','label','infer'][i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

// ─── Sensors ──────────────────────────────────────────────────────────────────
async function loadSensors() {
  try {
    const j = await (await fetch('/api/sensors')).json();
    for (const sel of [document.getElementById('labelSensorSel'), document.getElementById('inferSensorSel')]) {
      j.sensors.forEach(s => {
        const o = document.createElement('option');
        o.value = s; o.textContent = s;
        sel.appendChild(o);
      });
    }
  } catch(e) {}
}

// ─── Status ───────────────────────────────────────────────────────────────────
async function refreshStatus() {
  try {
    const j = await (await fetch('/api/ml/status')).json();
    const set = (id, val, cls) => {
      const el = document.getElementById(id);
      el.textContent = val;
      el.className = 'card-value ' + cls;
    };
    set('cardOcc',        j.occupancy_model ? 'Loaded' : 'No model', j.occupancy_model ? 'good' : 'warn');
    set('cardFever',      j.fever_model     ? 'Loaded' : 'No model', j.fever_model     ? 'good' : 'warn');
    set('cardLabels',     j.n_labels,        j.n_labels >= 10 ? 'good' : 'warn');
    set('cardFeverLabels',j.fever_label_count, j.fever_label_count >= 5 ? 'good' : 'idle');

    const dist = j.occupancy_distribution || {};
    document.getElementById('occDist').textContent =
      Object.keys(dist).sort((a,b)=>+a-+b).map(k => k + ' people: ' + dist[k]).join('  |  ') || 'No labels yet.';

    const trainBtn = document.getElementById('trainBtn');
    const trainMsg = document.getElementById('trainMsg');
    const t = j.training || {};
    if (t.state === 'running') {
      trainBtn.disabled = true;
      trainMsg.textContent = t.message || 'Training…';
      if (!statusPoll) statusPoll = setInterval(refreshStatus, 2000);
    } else {
      trainBtn.disabled = j.n_labels < 10;
      trainMsg.textContent = t.state === 'done' ? ('Done: ' + t.message) : t.state === 'error' ? ('Error: ' + t.message) : '';
      if (statusPoll && t.state !== 'running') { clearInterval(statusPoll); statusPoll = null; }
    }

    const logEl = document.getElementById('trainLog');
    const lines = t.log || [];
    logEl.textContent = lines.length ? lines.join('\\n') : (t.state === 'idle' ? 'No training run yet.' : t.message || '');
    logEl.scrollTop = logEl.scrollHeight;

    // Update label map from server
    try {
      const lj = await (await fetch('/api/ml/labels')).json();
      labelMap = {};
      (lj.labels || []).forEach(l => { labelMap[l.file] = l; });
      // refresh grids if frames loaded
      if (labelFrames.length) renderGrid('label');
      if (inferFrames.length) renderGrid('infer');
    } catch(e) {}
  } catch(e) {}
}

async function startTraining() {
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('trainMsg').textContent = 'Starting…';
  try {
    await fetch('/api/ml/train', {method:'POST'});
    statusPoll = setInterval(refreshStatus, 2000);
    await refreshStatus();
  } catch(e) {
    document.getElementById('trainMsg').textContent = 'Error: ' + e.message;
  }
}

// ─── Frame loading ────────────────────────────────────────────────────────────
async function loadFrames(tab) {
  const sensor = document.getElementById(tab + 'SensorSel').value;
  const dateVal = document.getElementById(tab + 'DateInput').value;
  const date = dateVal ? dateVal.replace(/-/g,'') : '';
  const params = new URLSearchParams({include_data: 'true', limit: '60'});
  if (sensor) params.set('sensor_id', sensor);
  if (date)   params.set('date', date);
  try {
    const j = await (await fetch('/api/thermal/history?' + params)).json();
    const frames = j.data || [];
    if (tab === 'label') {
      labelFrames = frames;
      if (tab === 'label') document.getElementById('labelCount').textContent = frames.length + ' frames loaded';
      renderGrid('label');
    } else {
      inferFrames = frames;
      renderGrid('infer');
    }
  } catch(e) {
    console.error('loadFrames error', e);
  }
}

// ─── Thermal canvas draw ──────────────────────────────────────────────────────
function drawThermal(canvas, frameData, scale) {
  if (!frameData || !frameData.pixels) return;
  const W = frameData.width || 32, H = frameData.height || 24;
  canvas.width = W * scale; canvas.height = H * scale;
  const ctx = canvas.getContext('2d');
  frameData.pixels.forEach(p => {
    ctx.fillStyle = 'rgb(' + p.r + ',' + p.g + ',' + p.b + ')';
    ctx.fillRect(p.col * scale, p.row * scale, scale, scale);
  });
}

function drawThermalFromPixelArray(canvas, pixels, W, H, scale) {
  canvas.width = W * scale; canvas.height = H * scale;
  const ctx = canvas.getContext('2d');
  pixels.forEach(p => {
    ctx.fillStyle = 'rgb(' + p.r + ',' + p.g + ',' + p.b + ')';
    ctx.fillRect(p.col * scale, p.row * scale, scale, scale);
  });
}

// ─── Grid rendering ────────────────────────────────────────────────────────────
function renderGrid(tab) {
  const frames = tab === 'label' ? labelFrames : inferFrames;
  const grid = document.getElementById(tab + 'Grid');
  const selected = tab === 'label' ? labelSelected : inferSelected;
  grid.innerHTML = '';
  if (!frames.length) { grid.innerHTML = '<div class="empty-msg">No frames found.</div>'; return; }

  frames.forEach(frame => {
    const wrap = document.createElement('div');
    wrap.className = 'thumb-wrap' + (frame.file === selected ? ' selected' : '');
    wrap.onclick = () => selectFrame(tab, frame);

    const c = document.createElement('canvas');
    wrap.appendChild(c);
    if (frame.data) drawThermal(c, frame.data, 2);

    const lbl = labelMap[frame.file];
    if (lbl) {
      const dot = document.createElement('div');
      dot.className = 'label-dot ' + (lbl.fever ? 'dot-fever' : 'dot-labelled');
      dot.title = 'Labelled: ' + lbl.occupancy + ' person(s)' + (lbl.fever ? ', fever' : '');
      wrap.appendChild(dot);
    }

    const ts = document.createElement('div');
    ts.style.cssText = 'font-size:0.6rem;color:#475569;padding:2px 3px;background:#0f1117;';
    ts.textContent = frame.timestamp ? new Date(frame.timestamp).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false }) : '';
    wrap.appendChild(ts);

    grid.appendChild(wrap);
  });
}

// ─── Frame selection ──────────────────────────────────────────────────────────
function selectFrame(tab, frame) {
  if (tab === 'label') labelSelected = frame.file;
  else                 inferSelected = frame.file;
  renderGrid(tab);

  const panel = document.getElementById(tab + 'Detail');
  const existing = labelMap[frame.file];

  const c = document.createElement('canvas');
  if (frame.data) drawThermal(c, frame.data, 8);

  const meta = document.createElement('div');
  meta.style.cssText = 'font-size:0.75rem;color:#64748b;';
  const fmtTs = frame.timestamp ? new Date(frame.timestamp).toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false }) : '';
  meta.textContent = frame.sensor_id + '  ·  ' + fmtTs;

  if (tab === 'label') {
    const occVal = existing ? existing.occupancy : 0;
    const feverVal = existing ? existing.fever : false;
    panel.innerHTML = '';
    panel.appendChild(c);
    panel.appendChild(meta);
    panel.insertAdjacentHTML('beforeend', `
      <div class="section-title">Label This Frame</div>
      <div class="field-row">
        <label>Occupancy:</label>
        <input type="number" id="occInput" value="${occVal}" min="0" max="20">
        <label><input type="checkbox" id="feverCb" ${feverVal?'checked':''}> Fever detected</label>
      </div>
      <button class="btn-success" onclick="saveLabel('${frame.file}')">Save Label</button>
      ${existing ? '<div style="font-size:0.75rem;color:#4ade80">Currently labelled: ' + existing.occupancy + ' person(s)' + (existing.fever ? ', fever' : '') + '</div>' : ''}
    `);
  } else {
    panel.innerHTML = '';
    panel.appendChild(c);
    panel.appendChild(meta);
    panel.insertAdjacentHTML('beforeend', `
      <button class="btn-primary" id="inferBtn" onclick="runInference('${frame.file}')">Run Inference</button>
      <div id="inferResults"></div>
    `);
  }
}

// ─── Labelling ─────────────────────────────────────────────────────────────────
async function saveLabel(file) {
  const occ = parseInt(document.getElementById('occInput').value) || 0;
  const fever = document.getElementById('feverCb').checked;
  try {
    await fetch('/api/ml/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file, occupancy: occ, fever})
    });
    labelMap[file] = {file, occupancy: occ, fever};
    renderGrid('label');
    // Re-select to refresh panel
    const frame = labelFrames.find(f => f.file === file);
    if (frame) selectFrame('label', frame);
    await refreshStatus();
  } catch(e) { alert('Save failed: ' + e.message); }
}

// ─── Inference ────────────────────────────────────────────────────────────────
async function runInference(file) {
  const btn = document.getElementById('inferBtn');
  btn.disabled = true; btn.textContent = 'Running…';
  try {
    const j = await (await fetch('/api/ml/infer?file=' + encodeURIComponent(file))).json();
    const h = j.heuristic || {};
    const m = j.ml || {};
    const lbl = j.existing_label;

    const fmtFever = v => v == null ? '<span class="badge badge-na">N/A</span>' :
      v ? '<span class="badge badge-yes">Fever</span>' : '<span class="badge badge-no">None</span>';
    const fmtOcc = v => v == null ? '—' : v;

    const results = document.getElementById('inferResults');
    results.innerHTML = `
      <table class="result-table">
        <tr><th></th><th>Occupancy</th><th>Fever</th><th>Confidence</th></tr>
        <tr><td style="color:#94a3b8">Heuristic</td>
            <td>${fmtOcc(h.occupancy)}</td>
            <td>${fmtFever(h.any_fever)}</td>
            <td>—</td></tr>
        <tr><td style="color:#a78bfa">ML Model</td>
            <td>${fmtOcc(m.ml_occupancy)}</td>
            <td>${fmtFever(m.ml_fever)}</td>
            <td>${m.ml_occupancy_confidence != null ? (m.ml_occupancy_confidence*100).toFixed(0)+'%' : '—'}</td></tr>
        ${lbl ? '<tr><td style="color:#4ade80">Ground Truth</td><td>'+lbl.occupancy+'</td><td>'+fmtFever(lbl.fever)+'</td><td>—</td></tr>' : ''}
      </table>
      ${m.ml_occupancy == null ? '<div style="font-size:0.75rem;color:#f87171;margin-top:6px">No ML model loaded. Train a model first.</div>' : ''}
    `;
  } catch(e) {
    document.getElementById('inferResults').innerHTML = '<div style="color:#f87171">Error: ' + e.message + '</div>';
  } finally {
    btn.disabled = false; btn.textContent = 'Run Inference';
  }
}

// ─── Init ──────────────────────────────────────────────────────────────────────
loadSensors();
refreshStatus();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
