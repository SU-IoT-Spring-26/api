#!/usr/bin/env python3
"""
Occupancy API – FastAPI data reception and storage for thermal/occupancy data.
Designed for Azure App Service (resource group: occupancy-rg, app: occupancy-api).
Stores data locally and optionally to Azure Blob Storage when configured.
"""

import json
import os
import gzip
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
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
    """Reload the most recent thermal frame and occupancy result per sensor from disk after a restart."""
    global latest_thermal_data, last_update_time, latest_occupancy

    if not DATA_DIR.exists():
        return

    # --- Thermal frames: latest frame per sensor from stored compact/expanded files ---
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

    for sid, (ts, data) in best_thermal.items():
        latest_thermal_by_sensor[sid] = data
        last_update_time_by_sensor[sid] = ts
        if latest_thermal_data is None or ts > (last_update_time or ""):
            latest_thermal_data = data
            last_update_time = ts

    # --- Occupancy: latest entry per sensor from daily jsonl files ---
    best_occ: Dict[str, Tuple[str, dict]] = {}  # sensor_id -> (timestamp_str, entry)
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

    for sid, (ts, entry) in best_occ.items():
        latest_occupancy_by_sensor[sid] = entry
        if sid in latest_thermal_by_sensor and ts > (last_update_time_by_sensor.get(sid) or ""):
            last_update_time_by_sensor[sid] = ts
        if latest_occupancy is None or ts > (last_update_time or ""):
            latest_occupancy = entry

    if best_thermal or best_occ:
        sensors = set(best_thermal) | set(best_occ)
        print(f"Restored state from disk for {len(sensors)} sensor(s): {sorted(sensors)}")

    _load_ml_labels()  # called unconditionally — handles missing DATA_DIR gracefully


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

# Occupancy detection parameters
MIN_HUMAN_TEMP = 30.0
MAX_HUMAN_TEMP = 45.0
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 200
ROOM_TEMP_THRESHOLD = float(os.environ.get("ROOM_TEMP_THRESHOLD", "0.5"))
FEVER_THRESHOLD_C = float(os.environ.get("FEVER_THRESHOLD_C", "37.5"))
FEVER_ELEVATED_THRESHOLD_C = float(os.environ.get("FEVER_ELEVATED_THRESHOLD_C", "37.0"))
FEVER_MIN_CONSECUTIVE_FRAMES = max(1, int(os.environ.get("FEVER_MIN_CONSECUTIVE_FRAMES", "2")))

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

occupancy_raw_history_by_sensor: Dict[str, deque] = {}
last_frame_median_by_sensor: Dict[str, float] = {}
last_raw_occupancy_by_sensor: Dict[str, int] = {}
last_smoothed_occupancy_by_sensor: Dict[str, int] = {}
fever_consecutive_by_sensor: Dict[str, int] = {}

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
_ml_training_status: dict = {"state": "idle", "message": "", "ts": None, "log": []}
_ml_training_lock = threading.Lock()


def _load_ml_labels() -> None:
    """Load persisted ML frame labels from disk into _ml_labels.

    Called unconditionally at startup — DATA_DIR may not exist in Blob-only
    mode, in which case local load is simply skipped.
    """
    global _ml_labels
    if not DATA_DIR.exists():
        return
    path = DATA_DIR / "ml_labels.jsonl"
    if not path.exists():
        return
    loaded: Dict[str, dict] = {}
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
    except Exception as e:
        print(f"Could not load ML labels: {e}")


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


def find_people_clusters(human_mask: np.ndarray, temp_array: np.ndarray) -> List[Dict]:
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(human_mask, structure=structure)
    people_clusters = []
    for i in range(1, num_features + 1):
        cluster_size = int(np.sum(labeled_array == i))
        if MIN_CLUSTER_SIZE <= cluster_size <= MAX_CLUSTER_SIZE:
            cluster_pixels = np.where(labeled_array == i)
            center_row = int(np.mean(cluster_pixels[0]))
            center_col = int(np.mean(cluster_pixels[1]))
            cluster_temps = temp_array[cluster_pixels[0], cluster_pixels[1]]
            representative_temp_c = float(np.percentile(cluster_temps, 90)) if cluster_temps.size else 0.0
            fever_detected = representative_temp_c >= FEVER_THRESHOLD_C
            elevated_temp = (
                FEVER_ELEVATED_THRESHOLD_C > 0
                and representative_temp_c >= FEVER_ELEVATED_THRESHOLD_C
                and representative_temp_c < FEVER_THRESHOLD_C
            )
            people_clusters.append({
                "id": i,
                "size": cluster_size,
                "center": (center_row, center_col),
                "representative_temp_c": round(representative_temp_c, 2),
                "elevated_temp": elevated_temp,
                "fever_detected": fever_detected,
            })
    return people_clusters


def estimate_occupancy(thermal_data: dict, sensor_id: Optional[str] = None) -> dict:
    try:
        temp_array_2d = thermal_data_to_array(thermal_data)
        room_temp = estimate_room_temperature(temp_array_2d)
        array_for_detection = temp_array_2d
        use_delta = False
        if sensor_id and sensor_id in thermal_background_by_sensor:
            background = thermal_background_by_sensor[sensor_id]
            if background.shape == temp_array_2d.shape:
                delta = np.maximum(0.0, temp_array_2d - background)
                array_for_detection = delta
                use_delta = True
        human_mask = detect_human_heat(
            array_for_detection,
            room_temp,
            use_delta=use_delta,
            absolute_temp_array=temp_array_2d if use_delta else None,
        )
        people_clusters = find_people_clusters(human_mask, temp_array_2d)
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
        print(f"Azure SQL connection failed ({type(e).__name__}); SQL saving disabled.")
        _sql_connection = False
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
        timestamp = datetime.now()
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
        print(f"Error saving thermal data: {e}")


def save_occupancy_data(occupancy_result: dict) -> None:
    if not SAVE_LOCAL_DATA and not SAVE_TO_BLOB:
        return
    try:
        timestamp = datetime.now()
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
        print(f"Error saving occupancy data: {e}")


def _iter_thermal_files() -> List[Path]:
    """Return all locally stored thermal frame files (legacy json + compressed json.gz)."""
    if not DATA_DIR.exists():
        return []
    files = [p for p in DATA_DIR.glob("thermal_*.json") if p.is_file()]
    files.extend([p for p in DATA_DIR.glob("thermal_*.json.gz") if p.is_file()])
    # Newest first (filenames include timestamp)
    files.sort(key=lambda p: p.name, reverse=True)
    return files


def _safe_int(value: Optional[int], default: int, min_value: int, max_value: int) -> int:
    try:
        v = int(value) if value is not None else int(default)
    except Exception:
        v = int(default)
    return max(min_value, min(max_value, v))


@app.get("/api/sensors")
def list_sensors() -> dict:
    """List known sensor IDs (from memory + stored files)."""
    sensors = set(latest_thermal_by_sensor.keys()) | set(latest_occupancy_by_sensor.keys())
    # Add sensors found on disk
    for p in _iter_thermal_files():
        # Filename: thermal_<safe_id>_<ts>_<suffix>.json(.gz) ; safe_id may differ from original.
        # Prefer payload sensor_id for correctness.
        try:
            payload = _read_json_payload(p)
            sid = payload.get("sensor_id")
            if sid:
                sensors.add(str(sid))
        except Exception:
            continue
    out = sorted(sensors)
    return {"count": len(out), "sensors": out}


@app.get("/api/thermal/history")
def get_thermal_history(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
    date: Optional[str] = Query(default=None, description="YYYYMMDD (optional)"),
    limit: int = Query(default=100, description="Max frames to return (1..500)"),
    offset: int = Query(default=0, description="Number of matching frames to skip"),
    include_data: bool = Query(default=False, description="If true, include full frame payload; else metadata only"),
) -> dict:
    """
    Return locally stored thermal frames (all sensors by default).
    Uses the saved thermal frame files under THERMAL_DATA_DIR, including legacy
    .json and compressed .json.gz files.
    """
    limit_i = _safe_int(limit, 100, 1, 500)
    offset_i = _safe_int(offset, 0, 0, 1_000_000_000)
    # Fetch one extra row so has_more is false when the page is full but no next page exists.
    fetch_limit = limit_i + 1

    matches: List[dict] = []
    seen = 0
    for p in _iter_thermal_files():
        try:
            payload = _read_json_payload(p)
        except Exception:
            continue

        sid = payload.get("sensor_id")
        ts = payload.get("timestamp")
        fmt = payload.get("format")

        if sensor_id is not None and str(sid) != str(sensor_id):
            continue
        if date:
            # timestamp is ISO; YYYY-MM-DD... compare by prefix after stripping dashes
            try:
                ymd = str(ts)[:10].replace("-", "")
            except Exception:
                ymd = ""
            if ymd != date:
                continue

        if seen < offset_i:
            seen += 1
            continue

        entry: dict = {
            "file": p.name,
            "timestamp": ts,
            "sensor_id": sid,
            "format": fmt,
        }
        if include_data:
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
    occupancy_result = estimate_occupancy(data, sensor_id=sensor_id)
    occupancy_result["sensor_id"] = sensor_id
    try:
        temp_array_2d = thermal_data_to_array(data)
        apply_occupancy_signal_processing(sensor_id, occupancy_result, temp_array_2d)
        _maybe_update_thermal_background(
            sensor_id, temp_array_2d, int(occupancy_result.get("occupancy_effective_raw", 0))
        )
        if _ml_engine is not None and _ml_engine.available:
            ml_result = _ml_engine.predict(
                temp_array_2d,
                background=thermal_background_by_sensor.get(sensor_id),
            )
            if ml_result:
                occupancy_result["ml"] = ml_result
    except Exception:
        pass
    latest_occupancy = occupancy_result
    now_iso = datetime.now().isoformat()
    last_update_time = now_iso
    # Per-sensor latest state
    latest_thermal_by_sensor[sensor_id] = dict(latest_thermal_data) if latest_thermal_data else {}
    latest_occupancy_by_sensor[sensor_id] = dict(occupancy_result)
    last_update_time_by_sensor[sensor_id] = now_iso
    save_thermal_data(compact_data, sensor_id)
    save_occupancy_data(occupancy_result)
    save_occupancy_data_sql(occupancy_result, timestamp_iso=now_iso)
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
    return out


@app.get("/api/thermal/current/all")
def get_all_thermal_data() -> dict:
    """Return latest thermal data for all sensors."""
    result = {}
    for sensor_id, data in latest_thermal_by_sensor.items():
        out = dict()
        out["building"] = data.get("building", "Other")
        out["last_update"] = last_update_time_by_sensor.get(sensor_id)
        occ = latest_occupancy_by_sensor.get(sensor_id)
        if occ:
            out["occupancy"] = occ.get("occupancy")
            out["room_temperature"] = occ.get("room_temperature")
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
    return {
        "date": date_str,
        "sensor_id": sensor_id,
        "total_readings": len(values),
        "min_occupancy": min(values),
        "max_occupancy": max(values),
        "avg_occupancy": round(sum(values) / len(values), 2),
        "current_occupancy": values[-1],
        "occupancy_distribution": dict(Counter(values)),
    }


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
    """Predict occupancy for next horizon_hours using same hour-of-day average over last 7 days."""
    now = datetime.now()
    # By hour-of-day (0..23), list of occupancy values from last 7 days
    by_hour: Dict[int, List[int]] = {h: [] for h in range(24)}
    for day_offset in range(1, 8):
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
                by_hour[dt.hour].append(entry["occupancy"])
    # Average per hour-of-day (no data -> 0)
    avg_by_hour: Dict[int, float] = {}
    for h in range(24):
        vals = by_hour[h]
        avg_by_hour[h] = round(sum(vals) / len(vals), 2) if vals else 0.0
    # Build prediction for next horizon_hours (starting at current hour)
    out = []
    for i in range(horizon_hours):
        t = now + timedelta(hours=i)
        h = t.hour
        bucket_start = t.replace(minute=0, second=0, microsecond=0)
        bucket_end = bucket_start + timedelta(hours=1)
        out.append({
            "bucket_start": bucket_start.isoformat(),
            "bucket_end": bucket_end.isoformat(),
            "expected_occupancy": avg_by_hour[h],
        })
    return out


@app.get("/api/occupancy/predict")
def get_occupancy_predict(
    sensor_id: Optional[str] = Query(default=None, description="Filter by sensor_id"),
    horizon_hours: int = Query(default=24, ge=1, le=48, description="Number of hours to predict (1..48)"),
) -> dict:
    """Predicted occupancy for next hours based on same hour-of-day average over last 7 days (heuristic)."""
    predictions = _predict_occupancy_heuristic(sensor_id, horizon_hours)
    return {
        "sensor_id": sensor_id,
        "horizon_hours": horizon_hours,
        "count": len(predictions),
        "data": predictions,
    }


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
    }


@app.get("/api/ml/labels")
def get_ml_labels_api() -> dict:
    """Return all stored frame labels."""
    with _ml_training_lock:
        labels_snap = list(_ml_labels.values())
    return {"labels": labels_snap, "count": len(labels_snap)}


@app.post("/api/ml/label")
def post_ml_label(req: _MLLabelRequest) -> dict:
    """Save or update a ground-truth label for a thermal frame file."""
    safe_file = Path(req.file).name
    if not safe_file:
        raise HTTPException(status_code=400, detail="Invalid filename")
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
    local_path = DATA_DIR / safe_file
    if not local_path.exists():
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
    if _ml_training_status.get("state") == "running":
        return {"status": "already_running", "training": _ml_training_status}
    if len(_ml_labels) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 labelled frames to train; have {len(_ml_labels)}.",
        )
    t = threading.Thread(target=_run_training_thread, daemon=True)
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
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415
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
    features, occ_targets, fever_targets = [], [], []

    for lbl in labels_snap:
        try:
            path = DATA_DIR / lbl["file"]
            if not path.exists():
                log(f"  Skipping missing file: {lbl['file']}")
                continue
            payload = _read_json_payload(path)
            compact = payload.get("data") or payload
            arr = thermal_data_to_array(compact)
            bg = thermal_background_by_sensor.get(payload.get("sensor_id"))
            features.append(_ml_extract(arr, bg))
            occ_targets.append(int(lbl["occupancy"]))
            fever_targets.append(1 if lbl.get("fever") else 0)
        except Exception as exc:
            log(f"  Warning: {lbl['file']}: {exc}")

    n = len(features)
    if n < 10:
        with _ml_training_lock:
            _ml_training_status = {
                "state": "error",
                "message": f"Only {n} usable feature vectors (need 10+). Check that labelled files still exist.",
                "ts": datetime.now().isoformat(),
                "log": log_lines,
            }
        return

    X = np.array(features, dtype=np.float32)
    y_occ = np.array(occ_targets)
    y_fever = np.array(fever_targets)
    out_dir = Path(os.environ.get("ML_MODEL_DIR", "ml_models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Training occupancy model on {n} samples (classes: {sorted(set(occ_targets))})…")
    occ_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    try:
        cv = min(5, max(2, n // 5))
        scores = cross_val_score(occ_clf, X, y_occ, cv=cv, scoring="accuracy")
        log(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except Exception as exc:
        log(f"  CV skipped: {exc}")
    occ_clf.fit(X, y_occ)
    _export_onnx_model(occ_clf, X.shape[1], out_dir / "occupancy_model.onnx", "occupancy_model", log)

    fever_pos = int(y_fever.sum())
    if fever_pos >= 5:
        log(f"Training fever model ({fever_pos}/{n} positive samples)…")
        fever_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        try:
            cv = min(5, max(2, n // 5))
            scores = cross_val_score(fever_clf, X, y_fever, cv=cv, scoring="f1")
            log(f"  CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as exc:
            log(f"  CV skipped: {exc}")
        fever_clf.fit(X, y_fever)
        _export_onnx_model(fever_clf, X.shape[1], out_dir / "fever_model.onnx", "fever_model", log)
    else:
        log(f"Skipping fever model — need ≥5 positive samples, have {fever_pos}")

    log("Reloading ML engine with new models…")
    if _ml_engine:
        _ml_engine.load(blob_container=_get_blob_container())

    with _ml_training_lock:
        _ml_training_status = {
            "state": "done",
            "message": f"Trained on {n} samples.",
            "ts": datetime.now().isoformat(),
            "log": log_lines,
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
  .canvas-wrap { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  canvas { image-rendering: pixelated; border: 1px solid #2d3748; border-radius: 4px; }
  #legend { display: flex; flex-direction: column; gap: 4px; }
  .legend-bar { width: 20px; height: 160px; border-radius: 3px; background: linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff); }
  .legend-label { font-size: 0.7rem; color: #94a3b8; }
  #status { margin-top: 14px; font-size: 0.8rem; color: #64748b; }
  .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
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
  const feverBadge = data.any_fever
    ? badge('badge-fever', '🌡 Fever detected')
    : data.any_elevated
      ? badge('badge-elevated', '⚠ Elevated temp')
      : badge('badge-ok', 'No fever');

  statsEl.innerHTML =
    badge('badge-occ',  `👤 ${occ} ${occ === 1 ? 'person' : 'people'}`) + ' ' +
    badge('badge-temp', `🌡 Room ${temp}`) + ' ' +
    feverBadge +
    (data.frame_valid === false ? ' ' + badge('badge-elevated', 'Frame invalid') : '');

  const ts = data.last_update ? new Date(data.last_update).toLocaleTimeString() : '—';
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

sensorSel.addEventListener('change', restart);
scaleSel.addEventListener('change', () => { /* re-render on next poll */ });
showClusters.addEventListener('change', restart);

loadSensors();
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
    ts.textContent = (frame.timestamp || '').substring(11,19);
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
  meta.textContent = frame.sensor_id + '  ·  ' + (frame.timestamp || '').replace('T',' ').substring(0,19);

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
