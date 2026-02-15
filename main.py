#!/usr/bin/env python3
"""
Occupancy API – FastAPI data reception and storage for thermal/occupancy data.
Designed for Azure App Service (resource group: occupancy-rg, app: occupancy-api).
Stores data locally and optionally to Azure Blob Storage when configured.
"""

import json
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.ndimage import label

# Optional Azure Blob Storage (only used if AZURE_STORAGE_CONNECTION_STRING is set)
# _blob_container_client: container client if connected, False if init failed, None if not tried
_blob_container_client: Any = None


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

app = FastAPI(
    title="Occupancy API",
    description="Receive and store thermal camera data; estimate and query occupancy.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path(os.environ.get("THERMAL_DATA_DIR", "thermal_data"))
SAVE_DATA = os.environ.get("SAVE_THERMAL_DATA", "true").lower() in ("1", "true", "yes")

# Occupancy detection parameters
MIN_HUMAN_TEMP = 30.0
MAX_HUMAN_TEMP = 45.0
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 200
ROOM_TEMP_THRESHOLD = 0.5
FEVER_THRESHOLD_C = float(os.environ.get("FEVER_THRESHOLD_C", "37.5"))

# Background subtraction (thermal): per-sensor background, updated when room empty
BACKGROUND_ALPHA = float(os.environ.get("BACKGROUND_ALPHA", "0.95"))  # EMA weight for existing background
BACKGROUND_MIN_FRAMES_EMPTY = int(os.environ.get("BACKGROUND_MIN_FRAMES_EMPTY", "3"))  # Consecutive empty frames to update
thermal_background_by_sensor: Dict[str, np.ndarray] = {}
empty_frame_count_by_sensor: Dict[str, int] = {}

# In-memory latest state
latest_thermal_data: Optional[dict] = None
last_update_time: Optional[str] = None
latest_occupancy: Optional[dict] = None
latest_thermal_by_sensor: Dict[str, dict] = {}
last_update_time_by_sensor: Dict[str, str] = {}
latest_occupancy_by_sensor: Dict[str, dict] = {}
_data_counter = 0

if SAVE_DATA:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    pixels = expanded_data["pixels"]
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
    temp_array: np.ndarray, room_temp: float, use_delta: bool = False
) -> np.ndarray:
    """Human heat mask. If use_delta True, temp_array is delta above background (same-threshold interpretation)."""
    if use_delta:
        human_mask = (temp_array >= ROOM_TEMP_THRESHOLD) & (temp_array <= MAX_HUMAN_TEMP)
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
            people_clusters.append({
                "id": i,
                "size": cluster_size,
                "center": (center_row, center_col),
                "representative_temp_c": round(representative_temp_c, 2),
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
        human_mask = detect_human_heat(array_for_detection, room_temp, use_delta=use_delta)
        people_clusters = find_people_clusters(human_mask, temp_array_2d)
        fever_count = sum(1 for c in people_clusters if c.get("fever_detected"))
        return {
            "occupancy": len(people_clusters),
            "room_temperature": room_temp,
            "people_clusters": people_clusters,
            "fever_count": fever_count,
            "any_fever": fever_count > 0,
        }
    except Exception as e:
        return {
            "occupancy": 0,
            "room_temperature": None,
            "people_clusters": [],
            "fever_count": 0,
            "any_fever": False,
            "error": str(e),
        }


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


def save_thermal_data(
    compact_data: dict, expanded_data: dict, sensor_id: Optional[str] = None
) -> None:
    global _data_counter
    if not SAVE_DATA:
        return
    sid = sensor_id or compact_data.get("sensor_id") or "unknown"
    safe_id = _sanitize_sensor_id_for_filename(sid)
    try:
        timestamp = datetime.now()
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        for fmt, data, suffix in [
            ("compact", compact_data, "compact"),
            ("expanded", expanded_data, "expanded"),
        ]:
            payload = {
                "timestamp": timestamp.isoformat(),
                "format": fmt,
                "sensor_id": sid,
                "data": data,
            }
            path = DATA_DIR / f"thermal_{safe_id}_{ts}_{suffix}.json"
            json_bytes = json.dumps(payload, indent=2).encode("utf-8")
            with open(path, "wb") as f:
                f.write(json_bytes)
            # Azure Blob: same content under thermal/ prefix
            _upload_blob(f"thermal/{path.name}", json_bytes)
        _data_counter += 1
    except Exception as e:
        print(f"Error saving thermal data: {e}")


def save_occupancy_data(occupancy_result: dict) -> None:
    if not SAVE_DATA:
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
            "room_temperature": (
                float(occupancy_result["room_temperature"])
                if occupancy_result.get("room_temperature") is not None
                else None
            ),
            "people_clusters": convert_numpy_types(occupancy_result.get("people_clusters", [])),
            "fever_count": int(occupancy_result.get("fever_count", 0)),
            "any_fever": bool(occupancy_result.get("any_fever", False)),
        }
        line = json.dumps(entry) + "\n"
        with open(path, "a") as f:
            f.write(line)
        # Azure Blob: append to daily append blob
        _append_to_blob(f"occupancy/occupancy_{date_str}.jsonl", line)
    except Exception as e:
        print(f"Error saving occupancy data: {e}")


def _iter_thermal_files() -> List[Path]:
    """Return all locally stored thermal frame files (compact + expanded)."""
    if not DATA_DIR.exists():
        return []
    files = [p for p in DATA_DIR.glob("thermal_*.json") if p.is_file()]
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
        # Filename: thermal_<safe_id>_<ts>_<suffix>.json ; safe_id may differ from original.
        # Prefer payload sensor_id for correctness.
        try:
            payload = json.loads(p.read_text())
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
    limit: int = Query(default=10000, description="Max frames to return (1..10000)"),
    offset: int = Query(default=0, description="Number of matching frames to skip"),
    include_data: bool = Query(default=False, description="If true, include full frame payload; else metadata only"),
) -> dict:
    """
    Return locally stored thermal frames (all sensors by default).
    Uses the saved JSON files under THERMAL_DATA_DIR.
    """
    limit_i = _safe_int(limit, 100, 1, 1000)
    offset_i = _safe_int(offset, 0, 0, 1_000_000_000)

    matches: List[dict] = []
    seen = 0
    for p in _iter_thermal_files():
        try:
            payload = json.loads(p.read_text())
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
            entry["data"] = payload.get("data")
        matches.append(entry)
        if len(matches) >= limit_i:
            break

    return {
        "sensor_id": sensor_id,
        "date": date,
        "limit": limit_i,
        "offset": offset_i,
        "count": len(matches),
        "data": matches,
    }


@app.get("/api/test")
def test() -> dict:
    """Health check for Azure and clients."""
    return {"status": "server is running", "time": datetime.now().isoformat()}


def _maybe_update_thermal_background(sensor_id: str, temp_array_2d: np.ndarray, occupancy: int) -> None:
    """Update per-sensor thermal background when room is empty for BACKGROUND_MIN_FRAMES_EMPTY consecutive frames."""
    global thermal_background_by_sensor, empty_frame_count_by_sensor
    if sensor_id not in empty_frame_count_by_sensor:
        empty_frame_count_by_sensor[sensor_id] = 0
    if occupancy > 0:
        empty_frame_count_by_sensor[sensor_id] = 0
        return
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
        compact_data = collapse_to_compact(latest_thermal_data)
    occupancy_result = estimate_occupancy(data, sensor_id=sensor_id)
    occupancy_result["sensor_id"] = sensor_id
    try:
        temp_array_2d = thermal_data_to_array(data)
        _maybe_update_thermal_background(sensor_id, temp_array_2d, occupancy_result["occupancy"])
    except Exception:
        pass
    latest_occupancy = occupancy_result
    now_iso = datetime.now().isoformat()
    last_update_time = now_iso
    # Per-sensor latest state
    latest_thermal_by_sensor[sensor_id] = dict(latest_thermal_data) if latest_thermal_data else {}
    latest_occupancy_by_sensor[sensor_id] = dict(occupancy_result)
    last_update_time_by_sensor[sensor_id] = now_iso
    save_thermal_data(compact_data, latest_thermal_data, sensor_id)
    save_occupancy_data(occupancy_result)
    pixel_count = len(latest_thermal_data.get("pixels", []))
    return {
        "status": "success",
        "received": pixel_count,
        "occupancy": occupancy_result["occupancy"],
        "fever_count": occupancy_result.get("fever_count", 0),
        "any_fever": occupancy_result.get("any_fever", False),
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
            out["room_temperature"] = occ.get("room_temperature")
            out["fever_count"] = occ.get("fever_count", 0)
            out["any_fever"] = occ.get("any_fever", False)
        return out

    if latest_thermal_data is None:
        raise HTTPException(status_code=404, detail="No data available")
    out = dict(latest_thermal_data)
    out["last_update"] = last_update_time
    if latest_occupancy:
        out["occupancy"] = latest_occupancy["occupancy"]
        out["room_temperature"] = latest_occupancy.get("room_temperature")
        out["fever_count"] = latest_occupancy.get("fever_count", 0)
        out["any_fever"] = latest_occupancy.get("any_fever", False)
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


# TODO: build actual prediction model
@app.get("/api/thermal/predicted/poll")
def get_predicted_thermal_data_poll(
    sensor_id: Optional[str] = Query(default=None, description="If set, return latest for this sensor_id"),
) -> dict:
    """Return latest thermal data (expanded format with occupancy)."""
    return get_thermal_data(sensor_id)


@app.get("/api/thermal/predicted/all")
def get_predicted_thermal_data() -> dict:
    """Return latest thermal data for all sensors."""
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
            if line.strip():
                entry = json.loads(line)
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
            if line.strip():
                entry = json.loads(line)
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
