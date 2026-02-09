#!/usr/bin/env python3
"""
Occupancy API – FastAPI data reception and storage for thermal/occupancy data.
Designed for Azure App Service (resource group: occupancy-rg, app: occupancy-api).
Stores data locally and optionally to Azure Blob Storage when configured.
"""

import json
import os
from collections import Counter
from datetime import datetime
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


def detect_human_heat(temp_array: np.ndarray, room_temp: float) -> np.ndarray:
    human_mask = (temp_array >= MIN_HUMAN_TEMP) & (temp_array <= MAX_HUMAN_TEMP)
    relative_mask = (temp_array - room_temp) >= ROOM_TEMP_THRESHOLD
    return (human_mask & relative_mask).astype(int)


def find_people_clusters(human_mask: np.ndarray) -> List[Dict]:
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(human_mask, structure=structure)
    people_clusters = []
    for i in range(1, num_features + 1):
        cluster_size = int(np.sum(labeled_array == i))
        if MIN_CLUSTER_SIZE <= cluster_size <= MAX_CLUSTER_SIZE:
            cluster_pixels = np.where(labeled_array == i)
            center_row = int(np.mean(cluster_pixels[0]))
            center_col = int(np.mean(cluster_pixels[1]))
            people_clusters.append({"id": i, "size": cluster_size, "center": (center_row, center_col)})
    return people_clusters


def estimate_occupancy(thermal_data: dict) -> dict:
    try:
        temp_array_2d = thermal_data_to_array(thermal_data)
        room_temp = estimate_room_temperature(temp_array_2d)
        human_mask = detect_human_heat(temp_array_2d, room_temp)
        people_clusters = find_people_clusters(human_mask)
        return {
            "occupancy": len(people_clusters),
            "room_temperature": room_temp,
            "people_clusters": people_clusters,
        }
    except Exception as e:
        return {
            "occupancy": 0,
            "room_temperature": None,
            "people_clusters": [],
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
    limit: int = Query(default=1000, description="Max frames to return (1..1000)"),
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


@app.post("/api/thermal")
def receive_thermal_data(data: dict) -> dict:
    """Receive thermal data from ESP32 (compact or expanded format)."""
    global latest_thermal_data, last_update_time, latest_occupancy
    if not data:
        raise HTTPException(status_code=400, detail="No data received")
    compact_data = dict(data)
    sensor_id = data.get("sensor_id") or "unknown"
    if "t" in data:
        try:
            expanded_data = expand_thermal_data(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data expansion failed: {e}") from e
        expanded_data["sensor_id"] = sensor_id
        latest_thermal_data = expanded_data
    else:
        latest_thermal_data = data if data.get("sensor_id") else {**data, "sensor_id": sensor_id}
    occupancy_result = estimate_occupancy(data)
    occupancy_result["sensor_id"] = sensor_id
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
    }


@app.get("/api/thermal")
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
        return out

    if latest_thermal_data is None:
        raise HTTPException(status_code=404, detail="No data available")
    out = dict(latest_thermal_data)
    out["last_update"] = last_update_time
    if latest_occupancy:
        out["occupancy"] = latest_occupancy["occupancy"]
        out["room_temperature"] = latest_occupancy.get("room_temperature")
    return out


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
