"""
Shared fixtures for all test modules.

Environment variables must be set before main is imported, so they live at
module level here — pytest loads conftest.py before any test module.
"""
import os

os.environ["SAVE_THERMAL_DATA"] = "false"
os.environ["SAVE_TO_SQL"] = "false"

import pytest
from fastapi.testclient import TestClient

import main


# ---------------------------------------------------------------------------
# In-memory state reset — runs before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_state():
    """Wipe all in-memory globals so tests are fully independent."""
    main.latest_thermal_data = None
    main.last_update_time = None
    main.latest_occupancy = None
    main.latest_thermal_by_sensor.clear()
    main.last_update_time_by_sensor.clear()
    main.latest_occupancy_by_sensor.clear()
    main.thermal_background_by_sensor.clear()
    main.empty_frame_count_by_sensor.clear()
    main.last_empty_frame_thermal_by_sensor.clear()
    main.occupancy_raw_history_by_sensor.clear()
    main.last_frame_median_by_sensor.clear()
    main.last_raw_occupancy_by_sensor.clear()
    main.last_smoothed_occupancy_by_sensor.clear()
    main.fever_consecutive_by_sensor.clear()
    main._data_counter = 0
    main._ground_truth.clear()
    yield


# ---------------------------------------------------------------------------
# HTTP client — redirects DATA_DIR to an isolated tmp dir so the startup
# handler never loads real on-disk data and file-based endpoints use tmp files
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    original = main.DATA_DIR
    main.DATA_DIR = tmp_path
    with TestClient(main.app, raise_server_exceptions=True) as c:
        yield c
    main.DATA_DIR = original


# ---------------------------------------------------------------------------
# Re-usable compact frame payloads
# ---------------------------------------------------------------------------

def _make_compact(w=4, h=2, base_temp=21.0, sensor_id=None):
    temps = [round(base_temp + i * 0.1, 1) for i in range(w * h)]
    d = {"w": w, "h": h, "min": min(temps), "max": max(temps), "t": temps}
    if sensor_id:
        d["sensor_id"] = sensor_id
    return d


def _make_compact_with_person(sensor_id="test-sensor"):
    """10×10 frame (room temp 21°C) with a 3×3 hot spot at 36°C."""
    W, H = 10, 10
    arr = [21.0] * (W * H)
    for r in range(3, 6):
        for c in range(3, 6):
            arr[r * W + c] = 36.0
    return {
        "w": W, "h": H,
        "min": 21.0, "max": 36.0,
        "t": arr,
        "sensor_id": sensor_id,
    }


def _make_compact_with_fever(sensor_id="test-sensor"):
    """10×10 frame with a 3×3 hot spot at 38°C (above fever threshold 37.5)."""
    W, H = 10, 10
    arr = [21.0] * (W * H)
    for r in range(3, 6):
        for c in range(3, 6):
            arr[r * W + c] = 38.0
    return {
        "w": W, "h": H,
        "min": 21.0, "max": 38.0,
        "t": arr,
        "sensor_id": sensor_id,
    }


@pytest.fixture
def compact_frame():
    return _make_compact(sensor_id="test-sensor")


@pytest.fixture
def compact_frame_with_person():
    return _make_compact_with_person()


@pytest.fixture
def compact_frame_with_fever():
    return _make_compact_with_fever()
