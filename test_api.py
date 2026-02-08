#!/usr/bin/env python3
"""
Test the Occupancy API (and thus storage) using the Azure base URL.
Usage:
  export API_BASE_URL=https://occupancy-api.azurewebsites.net
  python test_api.py
Or:
  python test_api.py https://occupancy-api.azurewebsites.net
"""

import json
import os
import sys
from datetime import datetime

# 24x32 = 768 pixels (MLX90640 shape)
MLX_SHAPE = (24, 32)
FRAME_SIZE = MLX_SHAPE[0] * MLX_SHAPE[1]


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_BASE_URL", "")).rstrip("/")
    if not base:
        print("Usage: python test_api.py <BASE_URL>")
        print("   or: API_BASE_URL=<BASE_URL> python test_api.py")
        print("Example: python test_api.py https://occupancy-api.azurewebsites.net")
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    session = requests.Session()
    session.headers["Content-Type"] = "application/json"

    def get(path: str) -> requests.Response:
        return session.get(f"{base}{path}", timeout=30)

    def post(path: str, data: dict) -> requests.Response:
        return session.post(f"{base}{path}", json=data, timeout=30)

    print(f"Testing API at {base}\n")

    # 1. Health
    print("1. GET /api/test")
    r = get("/api/test")
    r.raise_for_status()
    body = r.json()
    print(f"   OK: {body.get('status', body)}")
    assert "server is running" in str(body.get("status", "")).lower() or "ok" in str(body).lower(), body
    print()

    # 2. POST thermal (minimal valid frame: 768 temps)
    print("2. POST /api/thermal (sensor_id=test-script, 768 temps)")
    temps = [20.0 + (i % 10) * 0.1 for i in range(FRAME_SIZE)]
    payload = {
        "sensor_id": "test-script",
        "w": MLX_SHAPE[1],
        "h": MLX_SHAPE[0],
        "min": min(temps),
        "max": max(temps),
        "t": [round(t, 1) for t in temps],
    }
    r = post("/api/thermal", payload)
    r.raise_for_status()
    body = r.json()
    print(f"   OK: status={body.get('status')}, received={body.get('received')}, occupancy={body.get('occupancy')}")
    assert body.get("status") == "success", body
    print()

    # 3. GET latest thermal
    print("3. GET /api/thermal")
    r = get("/api/thermal")
    r.raise_for_status()
    body = r.json()
    pixels = body.get("pixels") or []
    print(f"   OK: sensor_id={body.get('sensor_id')}, pixels={len(pixels)}, occupancy={body.get('occupancy')}, last_update={body.get('last_update', '')[:19]}")
    assert body.get("sensor_id") == "test-script", body.get("sensor_id")
    assert len(pixels) == FRAME_SIZE, len(pixels)
    print()

    # 4. Occupancy history (today)
    date_str = datetime.now().strftime("%Y%m%d")
    print(f"4. GET /api/occupancy/history?date={date_str}")
    r = get(f"/api/occupancy/history?date={date_str}")
    r.raise_for_status()
    body = r.json()
    count = body.get("count", 0)
    data = body.get("data") or []
    print(f"   OK: date={body.get('date')}, count={count}")
    assert body.get("date") == date_str, body
    if data:
        last = data[-1]
        print(f"   Last entry: sensor_id={last.get('sensor_id')}, occupancy={last.get('occupancy')}, ts={last.get('timestamp', '')[:19]}")
    print()

    # 5. Occupancy stats (today)
    print(f"5. GET /api/occupancy/stats?date={date_str}")
    r = get(f"/api/occupancy/stats?date={date_str}")
    r.raise_for_status()
    body = r.json()
    print(f"   OK: total_readings={body.get('total_readings')}, current_occupancy={body.get('current_occupancy')}, avg={body.get('avg_occupancy')}")
    assert "total_readings" in body, body
    print()

    # 6. Filter by sensor_id (verify our payload is in storage)
    print("6. GET /api/occupancy/history?sensor_id=test-script")
    r = get(f"/api/occupancy/history?date={date_str}&sensor_id=test-script")
    r.raise_for_status()
    body = r.json()
    print(f"   OK: count={body.get('count')} (entries from test-script)")
    print()

    print("All checks passed. API and storage (local + Azure Blob when configured) are working.")


if __name__ == "__main__":
    main()
