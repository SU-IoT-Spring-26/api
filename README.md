# Occupancy API

occupancy-api-b8hcb8hyh7f7aph3.canadacentral-01.azurewebsites.net

FastAPI service for receiving and storing thermal camera data from ESP32 devices, with occupancy estimation and history.

**Azure:** Resource group `occupancy-rg`, App Service `occupancy-api`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/test` | Health check |
| POST | `/api/thermal` | Submit thermal data (compact or expanded JSON) |
| GET | `/api/thermal` | Latest thermal data + occupancy (optionally `?sensor_id=...` for specific sensor) |
| GET | `/api/sensors` | List all known sensor IDs |
| GET | `/api/thermal/history` | Browse stored thermal frames (`?sensor_id=...`, `?date=YYYYMMDD`, `?limit=...`, `?offset=...`, `?include_data=true`) |
| GET | `/api/occupancy/history` | Occupancy log for a date (`?date=YYYYMMDD`, `?sensor_id=...`) |
| GET | `/api/occupancy/stats` | Occupancy stats for a date (`?date=YYYYMMDD`, `?sensor_id=...`) |

## Local run

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with port from environment (e.g. Azure sets `PORT`):

```bash
PORT=8000 uvicorn main:app --host 0.0.0.0
```

## Azure App Service

- **Resource group:** `occupancy-rg`
- **App Service:** `occupancy-api`

**Startup command** (App Service → Configuration → General settings):

```bash
python main.py
```

This uses the `PORT` environment variable set by Azure (default 8000 when run locally).

Alternative (fixed port):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Environment variables (optional):**

- `THERMAL_DATA_DIR` – directory for thermal and occupancy files (default: `thermal_data`)
- `SAVE_THERMAL_DATA` – set to `false` to disable saving (default: `true`)
- `PORT` – port to bind (Azure sets this automatically)
- `AZURE_STORAGE_CONNECTION_STRING` – if set, thermal and occupancy data are also written to Azure Blob Storage (in addition to local disk)
- `AZURE_STORAGE_CONTAINER_NAME` – blob container name (default: `iotoccupancydata`)

## Test

**Script (recommended)** – exercises all endpoints and verifies storage:

```bash
export API_BASE_URL=https://occupancy-api.azurewebsites.net
python test_api.py
```

Or pass the base URL as an argument:

```bash
python test_api.py https://occupancy-api.azurewebsites.net
```

**Manual curl:**

```bash
curl https://<occupancy-api>.azurewebsites.net/api/test
curl -X POST https://<occupancy-api>.azurewebsites.net/api/thermal \
  -H "Content-Type: application/json" \
  -d '{"sensor_id":"test","w":32,"h":24,"min":20,"max":25,"t":[20.0]}'
```

## Data format (POST /api/thermal)

Compact format from ESP32:

- `sensor_id` (optional), `w`, `h`, `min`, `max`, `t` (list of temperatures, row-major)

**Local storage:** thermal frames under `THERMAL_DATA_DIR` as `thermal_<sensor_id>_<timestamp>_compact.json` and `_expanded.json`; occupancy as `occupancy_YYYYMMDD.jsonl` with one JSON object per line.

**Azure Blob Storage** (when `AZURE_STORAGE_CONNECTION_STRING` is set): same data is also written to the configured container. Thermal files go under the `thermal/` prefix (e.g. `thermal/thermal_sensor1_20250107_120000_compact.json`). Occupancy is appended to `occupancy/occupancy_YYYYMMDD.jsonl` (append blobs). Local storage is always used when `SAVE_THERMAL_DATA` is true; Blob is an additional copy.

## Multi-sensor support

The API tracks data from **all sensors** and makes it available via the endpoints:

### Latest data per sensor

- **`GET /api/thermal?sensor_id=<id>`** – Returns the latest thermal frame **for that specific sensor** (with occupancy).
- **`GET /api/thermal`** (no `sensor_id`) – Returns the latest frame from whichever sensor posted most recently (backwards compatible).

### List sensors

- **`GET /api/sensors`** – Returns a list of all known sensor IDs (from in-memory state and stored files).

### Browse stored thermal history

- **`GET /api/thermal/history`** – Returns stored thermal frames from disk (all sensors by default).

**Query parameters:**
- `sensor_id` (optional) – Filter to frames from a specific sensor
- `date=YYYYMMDD` (optional) – Filter to frames from a specific date
- `limit` (default: 100, max: 500) – Maximum number of frames to return
- `offset` (default: 0) – Number of matching frames to skip (for paging)
- `include_data` (default: false) – If `true`, include full frame payload; if `false`, return metadata only (timestamp, sensor_id, format, filename)

**Examples:**
- All sensors, newest 100 (metadata only): `GET /api/thermal/history`
- One sensor with full frames: `GET /api/thermal/history?sensor_id=living-room&include_data=true`
- All sensors for a specific day: `GET /api/thermal/history?date=20260207&limit=500`
- Page through results: `GET /api/thermal/history?limit=50&offset=0`, then `?limit=50&offset=50`, etc.

**Note:** The history endpoint reads from locally stored files under `THERMAL_DATA_DIR`. Each frame includes `timestamp` (server receive time) and `sensor_id` alongside the thermal data.
