# Occupancy API

occupancy-api-b8hcb8hyh7f7aph3.canadacentral-01.azurewebsites.net

FastAPI service for receiving and storing thermal camera data from ESP32 devices, with occupancy estimation and history.

**Azure:** Resource group `occupancy-rg`, App Service `occupancy-api`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/test` | Health check |
| POST | `/api/thermal` | Submit thermal data (compact or expanded JSON) |
| GET | `/api/thermal` | Latest thermal data + occupancy |
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
