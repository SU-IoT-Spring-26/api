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
| GET | `/api/occupancy/trends` | Occupancy and room temp by time bucket (`?date=...`, `?sensor_id=...`, `?bucket=hour\|day`) |
| GET | `/api/occupancy/predict` | Predicted occupancy next 24–48h (`?sensor_id=...`, `?horizon_hours=24`) |

## Local run

Create the virtualenv once, then **always** `source .venv/bin/activate` before `python`, `uvicorn`, or the scripts under `scripts/` (same shell session as the command you run).

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with port from environment (e.g. Azure sets `PORT`):

```bash
source .venv/bin/activate
PORT=8000 uvicorn main:app --host 0.0.0.0
```

## Docker

Build and run locally:

```bash
docker build -t occupancy-api .
docker run -p 8000:8000 -e PORT=8000 occupancy-api
```

Optional: mount a host directory for persistent thermal/occupancy data:

```bash
docker run -p 8000:8000 -e PORT=8000 \
  -v "$(pwd)/thermal_data:/app/thermal_data" \
  occupancy-api
```

The image uses `python main.py` so Azure’s `PORT` is respected when deployed.

## Azure deployment (container)

You can run the Docker image on **Azure App Service (Linux container)**, **Azure Container Apps**, or **Azure Container Instances**.

### Option A: App Service with container

1. **Build and push** the image to a registry (e.g. Azure Container Registry):

   ```bash
   az acr build --registry <your-acr-name> --image occupancy-api:latest .
   ```

2. In **App Service** → **Deployment Center** (or **Configuration** → **General settings**):
   - Choose **Docker** → **Single Container**.
   - Set **Registry source** to Azure Container Registry (or Docker Hub) and select `occupancy-api:latest`.
   - Set **Target port** to **80** (App Service sets `PORT=80` by default; the app listens on whatever `PORT` is).

3. Set **Application settings** (env vars): `AZURE_STORAGE_CONNECTION_STRING`, `THERMAL_DATA_DIR` (e.g. `/home/thermal_data` if using a mounted storage), and any of the optional variables listed below. App Service sets `PORT` automatically.

4. **Persistent storage:** By default the container filesystem is ephemeral. For durable thermal/occupancy data either:
   - Configure **Azure Blob** via `AZURE_STORAGE_CONNECTION_STRING` (recommended), or
   - Use **App Service** → **Storage** to mount Azure Storage as a path (e.g. `/home/thermal_data`) and set `THERMAL_DATA_DIR=/home/thermal_data`.

### Option B: Container Apps

1. Build and push the image to ACR (or another registry).
2. Create a Container App with the image. Set **Target port** to the value your platform uses for `PORT` (e.g. **80** for App Service, or **8000** if you set `PORT=8000` in app settings).
3. Add the same environment variables; use Azure Blob for persistence or a volume mount if supported.

### Option C: Auto-deploy container from GitHub

A GitHub Actions workflow builds the Docker image, pushes it to Azure Container Registry (ACR), and deploys to App Service. Use the workflow in [.github/workflows/deploy-container.yml](.github/workflows/deploy-container.yml).

**One-time setup**

1. **Azure Container Registry**  
   Create an ACR (e.g. in the same resource group as the app):
   ```bash
   az acr create --resource-group occupancy-rg --name myregistry --sku Basic --admin-enabled true
   ```
   Copy the **Login server** (e.g. `myregistry.azurecr.io`). In the repo, edit `.github/workflows/deploy-container.yml` and set `ACR_LOGIN_SERVER` to that value (e.g. `myregistry.azurecr.io`).

2. **App Service as container**  
   In Azure Portal → App Service **occupancy-api** → **Deployment Center** (or **Configuration** → **General settings**):
   - **Publish:** Docker Container.
   - **Registry:** Azure Container Registry; select your ACR and choose any image/tag for now (e.g. `occupancy-api:latest`). The workflow will overwrite it on each deploy.
   - **Target port:** set to **80**. (App Service sets `PORT=80`; the app listens on that port.)
   - Save. Ensure **Container settings** → **Startup Command** is empty (the image runs `python main.py`) or set to `python main.py`.

3. **GitHub secrets**  
   In the repo: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
   - `REGISTRY_USERNAME` – ACR **Username** (Azure Portal → ACR → **Access keys**).
   - `REGISTRY_PASSWORD` – ACR **password** (from the same **Access keys** blade).
   - `AZURE_CREDENTIALS` – service principal JSON so the workflow can update the App Service container. **Creating the SP and role assignment requires “Owner” or “User Access administrator” on the subscription or resource group.** If you get `AuthorizationFailed` on role assignment, ask an admin to run the steps below and share the JSON (or the `appId`/`password`/`tenant` values) with you.
     **Option A – You have permission:** Create the SP (no deprecated flags):
     ```bash
     SUBSCRIPTION_ID=$(az account show --query id -o tsv)
     az ad sp create-for-rbac --name "github-occupancy-api" --role contributor \
       --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/occupancy-rg
     ```
     From the output, build the secret value as one line (replace placeholders with the command output):
     ```json
     {"clientId":"<appId>","clientSecret":"<password>","subscriptionId":"<SUBSCRIPTION_ID>","tenantId":"<tenant>"}
     ```
     **Option B – Admin runs it:** An Owner/User Access admin runs the same `az ad sp create-for-rbac` command (no `--sdk-auth`), then builds the JSON above from `appId`, `password`, `tenant`, and your subscription ID, and gives you that JSON to store as `AZURE_CREDENTIALS`.

4. **Optional:** Add application settings (env vars) in App Service → **Configuration** → **Application settings**, e.g. `AZURE_STORAGE_CONNECTION_STRING`, `THERMAL_DATA_DIR`, etc.

After this, every push to `main` (or a manual **Run workflow** from the Actions tab) will build the image, push to ACR, and deploy to the Web App.

### Environment variables (container / Azure)

Set in App Service **Configuration** → **Application settings**, or as `-e` when running Docker locally:

- `THERMAL_DATA_DIR` – directory for thermal and occupancy files (default: `thermal_data`; in the image: `/app/thermal_data`)
- `SAVE_THERMAL_DATA` – set to `false` to disable saving (default: `true`)
- `PORT` – port to bind (Azure sets this automatically; default 8000 in the image)
- `AZURE_STORAGE_CONNECTION_STRING` – if set, thermal and occupancy data are also written to Azure Blob Storage
- `AZURE_STORAGE_CONTAINER_NAME` – blob container name (default: `iotoccupancydata`)
- `BACKGROUND_ALPHA` – EMA weight for thermal background (default: `0.95`)
- `BACKGROUND_MIN_FRAMES_EMPTY` – consecutive empty frames before updating background (default: `3`)
- `BACKGROUND_MAX_MEAN_ABS_DELTA_C` – max mean absolute frame delta (°C) between consecutive **empty** frames to count toward background update; larger motion resets the empty streak (default: `2.5`; `0` disables the check)
- `ROOM_TEMP_THRESHOLD` – degrees (°C) above room estimate a pixel must be to count as human heat in delta mode (default: `0.5`)
- `FEVER_THRESHOLD_C` – temperature (°C) above which a cluster is flagged as fever (default: `37.5`)
- `FEVER_ELEVATED_THRESHOLD_C` – lower band: cluster max temp between this and fever threshold is counted as elevated (default: `37.0`)
- `FEVER_MIN_CONSECUTIVE_FRAMES` – `any_fever` in API output is true only after this many consecutive frames with raw fever (default: `2`)
- `OCCUPANCY_SMOOTH_WINDOW` – median smoothing length over effective raw occupancy (default: `5`)
- `OCCUPANCY_HYSTERESIS_DELTA` – suppress small oscillations: changes smaller than this from last smoothed value are held (default: `1`)
- `FRAME_ROOM_MEDIAN_MAX_JUMP_C` – if the full-frame median temperature jumps more than this (°C) vs the previous frame, the frame is treated invalid and raw occupancy falls back to the last good value; final `occupancy` still uses the smoother (default: `4.0`; `0` disables)

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

## Server-side features

### Background subtraction

When the room is empty for several consecutive frames, the server updates a per-sensor thermal background (EMA). Occupancy detection then uses temperature *above* background to reduce false positives from equipment/HVAC. Backgrounds are persisted under `THERMAL_DATA_DIR` as `background_<sensor_id>.npy`.

### Fever detection

Each detected person cluster gets a representative temperature (90th percentile of pixels in the cluster). Clusters with representative temp **between** `FEVER_ELEVATED_THRESHOLD_C` and `FEVER_THRESHOLD_C` are tagged `elevated_temp`; clusters at or above `FEVER_THRESHOLD_C` are raw fever candidates (`fever_count`, `any_fever_raw`). The public `any_fever` flag is gated: it becomes true only after `FEVER_MIN_CONSECUTIVE_FRAMES` consecutive frames with raw fever, to reduce single-frame noise. Responses and occupancy JSONL include `elevated_count`, `any_elevated`, `any_fever_raw`, and `fever_consecutive_frames` where applicable.

### Occupancy signal processing (temporal)

After clustering, the server applies **frame sanity**, **median smoothing**, and **hysteresis** before exposing `occupancy`:

| Field | Meaning |
|--------|--------|
| `occupancy_raw_instant` | Person count from clustering this frame |
| `occupancy_effective_raw` | Same as raw if the frame is valid; if the frame fails the median jump check, repeats the last good raw count |
| `occupancy` | Smoothed + hysteresis output (what clients should use for stable room counts) |
| `frame_valid` | False when the median temperature jump test rejected the frame |

These fields are stored in daily `occupancy_YYYYMMDD.jsonl` and returned from `POST /api/thermal` and `GET /api/thermal` (latest).

### Trends and prediction

- **`GET /api/occupancy/trends`** – Aggregates occupancy and room temperature by hour or day for a given date.
- **`GET /api/occupancy/predict`** – Heuristic prediction for the next 1–48 hours using same hour-of-day average over the last 7 days.

### Ground truth and calibration

With labeled data you can score stored logs or replay thermal archives:

- **`scripts/compare_occupancy_accuracy.py`** – CSV `timestamp,sensor_id,actual_count` vs `occupancy_*.jsonl`. Options `--field` and `--compare-fields` let you compare smoothed `occupancy`, `occupancy_effective_raw`, or `occupancy_raw_instant`.
- **`scripts/compare_fever_accuracy.py`** – CSV `timestamp,sensor_id,fever` (0/1) vs fever flags in JSONL (`any_fever`, `any_fever_raw`, or `fever_count_positive`).
- **`scripts/replay_thermal_occupancy.py`** – Re-run `thermal_*_compact.json` through the pipeline offline (no writes by default); optional CSV export.
- **`scripts/calibrate_occupancy_thresholds.py`** – Small grid over `ROOM_TEMP_THRESHOLD` / `MIN_CLUSTER_SIZE` using replay + the same alignment as the accuracy script.

Details and examples: [scripts/README.md](scripts/README.md).
