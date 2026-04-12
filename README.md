# Occupancy API

**Live deployment:** https://occupancy-api-container.yellowbush-1452fab1.canadacentral.azurecontainerapps.io

FastAPI service that receives thermal camera frames from ESP32 devices (MLX90640 32×24 sensor), estimates room occupancy and detects fever using heuristic algorithms and optional ML models, and stores data on Azure Blob Storage.

**Azure:** Resource group `occupancy-rg`, Container App `occupancy-api-container`, Container Registry `occupancyregistry.azurecr.io`.

---

## Web interfaces

| URL | Description |
|-----|-------------|
| `/` | Live thermal camera dashboard — real-time heatmap, occupancy badges, cluster overlays, per-sensor view |
| `/ml` | ML Studio — label training data, train occupancy/fever models, run per-frame inference |
| `/docs` | Auto-generated Swagger UI for all API endpoints |

---

## API endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/test` | Health check — returns `{"status": "server is running", "time": "..."}` |

### Thermal data

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/thermal` | Ingest a thermal frame from an ESP32. Accepts compact or expanded format. Returns occupancy result immediately. |
| `GET` | `/api/thermal/current/poll` | Latest frame + occupancy for one sensor (`?sensor_id=`) or the most recently updated sensor |
| `GET` | `/api/thermal/current/all` | Latest frame summary (building, occupancy, room temp) for **all** sensors |
| `GET` | `/api/thermal/history` | Paginated list of stored frames. See [thermal history parameters](#thermal-history-parameters). |

Legacy aliases (identical behaviour, kept for backwards compatibility):

| Method | Path | Alias of |
|--------|------|----------|
| `GET` | `/api/thermal` | `/api/thermal/current/poll` |
| `GET` | `/api/thermal/predicted/poll` | `/api/thermal/current/poll` |
| `GET` | `/api/thermal/predicted/all` | `/api/thermal/current/all` |

### Sensors

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sensors` | List all known sensor IDs (from in-memory state and stored files) |

### Occupancy

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/occupancy/history` | All occupancy log entries for a date (`?date=YYYYMMDD`, `?sensor_id=`) |
| `GET` | `/api/occupancy/stats` | Aggregate statistics for a date: min, max, avg, current, distribution (`?date=`, `?sensor_id=`) |
| `GET` | `/api/occupancy/trends` | Occupancy and room temperature bucketed by hour or day (`?date=`, `?sensor_id=`, `?bucket=hour\|day`) |
| `GET` | `/api/occupancy/predict` | Heuristic 1–48 h occupancy forecast using same-hour averages from the past 7 days (`?sensor_id=`, `?horizon_hours=24`) |

### ML Studio

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ml/status` | Model load state, label counts, occupancy distribution, current training progress |
| `GET` | `/api/ml/labels` | All stored ground-truth frame labels |
| `POST` | `/api/ml/label` | Save or update a label: `{"file": "...", "occupancy": 2, "fever": false}` |
| `GET` | `/api/ml/infer?file=<name>` | Run heuristic and ML inference on a stored frame; returns both results and any existing label |
| `POST` | `/api/ml/train` | Trigger background model training from all labelled frames (requires ≥10 labels) |

---

## Thermal history parameters

`GET /api/thermal/history` accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensor_id` | — | Filter to a specific sensor |
| `date` | — | Filter to a date in `YYYYMMDD` format |
| `limit` | `100` | Max frames returned (1–500) |
| `offset` | `0` | Frames to skip (for pagination) |
| `include_data` | `false` | When `true`, includes full expanded pixel data in each result |

Pagination fields in the response: `has_more` (bool), `next_offset` (int or null).

```bash
# First page of metadata
curl "https://occupancy-api-container.yellowbush-1452fab1.canadacentral.azurecontainerapps.io/api/thermal/history?limit=100&offset=0"

# Full frames from one sensor on a specific date
curl "https://occupancy-api-container.yellowbush-1452fab1.canadacentral.azurecontainerapps.io/api/thermal/history?sensor_id=room-a&date=20260412&include_data=true"
```

---

## Submitting thermal data (POST /api/thermal)

Compact format (sent by ESP32 firmware):

```json
{
  "sensor_id": "room-a",
  "w": 32,
  "h": 24,
  "min": 19.5,
  "max": 34.2,
  "t": [20.1, 20.3, ...]
}
```

Fields `sensor_id`, `w`, and `h` are optional when the firmware always sends the same resolution. The `t` array is row-major, length `w × h`. Temperatures must be literal per-pixel values in degrees Celsius (floats or integers). The API does not decode 0–255 quantised arrays — if your firmware quantises readings via `min`/`max`, convert them back to degrees before submitting.

The response includes the full occupancy result: `occupancy`, `room_temperature`, `people_clusters`, `fever_count`, `any_fever`, `any_elevated`, `frame_valid`, and ML predictions (when a model is loaded).

---

## How occupancy and fever detection work

### Background subtraction

Each sensor builds a per-sensor thermal background using an exponential moving average (EMA, weight `BACKGROUND_ALPHA`). The background is updated only when the room appears empty for at least `BACKGROUND_MIN_FRAMES_EMPTY` consecutive frames. Once a background exists, occupancy detection uses temperature *above background* rather than absolute temperature, reducing false positives from fixed heat sources like monitors or HVAC vents. Backgrounds are persisted as `background_<sensor_id>.npy`.

### Clustering

Pixels above the detection threshold are clustered using connected-component labelling (8-connected). Clusters smaller than `MIN_CLUSTER_SIZE` or larger than `MAX_CLUSTER_SIZE` are ignored. Each surviving cluster is one detected person. The cluster's representative temperature is its 90th percentile pixel.

### Fever detection

A cluster with representative temperature ≥ `FEVER_THRESHOLD_C` (default 37.5 °C) is a raw fever candidate. A cluster between `FEVER_ELEVATED_THRESHOLD_C` (default 37.0 °C) and the fever threshold is flagged as elevated. The public `any_fever` response field is gated by `FEVER_MIN_CONSECUTIVE_FRAMES` (default 2) to suppress single-frame noise.

### Temporal smoothing

After clustering, the server applies:

1. **Frame sanity check** — if the frame-wide median temperature jumps more than `FRAME_ROOM_MEDIAN_MAX_JUMP_C` degrees from the previous frame, the raw count falls back to the last good value.
2. **Median smoothing** — a rolling median over the last `OCCUPANCY_SMOOTH_WINDOW` raw counts.
3. **Hysteresis** — changes smaller than `OCCUPANCY_HYSTERESIS_DELTA` from the last smoothed value are suppressed.

Response fields for the three occupancy stages:

| Field | Meaning |
|-------|---------|
| `occupancy_raw_instant` | Cluster count from this frame only |
| `occupancy_effective_raw` | Raw count after frame sanity (invalid frames repeat last good count) |
| `occupancy` | Final smoothed + hysteresis output — use this for stable room counts |
| `frame_valid` | `false` when the median jump test rejected this frame |

### ML inference (optional)

When ONNX models are present (loaded from `ml_models/` or Azure Blob `ml/`), each frame is also run through a `GradientBoostingClassifier` trained on labelled historical frames. ML results appear under the `ml` key in the POST response:

```json
"ml": {
  "ml_occupancy": 2,
  "ml_occupancy_confidence": 0.87,
  "ml_fever": false,
  "ml_fever_confidence": 0.94
}
```

The heuristic result is always present; ML results are additive and do not replace it.

---

## ML Studio workflow

The `/ml` page lets teammates build and improve ML models without leaving the browser.

### 1. Label Data tab

Load stored thermal frames (filter by sensor and date). Each 32×24 frame is shown as a thumbnail. Click a frame to expand it, set the true occupancy count and whether fever was present, then Save. A green dot marks already-labelled frames; a red dot indicates a labelled fever frame. Labels persist across restarts and are synced to Azure Blob.

### 2. Status & Train tab

Shows whether occupancy and fever models are loaded, total labelled frame counts, and the occupancy distribution across labels. Once ≥10 frames are labelled, the Train button becomes active. Training runs in the background — the log updates live every 2 seconds. On completion the ONNX models are saved locally and uploaded to Azure Blob (`ml/occupancy_model.onnx`, `ml/fever_model.onnx`) and the inference engine reloads automatically.

### 3. Run Inference tab

Browse frames, select one, and click Run Inference to see a side-by-side comparison of the heuristic result, the ML model result (with confidence), and the stored ground-truth label if one exists.

### Training offline

For larger datasets, run the training script locally (requires `scikit-learn`, `skl2onnx`, `onnx`):

```bash
source .venv/bin/activate
pip install scikit-learn skl2onnx onnx
python scripts/train_ml_models.py --data-dir thermal_data --out-dir ml_models
```

Upload the resulting `.onnx` files to Azure Blob:

```bash
az storage blob upload -f ml_models/occupancy_model.onnx \
  --container-name iotoccupancydata --name ml/occupancy_model.onnx \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

---

## Local development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` for the dashboard and `http://localhost:8000/ml` for ML Studio.

Install dev dependencies for running the test suite:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

---

## Docker

```bash
docker build -t occupancy-api .
docker run -p 8000:8000 -e PORT=8000 occupancy-api

# With persistent local storage
docker run -p 8000:8000 -e PORT=8000 \
  -v "$(pwd)/thermal_data:/app/thermal_data" \
  occupancy-api
```

---

## Deployment (CI/CD)

A GitHub Actions workflow builds the Docker image, pushes it to Azure Container Registry (`occupancyregistry.azurecr.io`), and deploys to the Container App on every push to `main`. Required GitHub secrets:

| Secret | Value |
|--------|-------|
| `OCCUPANCYAPICONTAINER_AZURE_CLIENT_ID` | Service principal client ID |
| `OCCUPANCYAPICONTAINER_AZURE_TENANT_ID` | Azure tenant ID |
| `OCCUPANCYAPICONTAINER_AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `OCCUPANCYAPICONTAINER_REGISTRY_USERNAME` | ACR username |
| `OCCUPANCYAPICONTAINER_REGISTRY_PASSWORD` | ACR password |

---

## Environment variables

Set in Container App environment or as `-e` flags when running Docker locally:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port to bind (Azure sets this automatically) |
| `THERMAL_DATA_DIR` | `thermal_data` | Directory for thermal frames and occupancy logs; in the image: `/app/thermal_data` |
| `AZURE_STORAGE_CONNECTION_STRING` | — | Enables Azure Blob Storage for durable frame and label storage |
| `AZURE_STORAGE_CONTAINER_NAME` | `iotoccupancydata` | Blob container name |
| `SAVE_TO_BLOB` | `true` when connection string is set | Write data to Azure Blob |
| `SAVE_LOCAL_DATA` | `false` when Blob is enabled | Write data to local disk. **Note:** the ML Studio (`/ml`) reads thermal frames from local disk via the history API; set this to `true` (and mount persistent storage) if you want ML Studio to work in a Blob-enabled deployment. |
| `ML_MODEL_DIR` | `ml_models` | Local directory where ONNX models are cached |
| `BACKGROUND_ALPHA` | `0.95` | EMA weight for thermal background update |
| `BACKGROUND_MIN_FRAMES_EMPTY` | `3` | Consecutive empty frames before background update |
| `BACKGROUND_MAX_MEAN_ABS_DELTA_C` | `2.5` | Max frame-to-frame delta (°C) between empty frames; larger resets the empty streak |
| `ROOM_TEMP_THRESHOLD` | `0.5` | °C above room estimate (delta mode) a pixel must be to count as human |
| `FEVER_THRESHOLD_C` | `37.5` | °C at or above which a cluster is flagged as fever |
| `FEVER_ELEVATED_THRESHOLD_C` | `37.0` | Lower bound for the elevated-temperature band |
| `FEVER_MIN_CONSECUTIVE_FRAMES` | `2` | Frames of consecutive raw fever before `any_fever` is set |
| `OCCUPANCY_SMOOTH_WINDOW` | `5` | Rolling median window length for occupancy smoothing |
| `OCCUPANCY_HYSTERESIS_DELTA` | `1` | Suppress occupancy changes smaller than this from the last smoothed value |
| `FRAME_ROOM_MEDIAN_MAX_JUMP_C` | `4.0` | Max median temperature jump (°C) before a frame is marked invalid; `0` disables |

---

## Analysis scripts

All scripts require the activated virtualenv (`source .venv/bin/activate`). See [scripts/README.md](scripts/README.md) for full usage.

| Script | Description |
|--------|-------------|
| `scripts/compare_occupancy_accuracy.py` | Score stored occupancy logs against a CSV ground-truth file |
| `scripts/compare_fever_accuracy.py` | Score fever flags against a CSV ground-truth file |
| `scripts/replay_thermal_occupancy.py` | Re-run stored compact frames through the pipeline offline |
| `scripts/calibrate_occupancy_thresholds.py` | Grid search over `ROOM_TEMP_THRESHOLD` / `MIN_CLUSTER_SIZE` |
| `scripts/train_ml_models.py` | Train occupancy and fever ONNX models from labelled frame data |
