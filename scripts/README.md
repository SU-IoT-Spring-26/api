# Scripts

Tools for accuracy evaluation and offline calibration. From the **repository root**, activate the project venv and use its `python`:

```bash
source .venv/bin/activate
pip install -r requirements.txt   # once, if not already done
```

Then run the examples below with that same activated environment.

## compare_occupancy_accuracy.py

Compares occupancy estimates to ground truth using daily `occupancy_YYYYMMDD.jsonl` under `THERMAL_DATA_DIR` (or `--data-dir`).

### Ground-truth CSV

Header optional:

```text
timestamp,sensor_id,actual_count
```

Example rows:

```text
2025-02-07T14:30:00,living-room,2
2025-02-07T14:35:00,living-room,1
```

### Usage

```bash
python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv

python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv --data-dir /path/to/thermal_data

# Compare smoothed output vs pre-processed columns in one run
python scripts/compare_occupancy_accuracy.py path/to/gt.csv \
  --compare-fields occupancy,occupancy_effective_raw,occupancy_raw_instant

# Single field (default occupancy)
python scripts/compare_occupancy_accuracy.py path/to/gt.csv --field occupancy_effective_raw --window 60 -v
```

### Output

- **MAE**: mean absolute error vs `actual_count`.
- **Accuracy (within 1)**: fraction of rows where |estimated − actual| ≤ 1.

Readings are aligned by taking all JSONL lines for the same `sensor_id` within `--window` seconds of each ground-truth timestamp; the **average** of the chosen field over those lines is the estimate for that row.

---

## compare_fever_accuracy.py

Binary fever labels vs occupancy JSONL.

### Ground-truth CSV

```text
timestamp,sensor_id,fever
2025-02-07T14:30:00,living-room,1
```

`fever` is `0` or `1`.

### Usage

```bash
python scripts/compare_fever_accuracy.py path/to/fever_gt.csv --field any_fever
python scripts/compare_fever_accuracy.py path/to/fever_gt.csv --field any_fever_raw
```

Prints aligned row count, accuracy, and balanced accuracy (mean of sensitivity and specificity on aligned points).

---

## replay_thermal_occupancy.py

Replays `thermal_*_compact.json` frames through `estimate_occupancy` and (by default) `apply_occupancy_signal_processing` + background updates. Does **not** write thermal or occupancy files unless you pass `--persist-background` (background `.npy` only). Uses **`main.DATA_DIR`** = `--data-dir` so `background_*.npy` loads match that folder.

```bash
python scripts/replay_thermal_occupancy.py --data-dir ./thermal_data

python scripts/replay_thermal_occupancy.py --data-dir ./thermal_data --sensor-id living-room --output-csv replay.csv

# Clustering only (no smoothing / background updates)
python scripts/replay_thermal_occupancy.py --data-dir ./thermal_data --pipeline estimate_only

# Continue in-memory state and load saved backgrounds from disk
python scripts/replay_thermal_occupancy.py --data-dir ./thermal_data --no-fresh --load-background
```

Default is **`--fresh`**: clears pipeline state before replay. Use **`--no-fresh`** to keep deques and backgrounds in memory across the run.

---

## calibrate_occupancy_thresholds.py

Runs a small parameter grid by mutating `main` globals, then `replay_thermal_occupancy.replay_compact_frames`, then the same alignment as `compare_occupancy_accuracy`. Use for offline threshold sweeps once you have compact thermal history and a count CSV.

```bash
python scripts/calibrate_occupancy_thresholds.py path/to/ground_truth.csv \
  --data-dir ./thermal_data \
  --room-threshold-grid 0.35 0.5 0.65 \
  --min-cluster-grid 2 3 \
  --field occupancy_effective_raw
```

Omit the grid arguments to score **current** code constants once. Restore your environment after experiments (the script resets `ROOM_TEMP_THRESHOLD` and `MIN_CLUSTER_SIZE` to their values at import time).
