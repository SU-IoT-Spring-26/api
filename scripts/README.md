# Scripts

## compare_occupancy_accuracy.py

Compares API occupancy estimates to ground truth for accuracy quantification (e.g. for experiments).

### Ground-truth CSV format

Header (optional):

```text
timestamp,sensor_id,actual_count
```

Rows:

```text
2025-02-07T14:30:00,living-room,2
2025-02-07T14:35:00,living-room,1
2025-02-07T15:00:00,office,0
```

- **timestamp**: ISO 8601 (e.g. `YYYY-MM-DDTHH:MM:SS` or with timezone).
- **sensor_id**: Must match `sensor_id` in the occupancy API data.
- **actual_count**: Integer ground-truth occupancy count.

### Usage

```bash
# Use default data dir (THERMAL_DATA_DIR or ./thermal_data)
python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv

# Specify occupancy data directory
python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv --data-dir /path/to/thermal_data

# Match API readings within 60 seconds of each ground-truth timestamp
python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv --window 60

# Print per-row comparison
python scripts/compare_occupancy_accuracy.py path/to/ground_truth.csv -v
```

### Output

- **MAE**: Mean absolute error (average of |estimated − actual|).
- **Accuracy (within 1)**: Fraction of rows where |estimated − actual| ≤ 1.

Readings are aligned by finding all API occupancy entries for the same sensor within `--window` seconds of each ground-truth timestamp; the average occupancy of those entries is used as the estimate for that row.
