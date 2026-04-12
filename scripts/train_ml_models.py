#!/usr/bin/env python3
"""
Train occupancy-count and fever-detection models from stored thermal data.

Usage (local, with activated venv):
    python scripts/train_ml_models.py [--data-dir thermal_data] [--out-dir ml_models]

After training, upload the .onnx files to Azure Blob under the ml/ prefix so the
API picks them up on next startup:
    az storage blob upload -f ml_models/occupancy_model.onnx \\
        --container-name iotoccupancydata --name ml/occupancy_model.onnx \\
        --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

Requirements (not in requirements.txt — install in your training environment):
    pip install scikit-learn skl2onnx onnx onnxruntime

The models will also work without ONNX if you save the sklearn estimator with
joblib; see the SAVE_SKLEARN_PICKLE option below.

Data format expected
--------------------
The script looks for pairs of files produced by the API's compact storage format:
  - thermal_<sensor>_<ts>_compact.json.gz  (or .json) — thermal frame
  - occupancy_YYYYMMDD.jsonl (or .jsonl.gz) — logged occupancy values

Occupancy ground truth: the "occupancy" field written by the heuristic is used as
a proxy label. Replace with real ground-truth labels if you have them (e.g. from
compare_occupancy_accuracy.py output).

Fever ground truth: any frame where any cluster's representative_temp_c >= 37.5°C
is labelled fever=1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so ml/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import extract  # noqa: E402

SAVE_SKLEARN_PICKLE = False  # set True to also save .pkl alongside .onnx


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path) -> tuple[list[np.ndarray], list[int], list[int]]:
    """
    Returns:
        features  : list of float32 feature vectors (one per frame)
        occ_labels: list of int occupancy counts
        fever_labels: list of int (0 or 1)
    """
    features: list[np.ndarray] = []
    occ_labels: list[int] = []
    fever_labels: list[int] = []

    # Collect compact thermal frames
    compact_files = sorted(data_dir.glob("thermal_*_compact.json*"))
    if not compact_files:
        print(f"No compact thermal files found in {data_dir}")
        return features, occ_labels, fever_labels

    # Load occupancy log index (date → {timestamp → occupancy})
    occ_index: dict[str, int] = {}
    for occ_file in sorted(data_dir.glob("occupancy_*.jsonl*")):
        opener = gzip.open if occ_file.suffix == ".gz" else open
        try:
            with opener(occ_file, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    ts = entry.get("timestamp") or entry.get("ts")
                    occ = entry.get("occupancy")
                    fever = 1 if entry.get("any_fever") else 0
                    if ts and occ is not None:
                        occ_index[str(ts)[:19]] = (int(occ), int(fever))
        except Exception as exc:
            print(f"Warning: could not read {occ_file}: {exc}")

    loaded = 0
    for path in compact_files:
        try:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt") as f:
                frame = json.load(f)

            # Extract temperature array
            temps_raw = frame.get("t") or frame.get("temperatures") or frame.get("pixels")
            w = frame.get("w", 32)
            h = frame.get("h", 24)
            if not temps_raw or len(temps_raw) != w * h:
                continue

            # Decode compact quantised format if needed
            if isinstance(temps_raw[0], int):
                min_t = frame.get("min", 0)
                max_t = frame.get("max", 50)
                n = len(temps_raw)
                temps = [min_t + (max_t - min_t) * v / 255.0 for v in temps_raw]
            else:
                temps = temps_raw

            arr = np.array(temps, dtype=np.float32).reshape(h, w)

            # Match to occupancy log by timestamp in filename
            # filename: thermal_<sensor>_<YYYYMMDD_HHMMSS>_compact.json(.gz)
            ts_str = None
            parts = path.stem.replace(".json", "").split("_")
            # look for date+time pattern YYYYMMDD_HHMMSS
            for i, p in enumerate(parts):
                if len(p) == 8 and p.isdigit() and i + 1 < len(parts) and len(parts[i + 1]) == 6:
                    ts_str = f"{p[:4]}-{p[4:6]}-{p[6:]}T{parts[i+1][:2]}:{parts[i+1][2:4]}:{parts[i+1][4:]}"
                    break

            occ, fever = (0, 0)
            if ts_str:
                match = occ_index.get(ts_str[:19])
                if match:
                    occ, fever = match

            feat = extract(arr)
            features.append(feat)
            occ_labels.append(occ)
            fever_labels.append(fever)
            loaded += 1
        except Exception as exc:
            print(f"Warning: skipping {path.name}: {exc}")

    print(f"Loaded {loaded} frames from {data_dir}")
    return features, occ_labels, fever_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_export(
    features: list[np.ndarray],
    occ_labels: list[int],
    fever_labels: list[int],
    out_dir: Path,
) -> None:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn skl2onnx onnx")
        sys.exit(1)

    X = np.array(features, dtype=np.float32)
    y_occ = np.array(occ_labels, dtype=np.int64)
    y_fever = np.array(fever_labels, dtype=np.int64)

    if len(X) < 20:
        print(f"Only {len(X)} samples — need at least 20 for meaningful training. Aborting.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Occupancy model ---
    print("\n--- Occupancy model ---")
    print(f"  Classes: {sorted(set(y_occ))}, samples: {len(y_occ)}")
    occ_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    scores = cross_val_score(occ_clf, X, y_occ, cv=min(5, len(X) // 5), scoring="accuracy")
    print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    occ_clf.fit(X, y_occ)
    _export_onnx(occ_clf, X.shape[1], out_dir / "occupancy_model.onnx", "occupancy_model")
    if SAVE_SKLEARN_PICKLE:
        import joblib
        joblib.dump(occ_clf, out_dir / "occupancy_model.pkl")
        print(f"  Saved sklearn pickle: {out_dir / 'occupancy_model.pkl'}")

    # --- Fever model ---
    print("\n--- Fever model ---")
    fever_pos = int(y_fever.sum())
    print(f"  Fever-positive samples: {fever_pos}/{len(y_fever)}")
    if fever_pos < 5:
        print("  Too few fever-positive samples to train a useful model. Skipping.")
    else:
        fever_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores = cross_val_score(fever_clf, X, y_fever, cv=min(5, len(X) // 5), scoring="f1")
        print(f"  CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
        fever_clf.fit(X, y_fever)
        _export_onnx(fever_clf, X.shape[1], out_dir / "fever_model.onnx", "fever_model")
        if SAVE_SKLEARN_PICKLE:
            import joblib
            joblib.dump(fever_clf, out_dir / "fever_model.pkl")
            print(f"  Saved sklearn pickle: {out_dir / 'fever_model.pkl'}")

    print(f"\nModels written to {out_dir}/")


def _export_onnx(clf, n_features: int, out_path: Path, name: str) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("  skl2onnx not installed — skipping ONNX export. Run: pip install skl2onnx onnx")
        return

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, name=name, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Exported ONNX: {out_path} ({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train occupancy/fever ML models")
    parser.add_argument("--data-dir", default="thermal_data", help="Directory with thermal data files")
    parser.add_argument("--out-dir", default="ml_models", help="Output directory for model files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    features, occ_labels, fever_labels = load_dataset(data_dir)
    if not features:
        print("No usable training data found.")
        sys.exit(1)

    train_and_export(features, occ_labels, fever_labels, out_dir)


if __name__ == "__main__":
    main()
