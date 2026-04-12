"""
ml/ — machine learning inference for occupancy and fever detection.

Architecture
------------
Models are trained offline (see scripts/train_ml_models.py), exported to ONNX,
and stored in Azure Blob under the `ml/` prefix (or locally under `ml_models/`).
At startup the FastAPI app downloads whichever models are available and registers
them here. If no model is present the functions fall back to returning None so the
heuristic path in main.py remains active.

Inference is inline (onnxruntime on CPU) — no external API calls at request time.
This keeps latency near zero and avoids any per-request cost or rate limits.

Retraining pipeline
-------------------
Run manually via scripts/train_ml_models.py (or trigger from the ML Studio
web UI at /ml). The pipeline:
  1. Reads compact thermal frames and ground-truth labels from local
     THERMAL_DATA_DIR (the same directory the API writes to).
  2. Builds feature vectors (see ml/features.py).
  3. Trains sklearn GradientBoostingClassifier (occupancy, multi-class) and
     GradientBoostingClassifier (fever, binary).
  4. Exports both to ONNX via skl2onnx.
  5. Saves models to ML_MODEL_DIR and uploads to Azure Blob:
     ml/occupancy_model.onnx, ml/fever_model.onnx.
"""

from ml.inference import MLInferenceEngine

__all__ = ["MLInferenceEngine"]
