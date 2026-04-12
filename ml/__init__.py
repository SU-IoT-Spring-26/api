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
  1. Pulls compact thermal frames + occupancy JSONL from Azure Blob.
  2. Builds feature vectors (see ml/features.py).
  3. Trains sklearn GradientBoostingClassifier (occupancy) and
     GradientBoostingClassifier (fever).
  4. Exports both to ONNX via skl2onnx.
  5. Uploads to Azure Blob: ml/occupancy_model.onnx, ml/fever_model.onnx.
"""

from ml.inference import MLInferenceEngine

__all__ = ["MLInferenceEngine"]
