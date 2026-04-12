"""
ML inference engine — loaded once at startup, called per thermal frame.

Usage in main.py
----------------
    from ml import MLInferenceEngine
    _ml = MLInferenceEngine()

    # at startup (inside lifespan or startup event):
    await _ml.load(blob_container_client, local_model_dir)

    # per frame:
    result = _ml.predict(temp_array_2d, background_array)
    # result is None when no model is loaded (heuristic path stays active)

Models
------
Two ONNX models, each a binary GradientBoosting classifier exported via skl2onnx:

  occupancy_model.onnx
    input:  float32 (1, N_FEATURES)
    output: label (int64) — predicted occupancy count (capped at MAX_OCCUPANCY)
            probabilities (float32, N_CLASSES)

  fever_model.onnx
    input:  float32 (1, N_FEATURES)
    output: label (int64) — 1 if any fever likely, 0 otherwise
            probabilities (float32, 2)

Both models are optional; whichever is present is used.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ml.features import extract

logger = logging.getLogger(__name__)

_OCCUPANCY_BLOB = "ml/occupancy_model.onnx"
_FEVER_BLOB = "ml/fever_model.onnx"
_LOCAL_MODEL_DIR = Path(os.environ.get("ML_MODEL_DIR", "ml_models"))


class MLInferenceEngine:
    """Holds loaded ONNX sessions and exposes a single predict() call."""

    def __init__(self) -> None:
        self._occ_session: Any = None   # onnxruntime.InferenceSession | None
        self._fever_session: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, blob_container=None) -> None:
        """
        Try to load ONNX models.

        Priority order:
          1. Local directory (ML_MODEL_DIR env var, default ml_models/)
          2. Azure Blob Storage (blob_container must be a BlobContainerClient)

        Silently skips any model that cannot be loaded; the heuristic remains
        active for unloaded models.
        """
        self._occ_session = self._load_model(
            _LOCAL_MODEL_DIR / "occupancy_model.onnx",
            _OCCUPANCY_BLOB,
            blob_container,
            "occupancy",
        )
        self._fever_session = self._load_model(
            _LOCAL_MODEL_DIR / "fever_model.onnx",
            _FEVER_BLOB,
            blob_container,
            "fever",
        )
        self._loaded = self._occ_session is not None or self._fever_session is not None
        if self._loaded:
            parts = []
            if self._occ_session:
                parts.append("occupancy")
            if self._fever_session:
                parts.append("fever")
            logger.info("ML models loaded: %s", ", ".join(parts))
        else:
            logger.info("No ML models found — heuristic inference only")

    def _load_model(
        self,
        local_path: Path,
        blob_name: str,
        blob_container,
        label: str,
    ):
        """Return an onnxruntime.InferenceSession or None."""
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            logger.debug("onnxruntime not installed; ML inference disabled")
            return None

        # 1. Local file
        if local_path.exists():
            try:
                sess = ort.InferenceSession(str(local_path), providers=["CPUExecutionProvider"])
                logger.info("Loaded %s model from local path %s", label, local_path)
                return sess
            except Exception as exc:
                logger.warning("Failed to load local %s model: %s", label, exc)

        # 2. Azure Blob
        if blob_container is not None:
            try:
                blob_client = blob_container.get_blob_client(blob_name)
                data = blob_client.download_blob().readall()
                _LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                sess = ort.InferenceSession(str(local_path), providers=["CPUExecutionProvider"])
                logger.info("Downloaded and loaded %s model from Blob (%s)", label, blob_name)
                return sess
            except Exception as exc:
                logger.debug("Could not load %s model from Blob: %s", label, exc)

        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._loaded

    def predict(
        self,
        temp_array: np.ndarray,
        background: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Run ML inference on one thermal frame.

        Returns a dict (or None if no models loaded):
          {
            "ml_occupancy": int | None,
            "ml_occupancy_confidence": float | None,
            "ml_fever": bool | None,
            "ml_fever_confidence": float | None,
          }
        """
        if not self._loaded:
            return None

        features = extract(temp_array, background).reshape(1, -1)

        occ_count = None
        occ_conf = None
        if self._occ_session is not None:
            try:
                input_name = self._occ_session.get_inputs()[0].name
                outputs = self._occ_session.run(None, {input_name: features})
                # outputs[0] = label array, outputs[1] = probability map
                occ_count = int(outputs[0][0])
                if len(outputs) > 1:
                    probs = outputs[1]
                    # probs may be a list of dicts (skl2onnx ZipMap) or an ndarray
                    if isinstance(probs, list) and isinstance(probs[0], dict):
                        occ_conf = float(probs[0].get(occ_count, 0.0))
                    elif hasattr(probs, "shape"):
                        occ_conf = float(np.max(probs[0]))
            except Exception as exc:
                logger.warning("Occupancy model inference error: %s", exc)

        fever_flag = None
        fever_conf = None
        if self._fever_session is not None:
            try:
                input_name = self._fever_session.get_inputs()[0].name
                outputs = self._fever_session.run(None, {input_name: features})
                fever_flag = bool(int(outputs[0][0]) == 1)
                if len(outputs) > 1:
                    probs = outputs[1]
                    if isinstance(probs, list) and isinstance(probs[0], dict):
                        fever_conf = float(probs[0].get(1, 0.0))
                    elif hasattr(probs, "shape"):
                        fever_conf = float(probs[0][1] if probs.shape[-1] > 1 else probs[0][0])
            except Exception as exc:
                logger.warning("Fever model inference error: %s", exc)

        return {
            "ml_occupancy": occ_count,
            "ml_occupancy_confidence": occ_conf,
            "ml_fever": fever_flag,
            "ml_fever_confidence": fever_conf,
        }
