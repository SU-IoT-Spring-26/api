"""
Feature extraction for ML models.

Both the training pipeline and the inference engine use this module so the
feature vector is always identical between training and serving.

Input
-----
temp_array  : np.ndarray shape (H, W) — absolute temperatures in °C
background  : np.ndarray shape (H, W) | None — per-sensor EMA background

Output
------
np.ndarray shape (N_FEATURES,) dtype float32
"""

from __future__ import annotations

import numpy as np

# ----- feature vector length (must match whatever the model was trained on) -----
# flat raw pixels (768) + derived stats (see below)
_SENSOR_H = 24
_SENSOR_W = 32
_N_PIXELS = _SENSOR_H * _SENSOR_W  # 768
_N_STATS = 16
N_FEATURES = _N_PIXELS + _N_STATS


def extract(temp_array: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """
    Return a float32 feature vector for one thermal frame.

    The vector is:
      [0:768]   raw pixel temperatures (row-major, normalised to 0–1 via
                (T - 15) / 25  — maps 15°C→0, 40°C→1, clipped)
      [768:784] 16 derived statistics (see _stats)

    Using normalised pixels rather than delta-from-background means the model
    can be useful even before a background has been established, and makes the
    features consistent across sensors in different ambient temperatures.
    """
    arr = temp_array.astype(np.float32)
    if arr.shape != (_SENSOR_H, _SENSOR_W):
        # Tolerate transposed or differently-shaped frames from old firmware
        arr = arr.flatten()
        if arr.size >= _N_PIXELS:
            arr = arr[:_N_PIXELS].reshape(_SENSOR_H, _SENSOR_W)
        else:
            arr = np.pad(arr, (0, _N_PIXELS - arr.size)).reshape(_SENSOR_H, _SENSOR_W)

    # Normalise pixels
    pixels_norm = np.clip((arr - 15.0) / 25.0, 0.0, 1.0).flatten()  # (768,)

    # Only pass background when it matches arr's shape after any reshape/pad;
    # otherwise _background_stats would misbroadcast or raise.
    bg = background if (background is not None and background.shape == arr.shape) else None
    stats = _stats(arr, bg)
    return np.concatenate([pixels_norm, stats]).astype(np.float32)


def _stats(arr: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    """Compute 16 summary statistics from a thermal frame."""
    flat = arr.flatten()
    ambient = float(np.percentile(flat, 10))

    # Pixels above likely-human thresholds (absolute and relative)
    above_abs_30 = float(np.mean(flat > 30.0))
    above_abs_33 = float(np.mean(flat > 33.0))
    above_rel_2 = float(np.mean(flat > ambient + 2.0))
    above_rel_4 = float(np.mean(flat > ambient + 4.0))
    above_rel_6 = float(np.mean(flat > ambient + 6.0))

    frame_stats = np.array([
        float(np.min(flat)),
        float(np.max(flat)),
        float(np.mean(flat)),
        float(np.median(flat)),
        float(np.std(flat)),
        float(np.percentile(flat, 75)),
        float(np.percentile(flat, 90)),
        float(np.percentile(flat, 99)),
        ambient,
        above_abs_30,
        above_abs_33,
        above_rel_2,
        above_rel_4,
        above_rel_6,
        # 2 background-delta stats (0 when no background)
        *(_background_stats(arr, background) if background is not None else [0.0, 0.0]),
    ], dtype=np.float32)

    assert len(frame_stats) == _N_STATS, f"stat count mismatch: {len(frame_stats)}"
    return frame_stats


def _background_stats(arr: np.ndarray, background: np.ndarray) -> list[float]:
    delta = np.maximum(0.0, arr - background)
    return [float(np.mean(delta)), float(np.max(delta))]
