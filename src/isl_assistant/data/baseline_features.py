"""Fixed-size feature engineering for classical landmark-model baselines."""

from __future__ import annotations

import numpy as np

LANDMARK_COUNT = 51
COORDINATE_DIMENSIONS = 3


def sequence_to_summary_features(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    detected: np.ndarray,
    valid_timesteps: np.ndarray,
) -> np.ndarray:
    """Convert one landmark sequence into deterministic SVM-ready features.

    The summary keeps location, range, start/end position, movement, confidence,
    and landmark detection-rate information. Only real, non-padded frames are
    used for statistics.
    """

    values = np.asarray(coordinates, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    mask = np.asarray(detected, dtype=np.bool_)
    valid = np.asarray(valid_timesteps, dtype=np.bool_)
    sequence_length = values.shape[0]
    if values.shape != (sequence_length, LANDMARK_COUNT, COORDINATE_DIMENSIONS):
        raise ValueError("coordinates must have shape [time, 51, 3]")
    if scores.shape != (sequence_length, LANDMARK_COUNT):
        raise ValueError("confidence must have shape [time, 51]")
    if mask.shape != (sequence_length, LANDMARK_COUNT):
        raise ValueError("detected must have shape [time, 51]")
    if valid.shape != (sequence_length,):
        raise ValueError("valid_timesteps must have shape [time]")

    frames = values[valid]
    frame_scores = scores[valid]
    frame_mask = mask[valid]
    if not len(frames):
        raise ValueError("at least one valid timestep is required")

    if len(frames) == 1:
        velocity = np.zeros_like(frames)
    else:
        velocity = np.diff(frames, axis=0)
    parts = (
        frames.mean(axis=0).ravel(),
        frames.std(axis=0).ravel(),
        frames.min(axis=0).ravel(),
        frames.max(axis=0).ravel(),
        frames[0].ravel(),
        frames[-1].ravel(),
        velocity.mean(axis=0).ravel(),
        velocity.std(axis=0).ravel(),
        frame_scores.mean(axis=0),
        frame_scores.std(axis=0),
        frame_mask.mean(axis=0, dtype=np.float32),
    )
    return np.concatenate(parts, dtype=np.float32)
