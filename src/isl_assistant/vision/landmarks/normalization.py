"""Coordinate normalization for signer-independent landmark features."""

from __future__ import annotations

import numpy as np


LANDMARK_COUNT = 51
LEFT_SHOULDER_INDEX = 43
RIGHT_SHOULDER_INDEX = 44


def normalize_sequence(
    coordinates: np.ndarray,
    detected: np.ndarray,
    valid_timesteps: np.ndarray,
    *,
    minimum_shoulder_width: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Center each valid frame on the shoulders and scale by shoulder width.

    Frames without two detected shoulders remain zero and are marked false in
    the returned normalization mask. Missing landmarks always remain zero.
    """
    points = np.asarray(coordinates, dtype=np.float32)
    visible = np.asarray(detected, dtype=bool)
    valid = np.asarray(valid_timesteps, dtype=bool)

    if points.ndim != 3 or points.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(
            f"coordinates must have shape [T, {LANDMARK_COUNT}, 3], got {points.shape}"
        )
    if visible.shape != points.shape[:2]:
        raise ValueError("detected must have shape [T, 51]")
    if valid.shape != (points.shape[0],):
        raise ValueError("valid_timesteps must have shape [T]")

    normalized = np.zeros_like(points, dtype=np.float32)
    normalized_timesteps = np.zeros(points.shape[0], dtype=bool)

    for frame_index in range(points.shape[0]):
        if not valid[frame_index]:
            continue
        if not (
            visible[frame_index, LEFT_SHOULDER_INDEX]
            and visible[frame_index, RIGHT_SHOULDER_INDEX]
        ):
            continue

        left = points[frame_index, LEFT_SHOULDER_INDEX]
        right = points[frame_index, RIGHT_SHOULDER_INDEX]
        shoulder_width = float(np.linalg.norm(left[:2] - right[:2]))
        if not np.isfinite(shoulder_width) or shoulder_width < minimum_shoulder_width:
            continue

        origin = (left + right) / 2.0
        frame_visible = visible[frame_index]
        normalized[frame_index, frame_visible] = (
            points[frame_index, frame_visible] - origin
        ) / shoulder_width
        normalized_timesteps[frame_index] = True

    return normalized, normalized_timesteps
