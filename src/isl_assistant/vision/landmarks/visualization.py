"""Drawing utilities for inspecting extracted MediaPipe landmarks."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
POSE_SLICE = slice(42, 51)

# MediaPipe hand landmark topology.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# Selected pose order: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
)

LEFT_HAND_COLOR = (80, 220, 80)       # BGR green
RIGHT_HAND_COLOR = (255, 150, 40)     # BGR blue
POSE_COLOR = (40, 170, 255)           # BGR orange


def normalized_to_pixel(
    x: float, y: float, width: int, height: int
) -> tuple[int, int] | None:
    """Convert normalized coordinates into a safe pixel location."""
    if width <= 0 or height <= 0 or not np.isfinite([x, y]).all():
        return None
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return None
    return min(round(x * (width - 1)), width - 1), min(
        round(y * (height - 1)), height - 1
    )


def _draw_group(
    image: np.ndarray,
    coordinates: np.ndarray,
    detected: np.ndarray,
    connections: Iterable[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    pixels = [
        normalized_to_pixel(float(point[0]), float(point[1]), width, height)
        if is_detected
        else None
        for point, is_detected in zip(coordinates, detected, strict=True)
    ]

    for start, end in connections:
        if pixels[start] is not None and pixels[end] is not None:
            cv2.line(image, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
    for point in pixels:
        if point is not None:
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
            cv2.circle(image, point, 4, (255, 255, 255), 1, cv2.LINE_AA)


def detection_status(detected: np.ndarray) -> dict[str, bool]:
    """Return group-level visibility flags for one frame."""
    mask = np.asarray(detected, dtype=bool)
    if mask.shape != (51,):
        raise ValueError(f"detected must have shape (51,), got {mask.shape}")
    return {
        "left": bool(mask[LEFT_HAND_SLICE].any()),
        "right": bool(mask[RIGHT_HAND_SLICE].any()),
        "pose": bool(mask[POSE_SLICE].any()),
    }


def draw_frame_landmarks(
    image: np.ndarray,
    frame_landmarks: object,
    *,
    frame_number: int | None = None,
    timestamp_ms: int | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw a ``FrameLandmarks``-compatible object over a BGR frame."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 BGR array with shape [H, W, 3]")

    coordinates = np.asarray(getattr(frame_landmarks, "coordinates"))
    detected = np.asarray(getattr(frame_landmarks, "detected"), dtype=bool)
    if coordinates.shape != (51, 3):
        raise ValueError(f"coordinates must have shape (51, 3), got {coordinates.shape}")
    if detected.shape != (51,):
        raise ValueError(f"detected must have shape (51,), got {detected.shape}")

    output = image.copy() if copy else image
    _draw_group(
        output, coordinates[LEFT_HAND_SLICE], detected[LEFT_HAND_SLICE],
        HAND_CONNECTIONS, LEFT_HAND_COLOR,
    )
    _draw_group(
        output, coordinates[RIGHT_HAND_SLICE], detected[RIGHT_HAND_SLICE],
        HAND_CONNECTIONS, RIGHT_HAND_COLOR,
    )
    _draw_group(
        output, coordinates[POSE_SLICE], detected[POSE_SLICE],
        POSE_CONNECTIONS, POSE_COLOR,
    )

    status = detection_status(detected)
    parts = []
    if frame_number is not None:
        parts.append(f"frame {frame_number}")
    if timestamp_ms is not None:
        parts.append(f"{timestamp_ms} ms")
    parts.extend(
        f"{name}:{'yes' if present else 'no'}" for name, present in status.items()
    )
    text = " | ".join(parts)
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 1, 700), 42), (0, 0, 0), -1)
    cv2.putText(output, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    return output
