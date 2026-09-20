from types import SimpleNamespace

import numpy as np
import pytest

from isl_assistant.vision.landmarks.visualization import (
    detection_status,
    draw_frame_landmarks,
    normalized_to_pixel,
)


def test_normalized_to_pixel_handles_boundaries_and_invalid_values() -> None:
    assert normalized_to_pixel(0.0, 0.0, 100, 50) == (0, 0)
    assert normalized_to_pixel(1.0, 1.0, 100, 50) == (99, 49)
    assert normalized_to_pixel(-0.01, 0.5, 100, 50) is None
    assert normalized_to_pixel(float("nan"), 0.5, 100, 50) is None


def test_detection_status_uses_feature_groups() -> None:
    detected = np.zeros(51, dtype=bool)
    detected[0] = True
    detected[42] = True
    assert detection_status(detected) == {
        "left": True,
        "right": False,
        "pose": True,
    }


def test_draw_preserves_source_by_default_and_draws_visible_points() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    original = image.copy()
    coordinates = np.zeros((51, 3), dtype=np.float32)
    coordinates[:, :2] = 0.5
    detected = np.zeros(51, dtype=bool)
    detected[0] = True
    frame = SimpleNamespace(coordinates=coordinates, detected=detected)

    rendered = draw_frame_landmarks(image, frame, frame_number=1, timestamp_ms=40)

    assert rendered.shape == image.shape
    assert rendered.dtype == np.uint8
    assert np.array_equal(image, original)
    assert np.any(rendered != image)


def test_draw_rejects_incorrect_feature_shape() -> None:
    frame = SimpleNamespace(
        coordinates=np.zeros((50, 3), dtype=np.float32),
        detected=np.zeros(50, dtype=bool),
    )
    with pytest.raises(ValueError, match="coordinates"):
        draw_frame_landmarks(np.zeros((10, 10, 3), dtype=np.uint8), frame)
