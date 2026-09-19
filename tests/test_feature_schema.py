from pathlib import Path

import numpy as np
import pytest

from isl_assistant.vision.landmarks.feature_schema import (
    COORDINATE_DIMENSION,
    HAND_LANDMARK_COUNT,
    LANDMARK_NAMES,
    POSE_LANDMARK_INDICES,
    SCHEMA_VERSION,
    TOTAL_LANDMARK_COUNT,
    FrameLandmarks,
    LandmarkSequence,
    empty_frame,
)


def test_schema_has_two_hands_and_nine_pose_landmarks() -> None:
    assert TOTAL_LANDMARK_COUNT == 51
    assert HAND_LANDMARK_COUNT == 21
    assert len(POSE_LANDMARK_INDICES) == 9
    assert len(LANDMARK_NAMES) == TOTAL_LANDMARK_COUNT
    assert len(set(LANDMARK_NAMES)) == TOTAL_LANDMARK_COUNT


def test_empty_frame_uses_zero_values_and_no_detections() -> None:
    frame = empty_frame(timestamp_ms=120)

    assert frame.coordinates.shape == (TOTAL_LANDMARK_COUNT, COORDINATE_DIMENSION)
    assert not np.any(frame.coordinates)
    assert not np.any(frame.confidence)
    assert not np.any(frame.detected)
    assert frame.timestamp_ms == 120


def test_frame_rejects_nonzero_coordinates_for_missing_landmark() -> None:
    coordinates = np.zeros((TOTAL_LANDMARK_COUNT, 3), dtype=np.float32)
    confidence = np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.float32)
    detected = np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.bool_)
    coordinates[0] = (0.1, 0.2, 0.3)

    with pytest.raises(ValueError, match="undetected landmarks"):
        FrameLandmarks(coordinates, confidence, detected, timestamp_ms=0)


def make_sequence(sequence_length: int = 4) -> LandmarkSequence:
    coordinates = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT, 3), dtype=np.float32)
    confidence = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT), dtype=np.float32)
    detected = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT), dtype=np.bool_)
    valid_timesteps = np.array([True, True, False, False], dtype=np.bool_)
    timestamps_ms = np.array([0, 40, 0, 0], dtype=np.int64)

    coordinates[:2, 0] = (0.1, 0.2, 0.3)
    confidence[:2, 0] = 1.0
    detected[:2, 0] = True

    return LandmarkSequence(
        coordinates=coordinates,
        confidence=confidence,
        detected=detected,
        valid_timesteps=valid_timesteps,
        timestamps_ms=timestamps_ms,
        label_id=22,
        relative_video_path="Greetings/48. Hello/MVI_0089.MOV",
    )


def test_sequence_produces_expected_flat_model_dimension() -> None:
    sequence = make_sequence()

    coordinate_only = sequence.model_features(include_confidence=False)
    with_confidence = sequence.model_features(include_confidence=True)

    assert coordinate_only.shape == (4, TOTAL_LANDMARK_COUNT * 3)
    assert with_confidence.shape == (4, TOTAL_LANDMARK_COUNT * 4)
    assert sequence.valid_frame_count == 2


def test_sequence_rejects_detections_in_padding() -> None:
    sequence = make_sequence()
    detected = sequence.detected.copy()
    detected[2, 0] = True

    with pytest.raises(ValueError, match="padded timesteps"):
        LandmarkSequence(
            coordinates=sequence.coordinates,
            confidence=sequence.confidence,
            detected=detected,
            valid_timesteps=sequence.valid_timesteps,
            timestamps_ms=sequence.timestamps_ms,
            label_id=sequence.label_id,
            relative_video_path=sequence.relative_video_path,
        )


def test_sequence_saves_versioned_npz(tmp_path: Path) -> None:
    sequence = make_sequence()
    output = sequence.save(tmp_path / "hello.npz")

    with np.load(output) as saved:
        assert saved["coordinates"].shape == (4, TOTAL_LANDMARK_COUNT, 3)
        assert saved["schema_version"].item() == SCHEMA_VERSION
        assert saved["label_id"].item() == 22
        assert saved["relative_video_path"].item().endswith("MVI_0089.MOV")
        assert saved["landmark_names"].shape == (TOTAL_LANDMARK_COUNT,)
