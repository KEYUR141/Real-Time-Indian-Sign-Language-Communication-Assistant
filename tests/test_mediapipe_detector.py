from types import SimpleNamespace

import numpy as np
import pytest

from isl_assistant.vision.landmarks.feature_schema import (
    LEFT_HAND_SLICE,
    POSE_LANDMARK_INDICES,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)
from isl_assistant.vision.landmarks.mediapipe_detector import (
    DetectorSettings,
    frame_from_mediapipe_results,
)


def point(
    x: float,
    y: float,
    z: float,
    *,
    visibility: float = 1.0,
    presence: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
        presence=presence,
    )


def hand(start: float) -> list[SimpleNamespace]:
    return [point(start + index, start + index + 0.1, -index) for index in range(21)]


def handedness(label: str, score: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(category_name=label, display_name=label, score=score)]


def test_maps_hands_by_handedness_instead_of_detection_order() -> None:
    right = hand(100.0)
    left = hand(200.0)
    hand_result = SimpleNamespace(
        hand_landmarks=[right, left],
        handedness=[handedness("Right", 0.8), handedness("Left", 0.9)],
    )
    pose_result = SimpleNamespace(pose_landmarks=[])

    frame = frame_from_mediapipe_results(hand_result, pose_result, timestamp_ms=40)

    np.testing.assert_allclose(frame.coordinates[LEFT_HAND_SLICE, 0], np.arange(200, 221))
    np.testing.assert_allclose(frame.coordinates[RIGHT_HAND_SLICE, 0], np.arange(100, 121))
    np.testing.assert_allclose(frame.confidence[LEFT_HAND_SLICE], 0.9)
    np.testing.assert_allclose(frame.confidence[RIGHT_HAND_SLICE], 0.8)
    assert frame.has_left_hand
    assert frame.has_right_hand


def test_can_swap_handedness_after_visual_validation() -> None:
    hand_result = SimpleNamespace(
        hand_landmarks=[hand(10.0)],
        handedness=[handedness("Right", 0.75)],
    )

    frame = frame_from_mediapipe_results(
        hand_result,
        SimpleNamespace(pose_landmarks=[]),
        timestamp_ms=0,
        swap_handedness_labels=True,
    )

    assert frame.has_left_hand
    assert not frame.has_right_hand


def test_selects_only_the_nine_configured_pose_points() -> None:
    pose = [
        point(index, index + 0.1, -index, visibility=0.8, presence=0.9)
        for index in range(33)
    ]

    frame = frame_from_mediapipe_results(
        SimpleNamespace(hand_landmarks=[], handedness=[]),
        SimpleNamespace(pose_landmarks=[pose]),
        timestamp_ms=80,
    )

    np.testing.assert_allclose(
        frame.coordinates[POSE_SLICE, 0],
        np.asarray(POSE_LANDMARK_INDICES),
    )
    np.testing.assert_allclose(frame.confidence[POSE_SLICE], 0.8)
    assert frame.has_pose


def test_missing_detections_remain_zero_and_masked() -> None:
    frame = frame_from_mediapipe_results(
        SimpleNamespace(hand_landmarks=[], handedness=[]),
        SimpleNamespace(pose_landmarks=[]),
        timestamp_ms=0,
    )

    assert not np.any(frame.coordinates)
    assert not np.any(frame.confidence)
    assert not np.any(frame.detected)


def test_duplicate_side_keeps_the_higher_confidence_hand() -> None:
    hand_result = SimpleNamespace(
        hand_landmarks=[hand(10.0), hand(20.0)],
        handedness=[handedness("Left", 0.4), handedness("Left", 0.9)],
    )

    frame = frame_from_mediapipe_results(
        hand_result,
        SimpleNamespace(pose_landmarks=[]),
        timestamp_ms=0,
    )

    np.testing.assert_allclose(frame.coordinates[LEFT_HAND_SLICE, 0], np.arange(20, 41))
    np.testing.assert_allclose(frame.confidence[LEFT_HAND_SLICE], 0.9)


def test_rejects_mismatched_hand_result_lengths() -> None:
    hand_result = SimpleNamespace(
        hand_landmarks=[hand(10.0)],
        handedness=[],
    )

    with pytest.raises(ValueError, match="different"):
        frame_from_mediapipe_results(
            hand_result,
            SimpleNamespace(pose_landmarks=[]),
            timestamp_ms=0,
        )


def test_detector_settings_validate_confidences(tmp_path) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        DetectorSettings(
            hand_model_path=tmp_path / "hand.task",
            pose_model_path=tmp_path / "pose.task",
            minimum_detection_confidence=1.1,
        )
