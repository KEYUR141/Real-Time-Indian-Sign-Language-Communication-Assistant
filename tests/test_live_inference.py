import numpy as np
import pytest

from isl_assistant.inference import (
    CandidateDecision,
    IsolatedLandmarkCapture,
    RejectionConfig,
    RollingLandmarkBuffer,
    TemporalStabilizer,
    UnknownRejector,
)
from isl_assistant.inference.temporal_predictor import Prediction
from isl_assistant.vision.landmarks.feature_schema import FrameLandmarks


def frame(timestamp: int, *, hands: bool = True, pose: bool = True) -> FrameLandmarks:
    coordinates = np.zeros((51, 3), dtype=np.float32)
    confidence = np.zeros(51, dtype=np.float32)
    detected = np.zeros(51, dtype=np.bool_)
    if hands:
        detected[:21] = True
        confidence[:21] = 0.9
        coordinates[:21, 0] = np.linspace(0.1, 0.3, 21)
    if pose:
        detected[42:] = True
        confidence[42:] = 0.9
        coordinates[43] = (0.4, 0.5, 0.0)
        coordinates[44] = (0.6, 0.5, 0.0)
    return FrameLandmarks(coordinates, confidence, detected, timestamp)


def test_rolling_buffer_pads_then_discards_oldest_frame() -> None:
    buffer = RollingLandmarkBuffer(maximum_length=3)
    buffer.append(frame(10))
    buffer.append(frame(20))

    partial = buffer.snapshot()
    assert partial.length == 2
    assert partial.timestamps_ms.tolist() == [10, 20, -1]
    assert partial.model_input()[0].shape == (3, 255)

    buffer.append(frame(30))
    buffer.append(frame(40))
    full = buffer.snapshot()
    assert buffer.is_full
    assert full.timestamps_ms.tolist() == [20, 30, 40]


def test_rolling_buffer_rejects_non_increasing_timestamp() -> None:
    buffer = RollingLandmarkBuffer(maximum_length=3)
    buffer.append(frame(10))

    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(frame(10))


def test_isolated_capture_uniformly_samples_complete_sequence() -> None:
    capture = IsolatedLandmarkCapture(target_length=3, maximum_frames=6)
    for timestamp in (10, 20, 30, 40, 50, 60):
        capture.append(frame(timestamp))

    snapshot = capture.snapshot()

    assert snapshot.length == 3
    assert snapshot.timestamps_ms.tolist() == [10, 30, 60]


def test_isolated_capture_pads_short_sequence_and_resets() -> None:
    capture = IsolatedLandmarkCapture(target_length=4, maximum_frames=6)
    capture.append(frame(10))
    capture.append(frame(20))

    assert capture.snapshot().timestamps_ms.tolist() == [10, 20, -1, -1]
    capture.reset()
    assert len(capture) == 0
    assert capture.snapshot().length == 0


def test_unknown_rejector_checks_coverage_probability_and_margin() -> None:
    buffer = RollingLandmarkBuffer(maximum_length=2)
    buffer.append(frame(10))
    buffer.append(frame(20))
    snapshot = buffer.snapshot()
    rejector = UnknownRejector(
        RejectionConfig(
            minimum_frames=2,
            minimum_hand_coverage=0.5,
            minimum_normalization_coverage=0.5,
            minimum_probability=0.7,
            minimum_margin=0.2,
        )
    )

    accepted = rejector.evaluate(
        snapshot,
        [Prediction(1, "hat", 0.9), Prediction(2, "old", 0.05)],
    )
    ambiguous = rejector.evaluate(
        snapshot,
        [Prediction(1, "hat", 0.75), Prediction(2, "old", 0.65)],
    )

    assert accepted.label == "hat"
    assert accepted.reason == "candidate"
    assert ambiguous.label is None
    assert ambiguous.reason == "ambiguous"


def test_unknown_rejector_rejects_missing_hands_before_model_scores() -> None:
    buffer = RollingLandmarkBuffer(maximum_length=2)
    buffer.append(frame(10, hands=False))
    buffer.append(frame(20, hands=False))

    decision = UnknownRejector(RejectionConfig(minimum_frames=2)).evaluate(
        buffer.snapshot(),
        [Prediction(1, "hat", 0.99), Prediction(2, "old", 0.001)],
    )

    assert decision.reason == "insufficient_hand_coverage"


def test_stabilizer_emits_once_until_unknown_release() -> None:
    stabilizer = TemporalStabilizer(
        required_consecutive=2,
        cooldown_updates=0,
        release_unknown_updates=2,
    )
    hat = CandidateDecision("hat", 0.9, 0.8, "candidate")
    unknown = CandidateDecision(None, 0.2, 0.01, "low_probability")

    assert stabilizer.update(hat).emitted is None
    assert stabilizer.update(hat).emitted == "hat"
    assert stabilizer.update(hat).reason == "duplicate_suppressed"
    stabilizer.update(unknown)
    stabilizer.update(unknown)
    assert stabilizer.update(hat).emitted is None
    assert stabilizer.update(hat).emitted == "hat"
