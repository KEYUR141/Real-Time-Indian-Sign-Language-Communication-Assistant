import numpy as np
import pytest

from isl_assistant.vision.landmarks.temporal_sampling import (
    build_sampling_plan,
    pad_feature_sequence,
    uniform_frame_indices,
)


def test_long_video_is_sampled_to_64_unique_frames() -> None:
    plan = build_sampling_plan(total_frames=200, target_length=64)

    assert plan.sampled_frame_count == 64
    assert plan.source_indices[0] == 0
    assert plan.source_indices[-1] == 199
    assert len(set(plan.source_indices)) == 64
    assert plan.source_indices == tuple(sorted(plan.source_indices))
    assert not plan.requires_padding


def test_short_video_keeps_every_frame_and_records_padding() -> None:
    plan = build_sampling_plan(total_frames=40, target_length=64)

    assert plan.source_indices == tuple(range(40))
    assert plan.sampled_frame_count == 40
    assert plan.padding_count == 24
    assert plan.requires_padding
    assert plan.valid_timestep_mask.tolist() == [True] * 40 + [False] * 24


def test_exact_length_video_requires_no_sampling_or_padding() -> None:
    plan = build_sampling_plan(total_frames=64, target_length=64)

    assert plan.source_indices == tuple(range(64))
    assert plan.padding_count == 0


def test_frame_indices_convert_to_millisecond_timestamps() -> None:
    plan = build_sampling_plan(total_frames=3, target_length=64)

    assert plan.timestamps_ms(25.0) == (0, 40, 80)


def test_padding_preserves_features_and_returns_mask() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    padded, mask = pad_feature_sequence(features, target_length=4)

    np.testing.assert_array_equal(padded[:2], features)
    np.testing.assert_array_equal(padded[2:], np.zeros((2, 2), dtype=np.float32))
    assert mask.tolist() == [True, True, False, False]


@pytest.mark.parametrize(
    ("total_frames", "maximum_frames"),
    [(0, 64), (-1, 64), (10, 0), (10, -1)],
)
def test_invalid_sampling_dimensions_are_rejected(
    total_frames: int,
    maximum_frames: int,
) -> None:
    with pytest.raises(ValueError):
        uniform_frame_indices(total_frames, maximum_frames)


def test_padding_rejects_a_sequence_longer_than_target() -> None:
    features = np.ones((65, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="longer than target_length"):
        pad_feature_sequence(features, target_length=64)
