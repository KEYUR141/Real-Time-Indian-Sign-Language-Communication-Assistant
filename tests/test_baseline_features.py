import numpy as np
import pytest

from isl_assistant.data.baseline_features import sequence_to_summary_features


def test_summary_features_ignore_right_padding() -> None:
    coordinates = np.zeros((4, 51, 3), dtype=np.float32)
    coordinates[0, :, 0] = 1.0
    coordinates[1, :, 0] = 3.0
    confidence = np.ones((4, 51), dtype=np.float32)
    detected = np.ones((4, 51), dtype=np.bool_)
    valid = np.array([True, True, False, False])

    features = sequence_to_summary_features(coordinates, confidence, detected, valid)

    assert features.dtype == np.float32
    assert features.shape == (1377,)
    np.testing.assert_allclose(features[:3], [2.0, 0.0, 0.0])


def test_summary_features_rejects_empty_real_sequence() -> None:
    with pytest.raises(ValueError, match="valid timestep"):
        sequence_to_summary_features(
            np.zeros((2, 51, 3), dtype=np.float32),
            np.zeros((2, 51), dtype=np.float32),
            np.zeros((2, 51), dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
