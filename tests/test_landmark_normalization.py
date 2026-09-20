import numpy as np

from isl_assistant.vision.landmarks.normalization import normalize_sequence


def test_normalization_centers_shoulders_and_scales_width() -> None:
    coordinates = np.zeros((1, 51, 3), dtype=np.float32)
    detected = np.zeros((1, 51), dtype=bool)
    valid = np.asarray([True])
    coordinates[0, 43] = [0.4, 0.5, 0.0]
    coordinates[0, 44] = [0.6, 0.5, 0.0]
    coordinates[0, 0] = [0.5, 0.3, 0.0]
    detected[0, [0, 43, 44]] = True

    normalized, normalization_valid = normalize_sequence(
        coordinates, detected, valid
    )

    assert normalization_valid.tolist() == [True]
    np.testing.assert_allclose(
    normalized[0, 43, :2], [-0.5, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        normalized[0, 44, :2], [0.5, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        normalized[0, 0, :2], [0.0, -1.0], atol=1e-6
    )


def test_missing_shoulders_produce_zero_features_and_invalid_mask() -> None:
    coordinates = np.ones((2, 51, 3), dtype=np.float32)
    detected = np.ones((2, 51), dtype=bool)
    detected[0, 43] = False
    valid = np.asarray([True, False])

    normalized, normalization_valid = normalize_sequence(
        coordinates, detected, valid
    )

    assert not normalization_valid.any()
    assert not normalized.any()
