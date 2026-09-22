from pathlib import Path

import numpy as np
import pytest
import torch

from isl_assistant.inference.temporal_predictor import (
    TemporalPredictor,
    temporal_features,
)
from isl_assistant.models.temporal import TemporalClassifier


def write_checkpoint(path: Path) -> None:
    torch.manual_seed(4)
    model = TemporalClassifier(
        architecture="transformer",
        input_size=8,
        hidden_size=16,
        number_of_classes=3,
        dropout=0.0,
        maximum_sequence_length=4,
        transformer_heads=4,
        transformer_layers=1,
        transformer_feedforward_size=32,
    )
    torch.save(
        {
            "architecture": "transformer",
            "input_size": 8,
            "hidden_size": 16,
            "number_of_classes": 3,
            "dropout": 0.0,
            "maximum_sequence_length": 4,
            "transformer_heads": 4,
            "transformer_layers": 1,
            "transformer_feedforward_size": 32,
            "state_dict": model.state_dict(),
            "normalization_mean": np.zeros(8, dtype=np.float32),
            "normalization_standard_deviation": np.ones(8, dtype=np.float32),
            "label_names": {0: "hello", 1: "thanks", 2: "yes"},
        },
        path,
    )


def test_loads_checkpoint_and_returns_ranked_predictions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)
    predictor = TemporalPredictor.from_checkpoint(checkpoint, device="cpu")

    predictions = predictor.predict(np.ones((4, 8), dtype=np.float32), 3, top_k=3)

    assert predictor.device.type == "cpu"
    assert {prediction.label for prediction in predictions} == {"hello", "thanks", "yes"}
    assert [prediction.probability for prediction in predictions] == sorted(
        (prediction.probability for prediction in predictions), reverse=True
    )
    assert sum(prediction.probability for prediction in predictions) == pytest.approx(1.0)


def test_predictor_rejects_wrong_sequence_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)
    predictor = TemporalPredictor.from_checkpoint(checkpoint, device="cpu")

    with pytest.raises(ValueError, match="sequence must have shape"):
        predictor.predict(np.zeros((3, 8), dtype=np.float32), 3)


def test_temporal_features_matches_training_representation() -> None:
    coordinates = np.ones((2, 51, 3), dtype=np.float32)
    confidence = np.full((2, 51), 0.5, dtype=np.float32)
    detected = np.ones((2, 51), dtype=np.bool_)

    sequence, length = temporal_features(
        {
            "coordinates": coordinates,
            "confidence": confidence,
            "detected": detected,
            "valid_timesteps": np.array([True, False]),
        }
    )

    assert sequence.shape == (2, 255)
    assert length == 1
    np.testing.assert_allclose(sequence[0, :153], 1.0)
    np.testing.assert_allclose(sequence[0, 153:204], 0.5)
    np.testing.assert_allclose(sequence[0, 204:], 1.0)


def test_temporal_features_requires_prefix_mask() -> None:
    with pytest.raises(ValueError, match="contiguous prefix"):
        temporal_features(
            {
                "coordinates": np.zeros((3, 51, 3), dtype=np.float32),
                "confidence": np.zeros((3, 51), dtype=np.float32),
                "detected": np.zeros((3, 51), dtype=np.bool_),
                "valid_timesteps": np.array([True, False, True]),
            }
        )

