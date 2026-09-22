"""Load trusted PyTorch checkpoints and classify landmark sequences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from isl_assistant.models.temporal import TemporalClassifier


@dataclass(frozen=True, slots=True)
class Prediction:
    """One ranked class returned by a temporal classifier."""

    label_id: int
    label: str
    probability: float


def _resolve_device(value: str) -> torch.device:
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def temporal_features(arrays: Mapping[str, Any]) -> tuple[NDArray[np.float32], int]:
    """Build the exact 255-value training representation from landmark arrays."""

    coordinates = np.asarray(arrays["coordinates"], dtype=np.float32)
    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
    detected = np.asarray(arrays["detected"], dtype=np.float32)
    valid = np.asarray(arrays["valid_timesteps"], dtype=np.bool_)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (51, 3):
        raise ValueError(f"coordinates must have shape [timesteps, 51, 3], got {coordinates.shape}")
    expected = coordinates.shape[:2]
    if confidence.shape != expected or detected.shape != expected:
        raise ValueError("confidence and detected must match coordinate timesteps and landmarks")
    if valid.shape != (coordinates.shape[0],):
        raise ValueError("valid_timesteps must contain one value per timestep")
    length = int(valid.sum())
    if length <= 0 or not np.all(valid[:length]) or np.any(valid[length:]):
        raise ValueError("valid timesteps must be one non-empty contiguous prefix")
    features = np.concatenate(
        (coordinates.reshape(coordinates.shape[0], -1), confidence, detected),
        axis=1,
        dtype=np.float32,
    )
    return features, length


class TemporalPredictor:
    """Inference-only wrapper around a trained temporal checkpoint."""

    def __init__(
        self,
        model: TemporalClassifier,
        mean: NDArray[np.float32],
        standard_deviation: NDArray[np.float32],
        label_names: Mapping[int, str],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.mean = np.asarray(mean, dtype=np.float32)
        self.standard_deviation = np.asarray(standard_deviation, dtype=np.float32)
        self.label_names = dict(label_names)
        self.input_size = int(self.mean.shape[0])
        self.maximum_sequence_length = model.maximum_sequence_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> TemporalPredictor:
        """Load a trusted project checkpoint and reconstruct its model."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        target_device = _resolve_device(device)
        # Project checkpoints contain NumPy normalization arrays. They are
        # trusted local artifacts and require full deserialization.
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a metadata dictionary")
        required = {
            "architecture",
            "input_size",
            "hidden_size",
            "number_of_classes",
            "dropout",
            "maximum_sequence_length",
            "transformer_heads",
            "transformer_layers",
            "transformer_feedforward_size",
            "state_dict",
            "normalization_mean",
            "normalization_standard_deviation",
            "label_names",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")

        input_size = int(checkpoint["input_size"])
        class_count = int(checkpoint["number_of_classes"])
        labels = {int(key): str(value) for key, value in checkpoint["label_names"].items()}
        if set(labels) != set(range(class_count)):
            raise ValueError("checkpoint label IDs must be contiguous from zero")
        mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        deviation = np.asarray(
            checkpoint["normalization_standard_deviation"], dtype=np.float32
        )
        if mean.shape != (input_size,) or deviation.shape != (input_size,):
            raise ValueError("checkpoint normalization shape does not match input size")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(deviation)):
            raise ValueError("checkpoint normalization contains non-finite values")
        if np.any(deviation <= 0):
            raise ValueError("checkpoint standard deviations must be positive")

        model = TemporalClassifier(
            architecture=str(checkpoint["architecture"]),
            input_size=input_size,
            hidden_size=int(checkpoint["hidden_size"]),
            number_of_classes=class_count,
            dropout=float(checkpoint["dropout"]),
            maximum_sequence_length=int(checkpoint["maximum_sequence_length"]),
            transformer_heads=int(checkpoint["transformer_heads"]),
            transformer_layers=int(checkpoint["transformer_layers"]),
            transformer_feedforward_size=int(checkpoint["transformer_feedforward_size"]),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return cls(model, mean, deviation, labels, target_device)

    def predict(
        self,
        sequence: NDArray[np.floating[Any]],
        length: int,
        *,
        top_k: int = 5,
    ) -> list[Prediction]:
        """Return top-k predictions for one padded temporal feature sequence."""

        features = np.asarray(sequence, dtype=np.float32)
        expected = (self.maximum_sequence_length, self.input_size)
        if features.shape != expected:
            raise ValueError(f"sequence must have shape {expected}, got {features.shape}")
        if not 1 <= length <= self.maximum_sequence_length:
            raise ValueError("length must identify at least one real frame within the sequence")
        if not np.all(np.isfinite(features[:length])):
            raise ValueError("real sequence frames contain non-finite values")
        if not 1 <= top_k <= len(self.label_names):
            raise ValueError("top_k must be between one and the number of classes")

        normalized = np.zeros_like(features, dtype=np.float32)
        normalized[:length] = (
            features[:length] - self.mean
        ) / self.standard_deviation
        sequence_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([length], dtype=torch.int64)
        with torch.inference_mode():
            logits = self.model(sequence_tensor, length_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probabilities, k=top_k)
        return [
            Prediction(
                label_id=int(label_id),
                label=self.label_names[int(label_id)],
                probability=float(probability),
            )
            for probability, label_id in zip(
                values.detach().cpu().tolist(),
                indices.detach().cpu().tolist(),
                strict=True,
            )
        ]
