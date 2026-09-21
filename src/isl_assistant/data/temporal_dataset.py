"""Load manifest-selected landmark sequences for temporal neural models."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .quality_report import ExpectedFeature
from .schemas import DatasetSplit


def load_temporal_arrays(
    output_root: str | Path,
    expected: Iterable[ExpectedFeature],
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ExpectedFeature]]:
    """Load one split as [samples, timesteps, 255] features and true lengths."""

    root = Path(output_root).expanduser().resolve()
    selected = [item for item in expected if item.split is split]
    if not selected:
        raise ValueError(f"no expected samples for split: {split.value}")

    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    labels: list[int] = []
    for item in selected:
        path = item.feature_path(root)
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest feature: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if str(np.asarray(archive["label"]).item()) != item.label:
                raise ValueError(f"label mismatch in {path}")
            if str(np.asarray(archive["split"]).item()) != split.value:
                raise ValueError(f"split mismatch in {path}")
            coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
            detected = np.asarray(archive["detected"], dtype=np.float32)
            valid = np.asarray(archive["valid_timesteps"], dtype=np.bool_)
            if coordinates.ndim != 3 or coordinates.shape[1:] != (51, 3):
                raise ValueError(f"invalid coordinate shape in {path}: {coordinates.shape}")
            if confidence.shape != coordinates.shape[:2] or detected.shape != coordinates.shape[:2]:
                raise ValueError(f"invalid confidence/detected shape in {path}")
            if valid.shape != (coordinates.shape[0],):
                raise ValueError(f"invalid valid-timestep shape in {path}")
            length = int(valid.sum())
            if not length:
                raise ValueError(f"no real frames in {path}")
            sequences.append(
                np.concatenate(
                    (coordinates.reshape(coordinates.shape[0], -1), confidence, detected),
                    axis=1,
                    dtype=np.float32,
                )
            )
            lengths.append(length)
            labels.append(item.label_id)

    return (
        np.stack(sequences),
        np.asarray(labels, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
        selected,
    )
