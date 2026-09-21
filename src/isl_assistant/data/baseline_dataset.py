"""Load manifest-selected archives as SVM summary-feature matrices."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .baseline_features import sequence_to_summary_features
from .quality_report import ExpectedFeature
from .schemas import DatasetSplit


def load_summary_matrix(
    output_root: str | Path,
    expected: Iterable[ExpectedFeature],
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, list[ExpectedFeature]]:
    """Load one official split in manifest order, excluding pilot artifacts."""

    root = Path(output_root).expanduser().resolve()
    selected = [item for item in expected if item.split is split]
    if not selected:
        raise ValueError(f"no expected samples for split: {split.value}")

    rows: list[np.ndarray] = []
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
            rows.append(
                sequence_to_summary_features(
                    archive["coordinates"],
                    archive["confidence"],
                    archive["detected"],
                    archive["valid_timesteps"],
                )
            )
            labels.append(item.label_id)

    return np.stack(rows), np.asarray(labels, dtype=np.int64), selected
