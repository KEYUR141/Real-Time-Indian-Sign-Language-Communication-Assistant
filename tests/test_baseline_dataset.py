from pathlib import Path, PurePosixPath

import numpy as np

from isl_assistant.data.baseline_dataset import load_summary_matrix
from isl_assistant.data.quality_report import ExpectedFeature
from isl_assistant.data.schemas import DatasetSplit


def test_load_summary_matrix_uses_only_requested_manifest_split(tmp_path: Path) -> None:
    item = ExpectedFeature(
        DatasetSplit.TRAIN,
        "loud",
        0,
        PurePosixPath("Adjectives/1. loud/MVI_0001.MOV"),
    )
    path = item.feature_path(tmp_path)
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        coordinates=np.zeros((2, 51, 3), dtype=np.float32),
        confidence=np.zeros((2, 51), dtype=np.float32),
        detected=np.zeros((2, 51), dtype=np.bool_),
        valid_timesteps=np.ones(2, dtype=np.bool_),
        label=np.asarray("loud"),
        label_id=np.asarray(0),
        split=np.asarray("train"),
    )

    features, labels, items = load_summary_matrix(tmp_path, [item], DatasetSplit.TRAIN)

    assert features.shape == (1, 1377)
    assert labels.tolist() == [0]
    assert items == [item]
