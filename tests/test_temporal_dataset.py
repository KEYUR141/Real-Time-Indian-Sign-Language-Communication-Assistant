from pathlib import Path, PurePosixPath

import numpy as np

from isl_assistant.data.quality_report import ExpectedFeature
from isl_assistant.data.schemas import DatasetSplit
from isl_assistant.data.temporal_dataset import load_temporal_arrays


def test_temporal_loader_combines_coordinates_confidence_and_detection(tmp_path: Path) -> None:
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
        coordinates=np.ones((2, 51, 3), dtype=np.float32),
        confidence=np.full((2, 51), 0.5, dtype=np.float32),
        detected=np.ones((2, 51), dtype=np.bool_),
        valid_timesteps=np.array([True, False]),
        label=np.asarray("loud"),
        label_id=np.asarray(0),
        split=np.asarray("train"),
    )

    sequences, labels, lengths, items = load_temporal_arrays(
        tmp_path, [item], DatasetSplit.TRAIN
    )

    assert sequences.shape == (1, 2, 255)
    assert labels.tolist() == [0]
    assert lengths.tolist() == [1]
    assert items == [item]
