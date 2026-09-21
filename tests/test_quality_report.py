from pathlib import Path, PurePosixPath

import numpy as np

from isl_assistant.data.quality_report import (
    ExpectedFeature,
    build_quality_report,
    inspect_feature,
)
from isl_assistant.data.schemas import DatasetSplit


def expected_feature() -> ExpectedFeature:
    return ExpectedFeature(
        split=DatasetSplit.TRAIN,
        label="loud",
        label_id=0,
        relative_video_path=PurePosixPath("Adjectives/1. loud/MVI_0001.MOV"),
    )


def write_feature(path: Path, feature: ExpectedFeature) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detected = np.zeros((4, 51), dtype=np.bool_)
    detected[:2, :21] = True
    coordinates = np.zeros((4, 51, 3), dtype=np.float32)
    coordinates[:2, :21] = 1.0
    np.savez_compressed(
        path,
        coordinates=coordinates,
        raw_coordinates=coordinates,
        confidence=detected.astype(np.float32),
        detected=detected,
        valid_timesteps=np.array([True, True, False, False]),
        normalization_valid=np.array([True, True, False, False]),
        timestamps_ms=np.array([0, 40, -1, -1], dtype=np.int64),
        source_frame_indices=np.array([0, 1, -1, -1], dtype=np.int64),
        source_frame_count=np.asarray(2, dtype=np.int64),
        source_fps=np.asarray(25.0, dtype=np.float32),
        label=np.asarray(feature.label),
        label_id=np.asarray(feature.label_id, dtype=np.int64),
        split=np.asarray(feature.split.value),
        relative_video_path=np.asarray(feature.relative_video_path.as_posix()),
        schema_version=np.asarray("mediapipe-hands-pose-v1"),
    )


def test_inspect_feature_calculates_hand_and_padding_metrics(tmp_path: Path) -> None:
    feature = expected_feature()
    path = tmp_path / "sample.npz"
    write_feature(path, feature)

    result = inspect_feature(path, feature, sequence_length=4)

    assert result["real_timesteps"] == 2
    assert result["padded_timesteps"] == 2
    assert result["left_hand_coverage"] == 1.0
    assert result["right_hand_coverage"] == 0.0


def test_quality_report_separates_manifest_and_unexpected_files(tmp_path: Path) -> None:
    feature = expected_feature()
    write_feature(feature.feature_path(tmp_path), feature)
    write_feature(tmp_path / "pilot" / "loud" / "sample.npz", feature)

    report = build_quality_report(tmp_path, [feature], sequence_length=4)

    assert report["expected_manifest_samples"] == 1
    assert report["valid_manifest_features"] == 1
    assert report["missing_manifest_features"] == 0
    assert report["unexpected_features"] == 1
    assert report["quality"]["samples_without_hands"] == 0
