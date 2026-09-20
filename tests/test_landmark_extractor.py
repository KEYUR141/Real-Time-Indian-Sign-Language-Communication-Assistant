from pathlib import Path, PurePosixPath

import numpy as np

from isl_assistant.data.schemas import DatasetSplit, QualityMetrics, VideoSample
from isl_assistant.vision.landmarks.extractor import (
    ExtractedVideo,
    MediaPipeTimestampClock,
    feature_path,
    save_extracted_video,
)


def sample(tmp_path: Path) -> VideoSample:
    return VideoSample(
        relative_path=PurePosixPath("Adjectives/1. loud/Extra/MVI_1234.MOV"),
        absolute_path=tmp_path / "MVI_1234.MP4",
        label="loud",
        label_id=7,
        split=DatasetSplit.TRAIN,
    )


def test_timestamp_clock_keeps_one_detector_timeline_monotonic() -> None:
    clock = MediaPipeTimestampClock()

    assert clock.allocate((0, 40, 80)) == (0, 40, 80)
    assert clock.allocate((0, 33, 67)) == (81, 114, 148)


def test_timestamp_clock_rejects_non_increasing_source_times() -> None:
    clock = MediaPipeTimestampClock()

    try:
        clock.allocate((0, 0))
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("expected duplicate timestamps to be rejected")


def test_feature_path_contains_split_class_and_stable_sample_id(tmp_path: Path) -> None:
    video_sample = sample(tmp_path)

    assert feature_path(tmp_path, video_sample) == (
        tmp_path
        / "train"
        / "loud"
        / "train__Adjectives__1. loud__Extra__MVI_1234.npz"
    )


def test_saved_archive_contains_manifest_metadata(tmp_path: Path) -> None:
    video_sample = sample(tmp_path)
    arrays = {
        "coordinates": np.zeros((2, 51, 3), dtype=np.float32),
        "raw_coordinates": np.zeros((2, 51, 3), dtype=np.float32),
        "confidence": np.zeros((2, 51), dtype=np.float32),
        "detected": np.zeros((2, 51), dtype=np.bool_),
        "valid_timesteps": np.ones(2, dtype=np.bool_),
        "normalization_valid": np.zeros(2, dtype=np.bool_),
        "timestamps_ms": np.array([0, 40], dtype=np.int64),
        "source_frame_indices": np.array([0, 1], dtype=np.int64),
        "source_frame_count": np.asarray(2, dtype=np.int64),
        "source_fps": np.asarray(25.0, dtype=np.float32),
    }
    extracted = ExtractedVideo(arrays, QualityMetrics(2, 2, 0, 0, 0))
    destination = tmp_path / "feature.npz"

    save_extracted_video(destination, extracted, video_sample)

    with np.load(destination, allow_pickle=False) as archive:
        assert archive["label"].item() == "loud"
        assert archive["label_id"].item() == 7
        assert archive["split"].item() == "train"
        assert archive["relative_video_path"].item().endswith("Extra/MVI_1234.MOV")
