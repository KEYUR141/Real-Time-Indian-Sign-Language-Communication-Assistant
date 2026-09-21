"""Typed data contracts used by dataset and landmark-processing code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class VideoSample:
    """One labelled video selected by an official dataset manifest."""

    relative_path: PurePosixPath
    absolute_path: Path
    label: str
    label_id: int
    split: DatasetSplit

    @property
    def sample_id(self) -> str:
        """Return a stable ID derived from the split and manifest path."""

        normalized = self.relative_path.with_suffix("").as_posix()
        safe_path = "__".join(normalized.split("/"))
        return f"{self.split.value}__{safe_path}"


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    total_frames: int
    sampled_frames: int
    frames_with_left_hand: int
    frames_with_right_hand: int
    frames_with_pose: int

    @staticmethod
    def _ratio(detected: int, total: int) -> float:
        return detected / total if total else 0.0

    @property
    def left_hand_coverage(self) -> float:
        return self._ratio(self.frames_with_left_hand, self.sampled_frames)

    @property
    def right_hand_coverage(self) -> float:
        return self._ratio(self.frames_with_right_hand, self.sampled_frames)

    @property
    def pose_coverage(self) -> float:
        return self._ratio(self.frames_with_pose, self.sampled_frames)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    sample: VideoSample
    feature_path: Path
    sequence_length: int
    feature_dimension: int
    quality: QualityMetrics

