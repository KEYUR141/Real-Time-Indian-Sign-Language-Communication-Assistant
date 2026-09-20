"""Reusable video-to-landmark extraction for manifest-selected samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from isl_assistant.data.schemas import QualityMetrics, VideoSample

from .feature_schema import SCHEMA_VERSION, TOTAL_LANDMARK_COUNT, FrameLandmarks
from .normalization import normalize_sequence
from .temporal_sampling import build_sampling_plan


class LandmarkDetector(Protocol):
    def detect_rgb(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameLandmarks: ...


@dataclass(slots=True)
class MediaPipeTimestampClock:
    """Map video-relative times onto one strictly increasing detector timeline."""

    next_timestamp_ms: int = 0

    def allocate(self, source_timestamps_ms: tuple[int, ...]) -> tuple[int, ...]:
        if not source_timestamps_ms:
            return ()
        if source_timestamps_ms[0] < 0 or any(
            later <= earlier
            for earlier, later in zip(
                source_timestamps_ms, source_timestamps_ms[1:], strict=False
            )
        ):
            raise ValueError("source timestamps must be non-negative and strictly increasing")

        offset = self.next_timestamp_ms - source_timestamps_ms[0]
        allocated = tuple(timestamp + offset for timestamp in source_timestamps_ms)
        self.next_timestamp_ms = allocated[-1] + 1
        return allocated


@dataclass(frozen=True, slots=True)
class ExtractedVideo:
    arrays: dict[str, np.ndarray]
    quality: QualityMetrics


def extract_video(
    video_path: str | Path,
    detector: LandmarkDetector,
    sequence_length: int,
    timestamp_clock: MediaPipeTimestampClock,
) -> ExtractedVideo:
    """Extract a fixed-length sequence while allowing detector reuse."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("Video reports no frames")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    plan = build_sampling_plan(frame_count, sequence_length)
    source_timestamps = plan.timestamps_ms(fps)
    detector_timestamps = timestamp_clock.allocate(source_timestamps)
    selected_positions = {
        source_index: output_index
        for output_index, source_index in enumerate(plan.source_indices)
    }
    raw = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT, 3), dtype=np.float32)
    confidence = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT), dtype=np.float32)
    detected = np.zeros((sequence_length, TOTAL_LANDMARK_COUNT), dtype=np.bool_)

    decoded = 0
    source_index = 0
    try:
        while decoded < plan.sampled_frame_count:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            output_index = selected_positions.get(source_index)
            if output_index is not None:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                frame = detector.detect_rgb(rgb_frame, detector_timestamps[output_index])
                raw[output_index] = frame.coordinates
                confidence[output_index] = frame.confidence
                detected[output_index] = frame.detected
                decoded += 1
            source_index += 1
    finally:
        capture.release()

    if decoded != plan.sampled_frame_count:
        raise RuntimeError(f"Decoded {decoded}/{plan.sampled_frame_count} selected frames")

    valid_timesteps = plan.valid_timestep_mask
    normalized, normalization_valid = normalize_sequence(raw, detected, valid_timesteps)
    timestamps_ms = np.full(sequence_length, -1, dtype=np.int64)
    timestamps_ms[: plan.sampled_frame_count] = source_timestamps
    source_indices = np.full(sequence_length, -1, dtype=np.int64)
    source_indices[: plan.sampled_frame_count] = plan.source_indices

    left = int(detected[valid_timesteps, :21].any(axis=1).sum())
    right = int(detected[valid_timesteps, 21:42].any(axis=1).sum())
    pose = int(detected[valid_timesteps, 42:].any(axis=1).sum())
    arrays = {
        "coordinates": normalized,
        "raw_coordinates": raw,
        "confidence": confidence,
        "detected": detected,
        "valid_timesteps": valid_timesteps,
        "normalization_valid": normalization_valid,
        "timestamps_ms": timestamps_ms,
        "source_frame_indices": source_indices,
        "source_frame_count": np.asarray(frame_count, dtype=np.int64),
        "source_fps": np.asarray(fps, dtype=np.float32),
    }
    return ExtractedVideo(
        arrays=arrays,
        quality=QualityMetrics(frame_count, plan.sampled_frame_count, left, right, pose),
    )


def feature_path(output_root: str | Path, sample: VideoSample) -> Path:
    """Return a collision-free path organized by official split and class."""

    return Path(output_root) / sample.split.value / sample.label / f"{sample.sample_id}.npz"


def save_extracted_video(
    output_path: str | Path,
    extracted: ExtractedVideo,
    sample: VideoSample,
) -> Path:
    """Atomically save features and complete manifest provenance."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **extracted.arrays,
            label=np.asarray(sample.label),
            label_id=np.asarray(sample.label_id, dtype=np.int64),
            split=np.asarray(sample.split.value),
            source_path=np.asarray(str(sample.absolute_path)),
            relative_video_path=np.asarray(sample.relative_path.as_posix()),
            schema_version=np.asarray(SCHEMA_VERSION),
        )
    temporary.replace(destination)
    return destination
