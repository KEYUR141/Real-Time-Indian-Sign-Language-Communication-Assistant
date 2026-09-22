"""Fixed-length rolling landmark buffer for live temporal inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from isl_assistant.vision.landmarks.feature_schema import (
    LEFT_HAND_SLICE,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    TOTAL_LANDMARK_COUNT,
    FrameLandmarks,
)
from isl_assistant.vision.landmarks.normalization import normalize_sequence
from isl_assistant.vision.landmarks.temporal_sampling import uniform_frame_indices

from .temporal_predictor import temporal_features


@dataclass(frozen=True, slots=True)
class BufferSnapshot:
    """Immutable, padded view of the current live landmark window."""

    coordinates: NDArray[np.float32]
    confidence: NDArray[np.float32]
    detected: NDArray[np.bool_]
    valid_timesteps: NDArray[np.bool_]
    normalization_valid: NDArray[np.bool_]
    timestamps_ms: NDArray[np.int64]

    @property
    def length(self) -> int:
        return int(self.valid_timesteps.sum())

    def _coverage(self, landmark_slice: slice) -> float:
        if not self.length:
            return 0.0
        active = self.detected[: self.length, landmark_slice].any(axis=1)
        return float(active.mean())

    @property
    def left_hand_coverage(self) -> float:
        return self._coverage(LEFT_HAND_SLICE)

    @property
    def right_hand_coverage(self) -> float:
        return self._coverage(RIGHT_HAND_SLICE)

    @property
    def any_hand_coverage(self) -> float:
        if not self.length:
            return 0.0
        active = self.detected[: self.length, : RIGHT_HAND_SLICE.stop].any(axis=1)
        return float(active.mean())

    @property
    def pose_coverage(self) -> float:
        return self._coverage(POSE_SLICE)

    @property
    def normalization_coverage(self) -> float:
        if not self.length:
            return 0.0
        return float(self.normalization_valid[: self.length].mean())

    def model_input(self) -> tuple[NDArray[np.float32], int]:
        return temporal_features(
            {
                "coordinates": self.coordinates,
                "confidence": self.confidence,
                "detected": self.detected,
                "valid_timesteps": self.valid_timesteps,
            }
        )


class RollingLandmarkBuffer:
    """Keep the most recent frames and create training-compatible snapshots."""

    def __init__(self, maximum_length: int = 64) -> None:
        if maximum_length <= 0:
            raise ValueError("maximum_length must be positive")
        self.maximum_length = maximum_length
        self._frames: deque[FrameLandmarks] = deque(maxlen=maximum_length)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self) == self.maximum_length

    def append(self, frame: FrameLandmarks) -> None:
        if self._frames and frame.timestamp_ms <= self._frames[-1].timestamp_ms:
            raise ValueError("live frame timestamps must be strictly increasing")
        self._frames.append(frame)

    def reset(self) -> None:
        self._frames.clear()

    def snapshot(self) -> BufferSnapshot:
        return _snapshot_from_frames(list(self._frames), self.maximum_length)


class IsolatedLandmarkCapture:
    """Collect one complete sign and sample it like an extracted video clip."""

    def __init__(self, target_length: int = 64, maximum_frames: int = 600) -> None:
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        if maximum_frames < target_length:
            raise ValueError("maximum_frames cannot be smaller than target_length")
        self.target_length = target_length
        self.maximum_frames = maximum_frames
        self._frames: list[FrameLandmarks] = []

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self) == self.maximum_frames

    def append(self, frame: FrameLandmarks) -> None:
        if self.is_full:
            raise OverflowError("isolated landmark capture reached its frame limit")
        if self._frames and frame.timestamp_ms <= self._frames[-1].timestamp_ms:
            raise ValueError("capture frame timestamps must be strictly increasing")
        self._frames.append(frame)

    def reset(self) -> None:
        self._frames.clear()

    def snapshot(self) -> BufferSnapshot:
        if not self._frames:
            return _snapshot_from_frames([], self.target_length)
        indices = uniform_frame_indices(len(self._frames), self.target_length)
        sampled = [self._frames[index] for index in indices]
        return _snapshot_from_frames(sampled, self.target_length)


def _snapshot_from_frames(
    frames: list[FrameLandmarks],
    target_length: int,
) -> BufferSnapshot:
        length = len(frames)
        raw = np.zeros((target_length, TOTAL_LANDMARK_COUNT, 3), dtype=np.float32)
        confidence = np.zeros(
            (target_length, TOTAL_LANDMARK_COUNT), dtype=np.float32
        )
        detected = np.zeros(
            (target_length, TOTAL_LANDMARK_COUNT), dtype=np.bool_
        )
        timestamps = np.full(target_length, -1, dtype=np.int64)
        for index, frame in enumerate(frames):
            raw[index] = frame.coordinates
            confidence[index] = frame.confidence
            detected[index] = frame.detected
            timestamps[index] = frame.timestamp_ms
        valid = np.zeros(target_length, dtype=np.bool_)
        valid[:length] = True
        normalized, normalization_valid = normalize_sequence(raw, detected, valid)
        return BufferSnapshot(
            coordinates=normalized,
            confidence=confidence,
            detected=detected,
            valid_timesteps=valid,
            normalization_valid=normalization_valid,
            timestamps_ms=timestamps,
        )
