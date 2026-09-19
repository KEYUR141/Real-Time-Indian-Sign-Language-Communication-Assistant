"""Versioned landmark feature contract shared by extraction and inference.

The schema deliberately stores structured arrays rather than prematurely
flattening them. Training code can later choose how to combine coordinates,
confidence values, and detection masks without regenerating landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "mediapipe-hands-pose-v1"

HAND_LANDMARK_COUNT = 21

# Indices defined by MediaPipe's 33-point pose model. Only upper-body and torso
# anchors useful to isolated signing are retained in the first feature version.
POSE_LANDMARK_INDICES: tuple[int, ...] = (
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
)

POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

LEFT_HAND_SLICE = slice(0, HAND_LANDMARK_COUNT)
RIGHT_HAND_SLICE = slice(HAND_LANDMARK_COUNT, HAND_LANDMARK_COUNT * 2)
POSE_SLICE = slice(HAND_LANDMARK_COUNT * 2, HAND_LANDMARK_COUNT * 2 + len(POSE_LANDMARK_INDICES))

TOTAL_LANDMARK_COUNT = HAND_LANDMARK_COUNT * 2 + len(POSE_LANDMARK_INDICES)
COORDINATE_DIMENSION = 3


def landmark_names() -> tuple[str, ...]:
    """Return the canonical landmark ordering used in every saved sequence."""

    left_hand = tuple(f"left_hand_{index}" for index in range(HAND_LANDMARK_COUNT))
    right_hand = tuple(f"right_hand_{index}" for index in range(HAND_LANDMARK_COUNT))
    pose = tuple(f"pose_{name}" for name in POSE_LANDMARK_NAMES)
    return left_hand + right_hand + pose


LANDMARK_NAMES = landmark_names()


@dataclass(frozen=True, slots=True)
class FrameLandmarks:
    """MediaPipe features extracted from one real video frame."""

    coordinates: NDArray[np.float32]
    confidence: NDArray[np.float32]
    detected: NDArray[np.bool_]
    timestamp_ms: int

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        detected = np.asarray(self.detected, dtype=np.bool_)

        expected_coordinates = (TOTAL_LANDMARK_COUNT, COORDINATE_DIMENSION)
        expected_vector = (TOTAL_LANDMARK_COUNT,)

        if coordinates.shape != expected_coordinates:
            raise ValueError(
                f"coordinates must have shape {expected_coordinates}, got {coordinates.shape}"
            )
        if confidence.shape != expected_vector:
            raise ValueError(f"confidence must have shape {expected_vector}, got {confidence.shape}")
        if detected.shape != expected_vector:
            raise ValueError(f"detected must have shape {expected_vector}, got {detected.shape}")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must contain only finite values")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence values must be between zero and one")
        if np.any(coordinates[~detected] != 0.0):
            raise ValueError("undetected landmarks must have zero coordinates")
        if np.any(confidence[~detected] != 0.0):
            raise ValueError("undetected landmarks must have zero confidence")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "detected", detected)

    @property
    def has_left_hand(self) -> bool:
        return bool(np.all(self.detected[LEFT_HAND_SLICE]))

    @property
    def has_right_hand(self) -> bool:
        return bool(np.all(self.detected[RIGHT_HAND_SLICE]))

    @property
    def has_pose(self) -> bool:
        return bool(np.any(self.detected[POSE_SLICE]))


@dataclass(frozen=True, slots=True)
class LandmarkSequence:
    """Fixed-length landmark arrays ready to save or pass to a model loader."""

    coordinates: NDArray[np.float32]
    confidence: NDArray[np.float32]
    detected: NDArray[np.bool_]
    valid_timesteps: NDArray[np.bool_]
    timestamps_ms: NDArray[np.int64]
    label_id: int
    relative_video_path: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        detected = np.asarray(self.detected, dtype=np.bool_)
        valid_timesteps = np.asarray(self.valid_timesteps, dtype=np.bool_)
        timestamps_ms = np.asarray(self.timestamps_ms, dtype=np.int64)

        if coordinates.ndim != 3:
            raise ValueError("coordinates must have shape [time, landmarks, xyz]")

        sequence_length = coordinates.shape[0]
        expected_coordinates = (
            sequence_length,
            TOTAL_LANDMARK_COUNT,
            COORDINATE_DIMENSION,
        )
        expected_matrix = (sequence_length, TOTAL_LANDMARK_COUNT)
        expected_time = (sequence_length,)

        if coordinates.shape != expected_coordinates:
            raise ValueError(
                f"coordinates must have shape {expected_coordinates}, got {coordinates.shape}"
            )
        if confidence.shape != expected_matrix:
            raise ValueError(f"confidence must have shape {expected_matrix}, got {confidence.shape}")
        if detected.shape != expected_matrix:
            raise ValueError(f"detected must have shape {expected_matrix}, got {detected.shape}")
        if valid_timesteps.shape != expected_time:
            raise ValueError(
                f"valid_timesteps must have shape {expected_time}, got {valid_timesteps.shape}"
            )
        if timestamps_ms.shape != expected_time:
            raise ValueError(
                f"timestamps_ms must have shape {expected_time}, got {timestamps_ms.shape}"
            )
        if sequence_length == 0:
            raise ValueError("landmark sequence cannot be empty")
        if self.label_id < 0:
            raise ValueError("label_id cannot be negative")
        if not self.relative_video_path:
            raise ValueError("relative_video_path cannot be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must contain only finite values")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence values must be between zero and one")
        if np.any(coordinates[~detected] != 0.0):
            raise ValueError("undetected landmarks must have zero coordinates")
        if np.any(confidence[~detected] != 0.0):
            raise ValueError("undetected landmarks must have zero confidence")
        if np.any(detected[~valid_timesteps]):
            raise ValueError("padded timesteps cannot contain detected landmarks")
        if np.any(coordinates[~valid_timesteps] != 0.0):
            raise ValueError("padded timesteps must have zero coordinates")
        if np.any(confidence[~valid_timesteps] != 0.0):
            raise ValueError("padded timesteps must have zero confidence")

        valid_timestamps = timestamps_ms[valid_timesteps]
        if np.any(valid_timestamps < 0):
            raise ValueError("valid timestamps cannot be negative")
        if valid_timestamps.size > 1 and np.any(np.diff(valid_timestamps) <= 0):
            raise ValueError("valid timestamps must be strictly increasing")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "detected", detected)
        object.__setattr__(self, "valid_timesteps", valid_timesteps)
        object.__setattr__(self, "timestamps_ms", timestamps_ms)

    @property
    def sequence_length(self) -> int:
        return self.coordinates.shape[0]

    @property
    def valid_frame_count(self) -> int:
        return int(np.count_nonzero(self.valid_timesteps))

    def model_features(self, include_confidence: bool = True) -> NDArray[np.float32]:
        """Flatten each frame while preserving the stable landmark order."""

        flat_coordinates = self.coordinates.reshape(self.sequence_length, -1)
        if not include_confidence:
            return flat_coordinates
        return np.concatenate((flat_coordinates, self.confidence), axis=1, dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        """Save the sequence as one compressed, self-describing NPZ artifact."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            coordinates=self.coordinates,
            confidence=self.confidence,
            detected=self.detected,
            valid_timesteps=self.valid_timesteps,
            timestamps_ms=self.timestamps_ms,
            label_id=np.asarray(self.label_id, dtype=np.int64),
            relative_video_path=np.asarray(self.relative_video_path),
            schema_version=np.asarray(self.schema_version),
            landmark_names=np.asarray(LANDMARK_NAMES),
            pose_landmark_indices=np.asarray(POSE_LANDMARK_INDICES, dtype=np.int64),
        )
        return destination


def empty_frame(timestamp_ms: int) -> FrameLandmarks:
    """Construct a valid frame result when MediaPipe detects nothing."""

    return FrameLandmarks(
        coordinates=np.zeros(
            (TOTAL_LANDMARK_COUNT, COORDINATE_DIMENSION), dtype=np.float32
        ),
        confidence=np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.float32),
        detected=np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.bool_),
        timestamp_ms=timestamp_ms,
    )

