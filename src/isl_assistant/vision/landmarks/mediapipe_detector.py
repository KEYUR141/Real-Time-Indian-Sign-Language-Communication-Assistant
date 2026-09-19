"""MediaPipe Tasks adapter for the project's landmark feature schema.

One detector instance is intended to process one video because MediaPipe VIDEO
mode requires timestamps to increase monotonically. Create a fresh context for
each source video whose timestamps restart at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from .feature_schema import (
    HAND_LANDMARK_COUNT,
    LEFT_HAND_SLICE,
    POSE_LANDMARK_INDICES,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    TOTAL_LANDMARK_COUNT,
    FrameLandmarks,
)


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    hand_model_path: Path
    pose_model_path: Path
    maximum_hands: int = 2
    minimum_detection_confidence: float = 0.5
    minimum_presence_confidence: float = 0.5
    minimum_tracking_confidence: float = 0.5
    swap_handedness_labels: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand_model_path", Path(self.hand_model_path))
        object.__setattr__(self, "pose_model_path", Path(self.pose_model_path))

        if self.maximum_hands not in {1, 2}:
            raise ValueError("maximum_hands must be either 1 or 2")

        for name in (
            "minimum_detection_confidence",
            "minimum_presence_confidence",
            "minimum_tracking_confidence",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


class MediaPipeLandmarkDetector:
    """Run hand and pose landmark detection in MediaPipe VIDEO mode."""

    def __init__(self, settings: DetectorSettings) -> None:
        self.settings = settings
        self._hand_landmarker: Any | None = None
        self._pose_landmarker: Any | None = None
        self._last_timestamp_ms: int | None = None

    def open(self) -> None:
        if self._hand_landmarker is not None or self._pose_landmarker is not None:
            raise RuntimeError("detector is already open")

        self._validate_model_file(self.settings.hand_model_path, "hand")
        self._validate_model_file(self.settings.pose_model_path, "pose")

        vision = mp.tasks.vision
        running_mode = vision.RunningMode.VIDEO

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.settings.hand_model_path.resolve())
            ),
            running_mode=running_mode,
            num_hands=self.settings.maximum_hands,
            min_hand_detection_confidence=self.settings.minimum_detection_confidence,
            min_hand_presence_confidence=self.settings.minimum_presence_confidence,
            min_tracking_confidence=self.settings.minimum_tracking_confidence,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.settings.pose_model_path.resolve())
            ),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=self.settings.minimum_detection_confidence,
            min_pose_presence_confidence=self.settings.minimum_presence_confidence,
            min_tracking_confidence=self.settings.minimum_tracking_confidence,
            output_segmentation_masks=False,
        )

        try:
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        except Exception:
            self.close()
            raise

        self._last_timestamp_ms = None

    def close(self) -> None:
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        self._last_timestamp_ms = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def detect_rgb(
        self,
        rgb_frame: NDArray[np.uint8],
        timestamp_ms: int,
    ) -> FrameLandmarks:
        """Detect hands and pose in one RGB uint8 video frame."""

        if self._hand_landmarker is None or self._pose_landmarker is None:
            raise RuntimeError("detector must be opened before calling detect_rgb")

        frame = np.asarray(rgb_frame)
        if frame.dtype != np.uint8:
            raise ValueError("rgb_frame must use uint8 pixels")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("rgb_frame must have shape [height, width, 3]")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("rgb_frame dimensions cannot be empty")
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamps must be strictly increasing within a video")

        contiguous_frame = np.ascontiguousarray(frame)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_frame)

        hand_result = self._hand_landmarker.detect_for_video(media_image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(media_image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        return frame_from_mediapipe_results(
            hand_result=hand_result,
            pose_result=pose_result,
            timestamp_ms=timestamp_ms,
            swap_handedness_labels=self.settings.swap_handedness_labels,
        )

    @staticmethod
    def _validate_model_file(path: Path, model_name: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{model_name} landmarker model does not exist: {path}")
        if path.suffix.casefold() != ".task":
            raise ValueError(f"{model_name} landmarker model must be a .task file: {path}")


def frame_from_mediapipe_results(
    hand_result: Any,
    pose_result: Any,
    timestamp_ms: int,
    *,
    swap_handedness_labels: bool = False,
) -> FrameLandmarks:
    """Convert MediaPipe result objects into the stable 51-landmark schema."""

    coordinates = np.zeros((TOTAL_LANDMARK_COUNT, 3), dtype=np.float32)
    confidence = np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.float32)
    detected = np.zeros(TOTAL_LANDMARK_COUNT, dtype=np.bool_)

    _write_hands(
        result=hand_result,
        coordinates=coordinates,
        confidence=confidence,
        detected=detected,
        swap_handedness_labels=swap_handedness_labels,
    )
    _write_pose(
        result=pose_result,
        coordinates=coordinates,
        confidence=confidence,
        detected=detected,
    )

    return FrameLandmarks(
        coordinates=coordinates,
        confidence=confidence,
        detected=detected,
        timestamp_ms=timestamp_ms,
    )


def _write_hands(
    result: Any,
    coordinates: NDArray[np.float32],
    confidence: NDArray[np.float32],
    detected: NDArray[np.bool_],
    *,
    swap_handedness_labels: bool,
) -> None:
    hand_landmarks = list(getattr(result, "hand_landmarks", []) or [])
    handedness_groups = list(getattr(result, "handedness", []) or [])

    if len(hand_landmarks) != len(handedness_groups):
        raise ValueError("MediaPipe returned different hand-landmark and handedness counts")

    best_score_by_side: dict[str, float] = {"left": -1.0, "right": -1.0}

    for landmarks, categories in zip(hand_landmarks, handedness_groups, strict=True):
        if len(landmarks) != HAND_LANDMARK_COUNT:
            raise ValueError(
                f"MediaPipe hand result must contain {HAND_LANDMARK_COUNT} landmarks"
            )
        if not categories:
            continue

        category = categories[0]
        side = _category_label(category).casefold()
        if side not in {"left", "right"}:
            continue
        if swap_handedness_labels:
            side = "right" if side == "left" else "left"

        score = _unit_interval(float(getattr(category, "score", 0.0)))
        if score <= best_score_by_side[side]:
            continue

        target = LEFT_HAND_SLICE if side == "left" else RIGHT_HAND_SLICE
        coordinates[target] = np.asarray(
            [(point.x, point.y, point.z) for point in landmarks], dtype=np.float32
        )
        confidence[target] = score
        detected[target] = True
        best_score_by_side[side] = score


def _write_pose(
    result: Any,
    coordinates: NDArray[np.float32],
    confidence: NDArray[np.float32],
    detected: NDArray[np.bool_],
) -> None:
    poses = list(getattr(result, "pose_landmarks", []) or [])
    if not poses:
        return

    pose = poses[0]
    if len(pose) <= max(POSE_LANDMARK_INDICES):
        raise ValueError("MediaPipe pose result does not contain all required landmarks")

    for output_offset, source_index in enumerate(POSE_LANDMARK_INDICES):
        point = pose[source_index]
        output_index = POSE_SLICE.start + output_offset
        visibility = _unit_interval(float(getattr(point, "visibility", 1.0)))
        presence = _unit_interval(float(getattr(point, "presence", 1.0)))

        coordinates[output_index] = (point.x, point.y, point.z)
        confidence[output_index] = min(visibility, presence)
        detected[output_index] = True


def _category_label(category: Any) -> str:
    for attribute in ("category_name", "display_name"):
        value = getattr(category, attribute, None)
        if value:
            return str(value)
    return ""


def _unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))

