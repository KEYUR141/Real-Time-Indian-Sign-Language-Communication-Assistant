"""Temporal frame sampling for variable-length sign videos.

This module creates deterministic sampling plans without reading video files.
Long videos are sampled uniformly. Short videos retain every real frame and
are padded only after landmark extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FrameSamplingPlan:
    """Mapping from source-video frames to a fixed-length model sequence."""

    total_source_frames: int
    target_length: int
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.total_source_frames <= 0:
            raise ValueError("total_source_frames must be greater than zero")
        if self.target_length <= 0:
            raise ValueError("target_length must be greater than zero")
        if not self.source_indices:
            raise ValueError("source_indices cannot be empty")
        if tuple(sorted(set(self.source_indices))) != self.source_indices:
            raise ValueError("source_indices must be unique and strictly increasing")
        if self.source_indices[0] < 0:
            raise ValueError("source frame indices cannot be negative")
        if self.source_indices[-1] >= self.total_source_frames:
            raise ValueError("source frame index exceeds the source video length")
        if len(self.source_indices) > self.target_length:
            raise ValueError("a sampling plan cannot exceed its target length")

    @property
    def sampled_frame_count(self) -> int:
        return len(self.source_indices)

    @property
    def padding_count(self) -> int:
        return self.target_length - self.sampled_frame_count

    @property
    def requires_padding(self) -> bool:
        return self.padding_count > 0

    @property
    def valid_timestep_mask(self) -> NDArray[np.bool_]:
        """Identify real sampled frames and right-side padding."""

        mask = np.zeros(self.target_length, dtype=np.bool_)
        mask[: self.sampled_frame_count] = True
        return mask

    def timestamps_ms(self, frames_per_second: float) -> tuple[int, ...]:
        """Convert source indices into MediaPipe-compatible timestamps."""

        if not np.isfinite(frames_per_second) or frames_per_second <= 0:
            raise ValueError("frames_per_second must be a positive finite number")

        return tuple(round(index * 1000.0 / frames_per_second) for index in self.source_indices)


def uniform_frame_indices(total_frames: int, maximum_frames: int) -> tuple[int, ...]:
    """Select unique frames distributed across the complete source video."""

    if total_frames <= 0:
        raise ValueError("total_frames must be greater than zero")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be greater than zero")

    if total_frames <= maximum_frames:
        return tuple(range(total_frames))

    positions = np.linspace(0, total_frames - 1, num=maximum_frames)
    indices = tuple(int(index) for index in np.rint(positions))

    if len(set(indices)) != maximum_frames:
        raise RuntimeError("uniform sampling unexpectedly produced duplicate indices")

    return indices


def build_sampling_plan(total_frames: int, target_length: int = 64) -> FrameSamplingPlan:
    """Build the standard plan used by offline landmark extraction."""

    return FrameSamplingPlan(
        total_source_frames=total_frames,
        target_length=target_length,
        source_indices=uniform_frame_indices(total_frames, target_length),
    )


def pad_feature_sequence(
    features: NDArray[np.floating],
    target_length: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Right-pad features and return a valid-timestep mask."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("features must have a time axis and at least one feature axis")
    if values.shape[0] == 0:
        raise ValueError("features cannot contain an empty time axis")
    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")
    if values.shape[0] > target_length:
        raise ValueError("features are longer than target_length; sample before padding")

    padded = np.zeros((target_length, *values.shape[1:]), dtype=np.float32)
    padded[: values.shape[0]] = values

    valid_mask = np.zeros(target_length, dtype=np.bool_)
    valid_mask[: values.shape[0]] = True
    return padded, valid_mask

