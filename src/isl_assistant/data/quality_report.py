"""Manifest-aware quality metrics for extracted landmark archives."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .manifests import (
    canonicalize_label,
    exclude_and_reindex_labels,
    label_from_path,
    load_label_map,
    read_manifest,
)
from .schemas import DatasetSplit

SCHEMA_VERSION = "mediapipe-hands-pose-v1"
REQUIRED_KEYS = {
    "coordinates",
    "raw_coordinates",
    "confidence",
    "detected",
    "valid_timesteps",
    "normalization_valid",
    "timestamps_ms",
    "source_frame_indices",
    "source_frame_count",
    "source_fps",
    "label",
    "label_id",
    "split",
    "relative_video_path",
    "schema_version",
}


@dataclass(frozen=True, slots=True)
class ExpectedFeature:
    split: DatasetSplit
    label: str
    label_id: int
    relative_video_path: PurePosixPath

    @property
    def sample_id(self) -> str:
        normalized = self.relative_video_path.with_suffix("").as_posix()
        return f"{self.split.value}__{'__'.join(normalized.split('/'))}"

    def feature_path(self, output_root: Path) -> Path:
        return output_root / self.split.value / self.label / f"{self.sample_id}.npz"


def load_expected_features(
    manifest_root: str | Path,
    label_map_path: str | Path,
    subset: str,
    splits: Iterable[DatasetSplit] = tuple(DatasetSplit),
    excluded_labels: Iterable[str] = (),
) -> list[ExpectedFeature]:
    """Load expected feature identities without reopening raw video files."""

    labels = exclude_and_reindex_labels(
        load_label_map(label_map_path), excluded_labels
    )
    manifest_root = Path(manifest_root).expanduser()
    expected: list[ExpectedFeature] = []
    labels_to_exclude = {canonicalize_label(label) for label in excluded_labels}
    for split in splits:
        manifest_path = manifest_root / f"{subset}_{split.value}.txt"
        for relative_path in read_manifest(manifest_path):
            label = label_from_path(relative_path)
            if label in labels_to_exclude:
                continue
            try:
                label_id = labels[label]
            except KeyError as error:
                raise ValueError(
                    f"Manifest label {label!r} is absent from the label map: {relative_path}"
                ) from error
            expected.append(
                ExpectedFeature(split, label, label_id, relative_path)
            )
    return expected


def _scalar_text(archive: Any, key: str) -> str:
    return str(np.asarray(archive[key]).item())


def inspect_feature(
    path: Path,
    expected: ExpectedFeature,
    sequence_length: int,
) -> dict[str, Any]:
    """Validate one archive and return model-relevant quality metrics."""

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_KEYS - set(archive.files))
        if missing:
            raise ValueError(f"missing keys: {', '.join(missing)}")
        coordinates = archive["coordinates"]
        confidence = archive["confidence"]
        detected = archive["detected"]
        valid = archive["valid_timesteps"]
        normalization_valid = archive["normalization_valid"]
        if coordinates.shape != (sequence_length, 51, 3):
            raise ValueError(f"unexpected coordinate shape: {coordinates.shape}")
        if confidence.shape != (sequence_length, 51) or detected.shape != (sequence_length, 51):
            raise ValueError("confidence/detected shape does not match the landmark schema")
        if valid.shape != (sequence_length,) or normalization_valid.shape != (sequence_length,):
            raise ValueError("timestep-mask shape does not match the configured sequence length")
        if _scalar_text(archive, "schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        if _scalar_text(archive, "label") != expected.label:
            raise ValueError("stored label differs from manifest label")
        # IDs are benchmark-specific. INCLUDE-50 and full INCLUDE can safely
        # reuse an archive when its text label and manifest identity agree.
        if _scalar_text(archive, "split") != expected.split.value:
            raise ValueError("stored split differs from manifest split")
        if _scalar_text(archive, "relative_video_path") != expected.relative_video_path.as_posix():
            raise ValueError("stored relative video path differs from manifest path")

        real = int(valid.sum())
        if real == 0:
            raise ValueError("archive contains no real timesteps")
        active = detected[valid]
        normalized = int(normalization_valid[valid].sum())
        left = int(active[:, :21].any(axis=1).sum())
        right = int(active[:, 21:42].any(axis=1).sum())
        pose = int(active[:, 42:].any(axis=1).sum())
        return {
            "sample_id": expected.sample_id,
            "split": expected.split.value,
            "label": expected.label,
            "label_id": expected.label_id,
            "relative_video_path": expected.relative_video_path.as_posix(),
            "feature_path": str(path),
            "real_timesteps": real,
            "padded_timesteps": sequence_length - real,
            "normalization_valid_timesteps": normalized,
            "left_hand_frames": left,
            "right_hand_frames": right,
            "pose_frames": pose,
            "left_hand_coverage": round(left / real, 6),
            "right_hand_coverage": round(right / real, 6),
            "pose_coverage": round(pose / real, 6),
        }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {"samples": 0}
    numeric = (
        "real_timesteps",
        "padded_timesteps",
        "normalization_valid_timesteps",
        "left_hand_coverage",
        "right_hand_coverage",
        "pose_coverage",
    )
    return {
        "samples": count,
        **{
            f"mean_{name}": round(sum(float(record[name]) for record in records) / count, 6)
            for name in numeric
        },
        "samples_without_hands": sum(
            record["left_hand_frames"] == 0 and record["right_hand_frames"] == 0
            for record in records
        ),
        "samples_without_pose": sum(record["pose_frames"] == 0 for record in records),
        "samples_without_normalization": sum(
            record["normalization_valid_timesteps"] == 0 for record in records
        ),
    }


def build_quality_report(
    output_root: str | Path,
    expected: list[ExpectedFeature],
    sequence_length: int,
) -> dict[str, Any]:
    """Audit exact manifest coverage and summarize valid landmark quality."""

    root = Path(output_root).expanduser().resolve()
    expected_paths = {item.feature_path(root).resolve(): item for item in expected}
    actual_paths = set(root.rglob("*.npz"))
    missing = sorted(path for path in expected_paths if path not in actual_paths)
    unexpected = sorted(path for path in actual_paths if path not in expected_paths)
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, str]] = []

    for path, item in sorted(expected_paths.items()):
        if path not in actual_paths:
            continue
        try:
            valid_records.append(inspect_feature(path, item, sequence_length))
        except (OSError, ValueError, KeyError) as error:
            invalid_records.append(
                {"sample_id": item.sample_id, "feature_path": str(path), "error": str(error)}
            )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid_records:
        by_split[record["split"]].append(record)
        by_class[record["label"]].append(record)
    expected_by_split = Counter(item.split.value for item in expected)
    found_by_split = Counter(record["split"] for record in valid_records)

    return {
        "output_root": str(root),
        "expected_manifest_samples": len(expected),
        "valid_manifest_features": len(valid_records),
        "missing_manifest_features": len(missing),
        "invalid_manifest_features": len(invalid_records),
        "unexpected_features": len(unexpected),
        "expected_by_split": dict(sorted(expected_by_split.items())),
        "valid_by_split": dict(sorted(found_by_split.items())),
        "quality": _summary(valid_records),
        "quality_by_split": {
            split: _summary(records) for split, records in sorted(by_split.items())
        },
        "quality_by_class": {
            label: _summary(records) for label, records in sorted(by_class.items())
        },
        "missing": [str(path) for path in missing],
        "invalid": invalid_records,
        "unexpected": [str(path) for path in unexpected],
        "samples": valid_records,
    }
