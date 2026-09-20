"""Validate and summarize a folder of landmark ``.npz`` feature files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SCHEMA_VERSION = "mediapipe-hands-pose-v1"
EXPECTED_KEYS = {
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
    "source_path",
    "schema_version",
}
MANIFEST_METADATA_KEYS = {"label_id", "split", "relative_video_path"}


def scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def validate_file(
    path: Path, sequence_length: int, expected_label: str | None
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing_keys = sorted(EXPECTED_KEYS - keys)
        if missing_keys:
            errors.append(f"missing keys: {', '.join(missing_keys)}")
            return {"file": path.name, "valid": False, "errors": errors}

        coordinates = archive["coordinates"]
        raw_coordinates = archive["raw_coordinates"]
        confidence = archive["confidence"]
        detected = archive["detected"]
        valid_timesteps = archive["valid_timesteps"]
        normalization_valid = archive["normalization_valid"]
        timestamps_ms = archive["timestamps_ms"]
        source_frame_indices = archive["source_frame_indices"]
        source_frame_count = int(archive["source_frame_count"].item())
        source_fps = float(archive["source_fps"].item())
        label = scalar_text(archive["label"])
        source_path = scalar_text(archive["source_path"])
        schema_version = scalar_text(archive["schema_version"])
        manifest_keys = keys & MANIFEST_METADATA_KEYS
        label_id: int | None = None
        split: str | None = None
        relative_video_path: str | None = None
        if manifest_keys and manifest_keys != MANIFEST_METADATA_KEYS:
            missing = sorted(MANIFEST_METADATA_KEYS - keys)
            errors.append(f"incomplete manifest metadata; missing: {', '.join(missing)}")
        elif manifest_keys == MANIFEST_METADATA_KEYS:
            label_id = int(archive["label_id"].item())
            split = scalar_text(archive["split"])
            relative_video_path = scalar_text(archive["relative_video_path"])
            if label_id < 0:
                errors.append("label_id must be non-negative")
            if split not in {"train", "val", "test"}:
                errors.append(f"split {split!r} is not train, val, or test")
            if not relative_video_path:
                errors.append("relative_video_path is empty")

        expected_shapes = {
            "coordinates": (sequence_length, 51, 3),
            "raw_coordinates": (sequence_length, 51, 3),
            "confidence": (sequence_length, 51),
            "detected": (sequence_length, 51),
            "valid_timesteps": (sequence_length,),
            "normalization_valid": (sequence_length,),
            "timestamps_ms": (sequence_length,),
            "source_frame_indices": (sequence_length,),
        }
        arrays = {
            "coordinates": coordinates,
            "raw_coordinates": raw_coordinates,
            "confidence": confidence,
            "detected": detected,
            "valid_timesteps": valid_timesteps,
            "normalization_valid": normalization_valid,
            "timestamps_ms": timestamps_ms,
            "source_frame_indices": source_frame_indices,
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                errors.append(
                    f"{name} shape {arrays[name].shape}, expected {expected_shape}"
                )

        for name in ("coordinates", "raw_coordinates", "confidence"):
            if arrays[name].dtype != np.float32:
                errors.append(f"{name} dtype {arrays[name].dtype}, expected float32")
            if not np.isfinite(arrays[name]).all():
                errors.append(f"{name} contains NaN or infinite values")

        for name in ("detected", "valid_timesteps", "normalization_valid"):
            if arrays[name].dtype != np.bool_:
                errors.append(f"{name} dtype {arrays[name].dtype}, expected bool")

        if schema_version != EXPECTED_SCHEMA_VERSION:
            errors.append(
                f"schema_version {schema_version!r}, expected {EXPECTED_SCHEMA_VERSION!r}"
            )
        if expected_label is not None and label != expected_label:
            errors.append(f"label {label!r}, expected {expected_label!r}")
        if not source_path:
            errors.append("source_path is empty")
        if source_frame_count <= 0:
            errors.append("source_frame_count must be positive")
        if not np.isfinite(source_fps) or source_fps <= 0:
            errors.append("source_fps must be positive and finite")

        real_count = int(valid_timesteps.sum())
        normalized_count = int(normalization_valid.sum())
        if real_count == 0:
            errors.append("no valid timesteps")
        if np.any(normalization_valid & ~valid_timesteps):
            errors.append("padded timestep marked normalization-valid")
        if normalized_count < real_count:
            warnings.append(
                f"normalization valid for {normalized_count}/{real_count} real frames"
            )

        padded = ~valid_timesteps
        if padded.any():
            if np.any(coordinates[padded]) or np.any(raw_coordinates[padded]):
                errors.append("padded coordinates are not zero")
            if np.any(detected[padded]) or np.any(confidence[padded]):
                errors.append("padded detection features are not zero")
            if not np.all(timestamps_ms[padded] == -1):
                errors.append("padded timestamps must be -1")
            if not np.all(source_frame_indices[padded] == -1):
                errors.append("padded source frame indices must be -1")

        real_detected = detected[valid_timesteps]
        left_frames = int(real_detected[:, :21].any(axis=1).sum())
        right_frames = int(real_detected[:, 21:42].any(axis=1).sum())
        pose_frames = int(real_detected[:, 42:].any(axis=1).sum())
        finite_values = coordinates[np.isfinite(coordinates)]
        coordinate_min = float(finite_values.min()) if finite_values.size else 0.0
        coordinate_max = float(finite_values.max()) if finite_values.size else 0.0

    return {
        "file": path.name,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_path": source_path,
        "label": label,
        "label_id": label_id,
        "split": split,
        "relative_video_path": relative_video_path,
        "source_frames": source_frame_count,
        "fps": round(source_fps, 3),
        "real_timesteps": real_count,
        "padded_timesteps": sequence_length - real_count,
        "normalization_valid_timesteps": normalized_count,
        "left_hand_frames": left_frames,
        "right_hand_frames": right_frames,
        "pose_frames": pose_frames,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "size_bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--label",
        help="Optional expected label (useful for validating a single class folder).",
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--report-name", default="validation_report.json",
        help="JSON filename written inside the input folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_directory = args.input.expanduser().resolve()
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_directory}")

    paths = sorted(input_directory.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No .npz files found in: {input_directory}")

    results: list[dict[str, Any]] = []
    for number, path in enumerate(paths, start=1):
        try:
            result = validate_file(path, args.sequence_length, args.label)
        except Exception as error:
            result = {
                "file": path.name,
                "valid": False,
                "errors": [f"could not load: {error}"],
                "warnings": [],
                "size_bytes": path.stat().st_size,
            }
        results.append(result)
        status = "OK" if result["valid"] else "INVALID"
        print(f"[{number}/{len(paths)}] {status} {path.name}")
        for message in result.get("errors", []):
            print(f"  ERROR: {message}")
        for message in result.get("warnings", []):
            print(f"  WARNING: {message}")

    valid_results = [result for result in results if result["valid"]]
    invalid_results = [result for result in results if not result["valid"]]
    total_size = sum(int(result.get("size_bytes", 0)) for result in results)
    report = {
        "input": str(input_directory),
        "expected_label": args.label,
        "files_found": len(results),
        "valid_files": len(valid_results),
        "invalid_files": len(invalid_results),
        "files_with_warnings": sum(bool(result.get("warnings")) for result in results),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 3),
        "total_real_timesteps": sum(
            int(result.get("real_timesteps", 0)) for result in valid_results
        ),
        "total_padded_timesteps": sum(
            int(result.get("padded_timesteps", 0)) for result in valid_results
        ),
        "total_normalization_valid_timesteps": sum(
            int(result.get("normalization_valid_timesteps", 0))
            for result in valid_results
        ),
        "by_split": {
            split: sum(result.get("split") == split for result in valid_results)
            for split in ("train", "val", "test")
        },
        "by_class": {
            label: sum(result.get("label") == label for result in valid_results)
            for label in sorted(
                {str(result["label"]) for result in valid_results if result.get("label")}
            )
        },
        "results": results,
    }
    report_path = input_directory / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Files found: {report['files_found']}")
    print(f"Valid files: {report['valid_files']}")
    print(f"Invalid files: {report['invalid_files']}")
    print(f"Files with warnings: {report['files_with_warnings']}")
    print(f"Total size: {report['total_size_mb']:.3f} MB")
    print(f"Report: {report_path}")
    return 1 if invalid_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
