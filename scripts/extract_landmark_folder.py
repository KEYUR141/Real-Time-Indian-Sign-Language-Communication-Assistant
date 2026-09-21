"""Extract INCLUDE/INCLUDE-50 landmarks selected by official manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.config import load_config  # noqa: E402
from isl_assistant.data.manifests import load_dataset_samples, read_manifest  # noqa: E402
from isl_assistant.data.schemas import DatasetSplit, VideoSample  # noqa: E402
from isl_assistant.vision.landmarks.extractor import (  # noqa: E402
    MediaPipeTimestampClock,
    extract_video,
    feature_path,
    save_extracted_video,
)
from isl_assistant.vision.landmarks.mediapipe_detector import (  # noqa: E402
    MediaPipeLandmarkDetector,
)
from isl_assistant.vision.landmarks.model_config import load_model_config  # noqa: E402

REQUIRED_RESUME_KEYS = {
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
    "source_path",
    "relative_video_path",
    "schema_version",
}


def scalar(archive: Any, key: str) -> Any:
    return np.asarray(archive[key]).item()


def completed_output(path: Path, sample: VideoSample, sequence_length: int) -> bool:
    """Return whether an existing artifact is safe to resume past."""

    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not REQUIRED_RESUME_KEYS.issubset(archive.files):
                return False
            return (
                archive["coordinates"].shape == (sequence_length, 51, 3)
                and str(scalar(archive, "label")) == sample.label
                and str(scalar(archive, "split")) == sample.split.value
                and str(scalar(archive, "relative_video_path"))
                == sample.relative_path.as_posix()
            )
    except (OSError, ValueError, KeyError):
        return False


def reuse_output(
    source: Path,
    destination: Path,
    sample: VideoSample,
    sequence_length: int,
) -> bool:
    """Reuse landmarks from another benchmark split without running MediaPipe."""

    if not source.is_file():
        return False
    try:
        with np.load(source, allow_pickle=False) as archive:
            if not REQUIRED_RESUME_KEYS.issubset(archive.files):
                return False
            if archive["coordinates"].shape != (sequence_length, 51, 3):
                return False
            if str(scalar(archive, "label")) != sample.label:
                return False
            if str(scalar(archive, "relative_video_path")) != sample.relative_path.as_posix():
                return False
            values = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, KeyError):
        return False

    values.update(
        label=np.asarray(sample.label),
        label_id=np.asarray(sample.label_id, dtype=np.int64),
        split=np.asarray(sample.split.value),
        relative_video_path=np.asarray(sample.relative_path.as_posix()),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(destination)
    return True


def quality_record(extracted: Any) -> dict[str, Any]:
    quality = extracted.quality
    return {
        "source_frames": quality.total_frames,
        "sampled_frames": quality.sampled_frames,
        "left_hand_frames": quality.frames_with_left_hand,
        "right_hand_frames": quality.frames_with_right_hand,
        "pose_frames": quality.frames_with_pose,
        "left_hand_coverage": round(quality.left_hand_coverage, 6),
        "right_hand_coverage": round(quality.right_hand_coverage, 6),
        "pose_coverage": round(quality.pose_coverage, 6),
    }


def build_report(
    args: argparse.Namespace,
    samples: list[VideoSample],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(result["status"] for result in results)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        by_split[result["split"]][result["status"]] += 1
        by_class[result["label"]][result["status"]] += 1

    def summaries(groups: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
        return {
            name: {
                "total": sum(counts.values()),
                "succeeded": counts["succeeded"],
                "resumed": counts["resumed"],
                "reused": counts["reused"],
                "failed": counts["failed"],
            }
            for name, counts in sorted(groups.items())
        }

    return {
        "config": str(args.config.expanduser().resolve()),
        "subset": args.subset,
        "reuse_subset": args.reuse_subset,
        "excluded_labels": list(args.excluded_labels),
        "splits": args.splits,
        "samples_selected": len(samples),
        "processed": len(results),
        "succeeded": statuses["succeeded"],
        "resumed": statuses["resumed"],
        "reused": statuses["reused"],
        "failed": statuses["failed"],
        "by_split": summaries(by_split),
        "by_class": summaries(by_class),
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "landmarks.toml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mediapipe_models.toml",
    )
    parser.add_argument("--subset", choices=("include", "include50"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(split.value for split in DatasetSplit),
        default=[split.value for split in DatasetSplit],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve and count selected videos without loading MediaPipe or extracting.",
    )
    parser.add_argument(
        "--reuse-subset",
        choices=("include50",),
        help="Reuse existing landmarks selected by this manifest subset.",
    )
    parser.add_argument("--swap-handedness-labels", action="store_true")
    parser.add_argument("--report-name", default="extraction_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    args.subset = args.subset or config.dataset.subset
    args.excluded_labels = config.dataset.excluded_labels
    splits = tuple(DatasetSplit(value) for value in args.splits)
    output_root = config.output.root.expanduser().resolve()
    report_path = output_root / args.report_name
    reusable_splits = {}
    if args.reuse_subset:
        for split in DatasetSplit:
            for relative_path in read_manifest(
                config.dataset.manifest_root / f"{args.reuse_subset}_{split.value}.txt"
            ):
                if relative_path in reusable_splits:
                    raise ValueError(f"Reuse subset repeats video path: {relative_path}")
                reusable_splits[relative_path] = split
    samples = load_dataset_samples(
        manifest_root=config.dataset.manifest_root,
        label_map_path=config.dataset.label_map_path,
        raw_root=config.dataset.raw_root,
        subset=args.subset,
        splits=splits,
        excluded_labels=config.dataset.excluded_labels,
    )
    if args.plan_only:
        by_split = Counter(sample.split.value for sample in samples)
        by_class = Counter(sample.label for sample in samples)
        reusable = sum(sample.relative_path in reusable_splits for sample in samples)
        print("=== EXTRACTION PLAN ===")
        print(f"Subset: {args.subset}")
        print(f"Reuse subset: {args.reuse_subset or 'none'}")
        print(f"Excluded labels: {list(config.dataset.excluded_labels)}")
        print(f"Benchmark videos: {len(samples)}")
        print(f"Classes selected: {len(by_class)}")
        print(f"By split: {dict(sorted(by_split.items()))}")
        print(f"Reusable landmark archives: {reusable}")
        print(f"MediaPipe conversions required: {len(samples) - reusable}")
        return 0
    models = load_model_config(args.model_config, verify=True)
    settings = models.detector_settings(
        swap_handedness_labels=args.swap_handedness_labels
    )
    sequence_length = config.landmarks.sequence_length
    overwrite = args.overwrite or config.output.overwrite
    clock = MediaPipeTimestampClock()
    results: list[dict[str, Any]] = []

    print(f"Selected {len(samples)} {args.subset} manifest samples")
    with MediaPipeLandmarkDetector(settings) as detector:
        for number, sample in enumerate(samples, start=1):
            destination = feature_path(output_root, sample)
            base = {
                "sample_id": sample.sample_id,
                "split": sample.split.value,
                "label": sample.label,
                "label_id": sample.label_id,
                "relative_video_path": sample.relative_path.as_posix(),
                "feature_path": str(destination),
            }
            if not overwrite and completed_output(destination, sample, sequence_length):
                results.append({**base, "status": "resumed"})
                print(f"[{number}/{len(samples)}] RESUME {sample.sample_id}")
            elif not overwrite and sample.relative_path in reusable_splits:
                reuse_sample = VideoSample(
                    relative_path=sample.relative_path,
                    absolute_path=sample.absolute_path,
                    label=sample.label,
                    label_id=sample.label_id,
                    split=reusable_splits[sample.relative_path],
                )
                source = feature_path(output_root, reuse_sample)
                if reuse_output(source, destination, sample, sequence_length):
                    results.append({**base, "status": "reused", "reused_from": str(source)})
                    print(f"[{number}/{len(samples)}] REUSE {sample.sample_id}")
                    write_report(report_path, build_report(args, samples, results))
                    continue
                try:
                    extracted = extract_video(
                        sample.absolute_path, detector, sequence_length, clock
                    )
                    save_extracted_video(destination, extracted, sample)
                    results.append(
                        {**base, "status": "succeeded", **quality_record(extracted)}
                    )
                    print(f"[{number}/{len(samples)}] OK {sample.sample_id}")
                except Exception as error:
                    results.append({**base, "status": "failed", "error": str(error)})
                    print(f"[{number}/{len(samples)}] FAIL {sample.sample_id}: {error}")
            else:
                try:
                    extracted = extract_video(
                        sample.absolute_path, detector, sequence_length, clock
                    )
                    save_extracted_video(destination, extracted, sample)
                    results.append(
                        {**base, "status": "succeeded", **quality_record(extracted)}
                    )
                    quality = extracted.quality
                    print(
                        f"[{number}/{len(samples)}] OK {sample.sample_id} "
                        f"frames={quality.sampled_frames}, "
                        f"left={quality.frames_with_left_hand}, "
                        f"right={quality.frames_with_right_hand}, "
                        f"pose={quality.frames_with_pose}"
                    )
                except Exception as error:
                    results.append({**base, "status": "failed", "error": str(error)})
                    print(f"[{number}/{len(samples)}] FAIL {sample.sample_id}: {error}")

            # Checkpoint after every sample so an interrupted run has a useful report.
            write_report(report_path, build_report(args, samples, results))

    report = build_report(args, samples, results)
    write_report(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Report: {report_path}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
