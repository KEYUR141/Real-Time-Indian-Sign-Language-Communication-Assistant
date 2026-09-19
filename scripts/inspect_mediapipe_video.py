"""Run a real MediaPipe smoke test on one INCLUDE-50 video.

This script reads but never modifies the source video. It reports landmark
coverage only; it does not save training features.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from isl_assistant.config import load_config  # noqa: E402
from isl_assistant.data.manifests import load_dataset_samples  # noqa: E402
from isl_assistant.data.schemas import DatasetSplit  # noqa: E402
from isl_assistant.vision.landmarks.mediapipe_detector import (  # noqa: E402
    MediaPipeLandmarkDetector,
)
from isl_assistant.vision.landmarks.model_config import load_model_config  # noqa: E402
from isl_assistant.vision.landmarks.temporal_sampling import (  # noqa: E402
    build_sampling_plan,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "landmarks.toml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mediapipe_models.toml",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional direct video path. Otherwise the first INCLUDE-50 test sample is used.",
    )
    parser.add_argument("--sequence-length", type=int, help="Override configured frame limit.")
    parser.add_argument(
        "--swap-handedness-labels",
        action="store_true",
        help="Swap MediaPipe Left/Right labels for visual-orientation experiments.",
    )
    return parser.parse_args()


def select_video(arguments: argparse.Namespace, pipeline_config) -> tuple[Path, str]:
    if arguments.video is not None:
        video_path = arguments.video.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
        return video_path, "direct-video"

    samples = load_dataset_samples(
        manifest_root=pipeline_config.dataset.manifest_root,
        label_map_path=pipeline_config.dataset.label_map_path,
        raw_root=pipeline_config.dataset.raw_root,
        subset=pipeline_config.dataset.subset,
        splits=(DatasetSplit.TEST,),
    )
    if not samples:
        raise RuntimeError("The selected test manifest contains no samples")
    sample = samples[0]
    return sample.absolute_path, f"{sample.label} ({sample.label_id})"


def inspect_video(
    video_path: Path,
    label_description: str,
    detector_settings,
    sequence_length: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0:
            raise RuntimeError(f"Video reports an invalid frame count: {total_frames}")
        if frames_per_second <= 0:
            raise RuntimeError(f"Video reports an invalid FPS: {frames_per_second}")

        plan = build_sampling_plan(total_frames, sequence_length)
        requested_indices = set(plan.source_indices)
        timestamps = dict(zip(plan.source_indices, plan.timestamps_ms(frames_per_second), strict=True))
        coverage: Counter[str] = Counter()
        processed_frames = 0

        with MediaPipeLandmarkDetector(detector_settings) as detector:
            frame_index = 0
            while frame_index < total_frames:
                decoded, bgr_frame = capture.read()
                if not decoded:
                    break

                if frame_index in requested_indices:
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    result = detector.detect_rgb(rgb_frame, timestamps[frame_index])
                    processed_frames += 1
                    coverage["left_hand"] += int(result.has_left_hand)
                    coverage["right_hand"] += int(result.has_right_hand)
                    coverage["pose"] += int(result.has_pose)

                frame_index += 1

        print(f"Video: {video_path}")
        print(f"Label: {label_description}")
        print(f"Source frames: {total_frames}")
        print(f"FPS: {frames_per_second:.3f}")
        print(f"Sampled frames: {processed_frames}/{plan.sampled_frame_count}")
        print(f"Padding required: {plan.padding_count}")
        for key in ("left_hand", "right_hand", "pose"):
            percentage = 100.0 * coverage[key] / processed_frames if processed_frames else 0.0
            print(f"{key} coverage: {coverage[key]}/{processed_frames} ({percentage:.1f}%)")
    finally:
        capture.release()


def main() -> None:
    arguments = parse_arguments()
    pipeline_config = load_config(arguments.pipeline_config)
    model_config = load_model_config(arguments.model_config, project_root=PROJECT_ROOT)
    sequence_length = arguments.sequence_length or pipeline_config.landmarks.sequence_length

    if sequence_length <= 0:
        raise ValueError("sequence length must be greater than zero")

    video_path, label_description = select_video(arguments, pipeline_config)
    settings = model_config.detector_settings(
        maximum_hands=pipeline_config.landmarks.maximum_hands,
        minimum_detection_confidence=(
            pipeline_config.landmarks.minimum_detection_confidence
        ),
        minimum_presence_confidence=(
            pipeline_config.landmarks.minimum_detection_confidence
        ),
        minimum_tracking_confidence=(
            pipeline_config.landmarks.minimum_tracking_confidence
        ),
        swap_handedness_labels=arguments.swap_handedness_labels,
    )
    inspect_video(video_path, label_description, settings, sequence_length)


if __name__ == "__main__":
    main()

