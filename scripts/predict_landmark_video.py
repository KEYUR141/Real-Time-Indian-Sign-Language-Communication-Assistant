"""Classify one video with MediaPipe and a trained temporal checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.inference.temporal_predictor import (  # noqa: E402
    TemporalPredictor,
    temporal_features,
)
from isl_assistant.vision.landmarks.extractor import (  # noqa: E402
    MediaPipeTimestampClock,
    extract_video,
)
from isl_assistant.vision.landmarks.mediapipe_detector import (  # noqa: E402
    MediaPipeLandmarkDetector,
)
from isl_assistant.vision.landmarks.model_config import load_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mediapipe_models.toml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--swap-handedness-labels", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video does not exist: {video}")
    predictor = TemporalPredictor.from_checkpoint(args.checkpoint, device=args.device)
    models = load_model_config(args.model_config, verify=True)
    settings = models.detector_settings(
        swap_handedness_labels=args.swap_handedness_labels
    )
    with MediaPipeLandmarkDetector(settings) as detector:
        extracted = extract_video(
            video,
            detector,
            predictor.maximum_sequence_length,
            MediaPipeTimestampClock(),
        )
    sequence, length = temporal_features(extracted.arrays)
    predictions = predictor.predict(sequence, length, top_k=args.top_k)
    quality = extracted.quality

    print("=== VIDEO PREDICTION ===")
    print(f"Video: {video}")
    print(f"Device: {predictor.device}")
    print(f"Real timesteps: {length}/{predictor.maximum_sequence_length}")
    print(
        "Coverage: "
        f"left={quality.left_hand_coverage:.1%}, "
        f"right={quality.right_hand_coverage:.1%}, "
        f"pose={quality.pose_coverage:.1%}"
    )
    print("Predictions:")
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}. {prediction.label} "
            f"(id={prediction.label_id}, probability={prediction.probability:.2%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

