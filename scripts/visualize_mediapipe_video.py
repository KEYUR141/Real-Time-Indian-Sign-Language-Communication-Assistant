"""Preview or export MediaPipe landmarks drawn over a source video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.vision.landmarks.mediapipe_detector import (  # noqa: E402
    MediaPipeLandmarkDetector,
)
from isl_assistant.vision.landmarks.model_config import load_model_config  # noqa: E402
from isl_assistant.vision.landmarks.visualization import (  # noqa: E402
    draw_frame_landmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mediapipe_models.toml",
    )
    parser.add_argument("--swap-handedness-labels", action="store_true")
    parser.add_argument("--no-preview", action="store_true",
                        help="Export without opening an interactive window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.no_preview and args.output is None:
        raise ValueError("--no-preview requires --output")

    models = load_model_config(args.model_config, verify=True)
    settings = models.detector_settings(
        swap_handedness_labels=args.swap_handedness_labels
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

    paused = False
    step_once = False
    frame_number = 0
    last_rendered = None
    try:
        with MediaPipeLandmarkDetector(settings) as detector:
            while True:
                if not paused or step_once:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    timestamp_ms = round(frame_number * 1000.0 / fps)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    landmarks = detector.detect_rgb(rgb, timestamp_ms)
                    last_rendered = draw_frame_landmarks(
                        bgr,
                        landmarks,
                        frame_number=frame_number,
                        timestamp_ms=timestamp_ms,
                    )
                    if writer is not None:
                        writer.write(last_rendered)
                    frame_number += 1
                    step_once = False

                if args.no_preview:
                    continue
                if last_rendered is not None:
                    cv2.imshow("ISL MediaPipe landmark inspection", last_rendered)
                key = cv2.waitKey(0 if paused else max(1, round(1000 / fps))) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    paused = not paused
                elif paused and key in (ord("n"), ord("d"), 83):
                    step_once = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed frames: {frame_number}")
    if args.output is not None:
        print(f"Annotated video: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
