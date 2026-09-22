"""Run live MediaPipe landmark inference with rejection and temporal smoothing."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isl_assistant.inference import (  # noqa: E402
    CandidateDecision,
    IsolatedLandmarkCapture,
    JsonlEvaluationWriter,
    LiveEvaluationRecord,
    LiveTrialTracker,
    Prediction,
    RejectionConfig,
    StabilityResult,
    TemporalPredictor,
    UnknownRejector,
)
from isl_assistant.vision.landmarks.mediapipe_detector import (  # noqa: E402
    MediaPipeLandmarkDetector,
)
from isl_assistant.vision.landmarks.model_config import load_model_config  # noqa: E402
from isl_assistant.vision.landmarks.visualization import (  # noqa: E402
    draw_frame_landmarks,
)

WINDOW_NAME = "Real-Time ISL Recognition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mediapipe_models.toml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--minimum-frames", type=int, default=16)
    parser.add_argument("--maximum-capture-frames", type=int, default=600)
    parser.add_argument("--minimum-hand-coverage", type=float, default=0.5)
    parser.add_argument("--minimum-normalization-coverage", type=float, default=0.8)
    parser.add_argument("--minimum-probability", type=float, default=0.75)
    parser.add_argument("--minimum-margin", type=float, default=0.2)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--swap-handedness-labels", action="store_true")
    parser.add_argument("--log", type=Path, help="Write inference attempts as JSONL.")
    parser.add_argument("--expected-label", help="Prompted trial's intended canonical label.")
    parser.add_argument("--trial-id", help="Identifier stored in the evaluation log.")
    return parser.parse_args()


def camera_backend(name: str) -> int:
    return {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }[name]


def put_lines(image: object, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    height = 34 + 27 * len(lines)
    cv2.rectangle(image, (8, 50), (760, height), (0, 0, 0), -1)
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            image,
            line,
            (16, 76 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("camera width and height must be positive")

    predictor = TemporalPredictor.from_checkpoint(args.checkpoint, device=args.device)
    if (
        args.expected_label is not None
        and args.expected_label not in predictor.label_names.values()
    ):
        raise ValueError(f"unknown --expected-label: {args.expected_label}")
    if args.minimum_frames > predictor.maximum_sequence_length:
        raise ValueError("--minimum-frames cannot exceed checkpoint sequence length")
    models = load_model_config(args.model_config, verify=True)
    detector_settings = models.detector_settings(
        swap_handedness_labels=args.swap_handedness_labels
    )
    rejector = UnknownRejector(
        RejectionConfig(
            minimum_frames=args.minimum_frames,
            minimum_hand_coverage=args.minimum_hand_coverage,
            minimum_normalization_coverage=args.minimum_normalization_coverage,
            minimum_probability=args.minimum_probability,
            minimum_margin=args.minimum_margin,
        )
    )
    capture_buffer = IsolatedLandmarkCapture(
        target_length=predictor.maximum_sequence_length,
        maximum_frames=args.maximum_capture_frames,
    )
    tracker = LiveTrialTracker(args.expected_label)
    writer = (
        JsonlEvaluationWriter(
            args.log,
            {
                "created_at": datetime.now(UTC).isoformat(),
                "trial_id": args.trial_id,
                "expected_label": args.expected_label,
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "device": str(predictor.device),
                "camera": args.camera,
                "mirror": args.mirror,
                "mode": "manual_isolated_capture",
                "rejection": {
                    "minimum_frames": args.minimum_frames,
                    "minimum_hand_coverage": args.minimum_hand_coverage,
                    "minimum_normalization_coverage": args.minimum_normalization_coverage,
                    "minimum_probability": args.minimum_probability,
                    "minimum_margin": args.minimum_margin,
                },
                "maximum_capture_frames": args.maximum_capture_frames,
            },
        )
        if args.log is not None
        else None
    )

    capture = cv2.VideoCapture(args.camera, camera_backend(args.camera_backend))
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_count = 0
    emitted_words: list[str] = []
    predictions: list[Prediction] = []
    decision = CandidateDecision(None, 0.0, 0.0, "collecting")
    stability = StabilityResult(None, None, 0, "collecting")
    capturing = False
    inference_ms = 0.0
    smoothed_fps = 0.0
    started = time.perf_counter()
    previous_frame_time = started
    last_timestamp_ms = -1

    print("Webcam controls: Space start/stop capture | R discard | C clear | Q/Esc quit")
    try:
        with MediaPipeLandmarkDetector(detector_settings) as detector:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("camera stopped returning frames")
                if args.mirror:
                    bgr = cv2.flip(bgr, 1)

                now = time.perf_counter()
                timestamp_ms = max(last_timestamp_ms + 1, int((now - started) * 1000))
                last_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect_rgb(rgb, timestamp_ms)
                frame_count += 1
                if capturing and not capture_buffer.is_full:
                    capture_buffer.append(landmarks)
                snapshot = capture_buffer.snapshot()

                frame_elapsed = max(now - previous_frame_time, 1e-6)
                instant_fps = 1.0 / frame_elapsed
                smoothed_fps = (
                    instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                )
                previous_frame_time = now

                rendered = draw_frame_landmarks(
                    bgr,
                    landmarks,
                    frame_number=frame_count,
                    timestamp_ms=timestamp_ms,
                )
                top_text = (
                    f"{predictions[0].label} {predictions[0].probability:.1%}"
                    if predictions
                    else "none"
                )
                status_color = (80, 220, 80) if decision.accepted else (80, 180, 255)
                put_lines(
                    rendered,
                    [
                        (
                            f"Capture: {'RECORDING' if capturing else 'READY'} | "
                            f"frames {len(capture_buffer)}/{capture_buffer.maximum_frames} | "
                            f"hands {snapshot.any_hand_coverage:.0%} | "
                            f"normalized {snapshot.normalization_coverage:.0%}",
                            (80, 80, 255) if capturing else (255, 255, 255),
                        ),
                        (f"Top: {top_text}", (255, 255, 255)),
                        (
                            f"Decision: {decision.reason}",
                            status_color,
                        ),
                        (
                            f"FPS: {smoothed_fps:.1f} | model: {inference_ms:.1f} ms | "
                            f"device: {predictor.device}",
                            (255, 255, 255),
                        ),
                        ("Text: " + " ".join(emitted_words[-8:]), (80, 220, 80)),
                        ("Space start/stop | R discard | C clear | Q quit", (180, 180, 180)),
                    ],
                )
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    if not capturing:
                        capture_buffer.reset()
                        predictions = []
                        decision = CandidateDecision(None, 0.0, 0.0, "capturing")
                        stability = StabilityResult(None, None, 0, "capturing")
                        capturing = True
                        print("CAPTURE START")
                    else:
                        capturing = False
                        snapshot = capture_buffer.snapshot()
                        if snapshot.length < args.minimum_frames:
                            predictions = []
                            decision = rejector.evaluate(snapshot, predictions)
                        else:
                            sequence, length = snapshot.model_input()
                            inference_started = time.perf_counter()
                            predictions = predictor.predict(sequence, length, top_k=5)
                            inference_ms = (time.perf_counter() - inference_started) * 1000
                            decision = rejector.evaluate(snapshot, predictions)
                        emitted = decision.label if decision.accepted else None
                        stability = StabilityResult(
                            decision.label,
                            emitted,
                            1 if decision.accepted else 0,
                            "emitted" if decision.accepted else decision.reason,
                        )
                        record = LiveEvaluationRecord(
                            elapsed_ms=timestamp_ms,
                            frame_number=frame_count,
                            buffer_length=snapshot.length,
                            top_label=predictions[0].label if predictions else None,
                            top_probability=predictions[0].probability if predictions else 0.0,
                            second_label=predictions[1].label if len(predictions) > 1 else None,
                            second_probability=(
                                predictions[1].probability if len(predictions) > 1 else 0.0
                            ),
                            margin=decision.margin,
                            decision_reason=decision.reason,
                            stable_count=stability.consecutive,
                            emitted_label=emitted,
                            hand_coverage=snapshot.any_hand_coverage,
                            normalization_coverage=snapshot.normalization_coverage,
                            model_latency_ms=inference_ms,
                            fps=smoothed_fps,
                        )
                        tracker.add(record)
                        if writer is not None:
                            writer.write_record(record)
                        print(
                            f"CAPTURE STOP frames={len(capture_buffer)}, "
                            f"sampled={snapshot.length}, hands={snapshot.any_hand_coverage:.1%}"
                        )
                        for rank, prediction in enumerate(predictions, start=1):
                            print(
                                f"  {rank}. {prediction.label} "
                                f"({prediction.probability:.2%})"
                            )
                        if emitted is not None:
                            emitted_words.append(emitted)
                            print(
                                f"ACCEPT {emitted} probability={decision.probability:.2%} "
                                f"margin={decision.margin:.2%}"
                            )
                        else:
                            print(f"REJECT {decision.reason}")
                if key == ord("r"):
                    capture_buffer.reset()
                    capturing = False
                    predictions = []
                    decision = CandidateDecision(None, 0.0, 0.0, "ready")
                    stability = StabilityResult(None, None, 0, "ready")
                    print("CAPTURE DISCARDED")
                elif key == ord("c"):
                    emitted_words.clear()
    finally:
        capture.release()
        cv2.destroyAllWindows()

    summary = tracker.summary()
    if writer is not None:
        writer.write_summary(summary)
        writer.close()
    print("Recognized text:", " ".join(emitted_words) or "<empty>")
    print(
        "Trial summary: "
        f"attempts={summary.attempts}, "
        f"top1_expected={summary.top1_expected_attempts}, "
        f"correct_emissions={summary.correct_emissions}, "
        f"false_emissions={summary.false_emissions}, "
        f"mean_model_latency={summary.mean_model_latency_ms:.2f} ms"
    )
    if args.log is not None:
        print(f"Evaluation log: {args.log.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
