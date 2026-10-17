"""Thread-safe bridge between uploaded videos and the temporal classifier."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from isl_assistant.inference.decision import RejectionConfig, UnknownRejector
from isl_assistant.inference.rolling_buffer import IsolatedLandmarkCapture
from isl_assistant.inference.temporal_predictor import TemporalPredictor, temporal_features
from isl_assistant.vision.landmarks.extractor import MediaPipeTimestampClock, extract_video
from isl_assistant.vision.landmarks.feature_schema import FrameLandmarks
from isl_assistant.vision.landmarks.mediapipe_detector import MediaPipeLandmarkDetector
from isl_assistant.vision.landmarks.model_config import load_model_config


class InferenceService:
    """Lazily load the checkpoint and run one MediaPipe request at a time."""

    def __init__(self, checkpoint: Path, model_config: Path, device: str = "auto") -> None:
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.requested_device = device
        self._predictor: TemporalPredictor | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        if self._predictor is not None:
            return {
                "ready": True,
                "model_loaded": True,
                "device": str(self._predictor.device),
                "classes": len(self._predictor.label_names),
                "sequence_length": self._predictor.maximum_sequence_length,
                "error": None,
            }
        return {
            "ready": self.checkpoint.is_file() and self.model_config.is_file(),
            "model_loaded": False,
            "device": self.requested_device,
            "classes": 262,
            "sequence_length": 64,
            "error": self._load_error,
        }

    def _load_predictor(self) -> TemporalPredictor:
        if self._predictor is None:
            try:
                self._predictor = TemporalPredictor.from_checkpoint(
                    self.checkpoint, device=self.requested_device
                )
                self._load_error = None
            except Exception as error:
                self._load_error = str(error)
                raise
        return self._predictor

    def predict_video(self, video_path: Path, top_k: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            predictor = self._load_predictor()
            models = load_model_config(self.model_config, verify=True)
            settings = models.detector_settings()
            with MediaPipeLandmarkDetector(settings) as detector:
                extracted = extract_video(
                    video_path,
                    detector,
                    predictor.maximum_sequence_length,
                    MediaPipeTimestampClock(),
                )
            sequence, length = temporal_features(extracted.arrays)
            predictions = predictor.predict(sequence, length, top_k=top_k)

        quality = extracted.quality
        ranked = [
            {
                "rank": rank,
                "label_id": prediction.label_id,
                "label": prediction.label,
                "probability": prediction.probability,
            }
            for rank, prediction in enumerate(predictions, start=1)
        ]
        top_probability = ranked[0]["probability"]
        margin = top_probability - ranked[1]["probability"] if len(ranked) > 1 else top_probability
        best_hand_coverage = max(quality.left_hand_coverage, quality.right_hand_coverage)
        accepted = top_probability >= 0.75 and margin >= 0.20 and best_hand_coverage >= 0.25
        return {
            "accepted": accepted,
            "decision": ranked[0]["label"] if accepted else "unknown",
            "reason": (
                None
                if accepted
                else "Prediction did not pass confidence, margin, or hand-coverage checks."
            ),
            "predictions": ranked,
            "quality": {
                "source_frames": quality.total_frames,
                "sampled_frames": quality.sampled_frames,
                "real_timesteps": length,
                "left_hand_coverage": quality.left_hand_coverage,
                "right_hand_coverage": quality.right_hand_coverage,
                "pose_coverage": quality.pose_coverage,
            },
            "runtime": {
                "device": str(predictor.device),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        }

    def predict_landmarks(
        self,
        frame_values: list[dict[str, Any]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify browser-detected landmarks using the desktop capture pipeline."""

        started = time.perf_counter()
        with self._lock:
            predictor = self._load_predictor()
            capture = IsolatedLandmarkCapture(
                target_length=predictor.maximum_sequence_length,
                maximum_frames=max(600, len(frame_values)),
            )
            for values in frame_values:
                capture.append(
                    FrameLandmarks(
                        coordinates=np.asarray(values["coordinates"], dtype=np.float32),
                        confidence=np.asarray(values["confidence"], dtype=np.float32),
                        detected=np.asarray(values["detected"], dtype=np.bool_),
                        timestamp_ms=int(values["timestamp_ms"]),
                    )
                )
            snapshot = capture.snapshot()
            sequence, length = snapshot.model_input()
            predictions = predictor.predict(sequence, length, top_k=top_k)
            decision = UnknownRejector(
                RejectionConfig(
                    minimum_frames=16,
                    minimum_hand_coverage=0.5,
                    minimum_normalization_coverage=0.8,
                    minimum_probability=0.75,
                    minimum_margin=0.2,
                )
            ).evaluate(snapshot, predictions)

        ranked = [
            {
                "rank": rank,
                "label_id": prediction.label_id,
                "label": prediction.label,
                "probability": prediction.probability,
            }
            for rank, prediction in enumerate(predictions, start=1)
        ]
        return {
            "accepted": decision.accepted,
            "decision": decision.label or "unknown",
            "reason": decision.reason,
            "predictions": ranked,
            "quality": {
                "source_frames": len(frame_values),
                "sampled_frames": length,
                "real_timesteps": length,
                "left_hand_coverage": snapshot.left_hand_coverage,
                "right_hand_coverage": snapshot.right_hand_coverage,
                "pose_coverage": snapshot.pose_coverage,
                "normalization_coverage": snapshot.normalization_coverage,
            },
            "runtime": {
                "device": str(predictor.device),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input": "browser_landmarks",
            },
        }
