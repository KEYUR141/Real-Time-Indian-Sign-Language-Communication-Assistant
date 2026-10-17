from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from isl_assistant.api.app import create_app


class FakeInferenceService:
    def status(self) -> dict[str, object]:
        return {
            "ready": True,
            "model_loaded": True,
            "device": "cpu",
            "classes": 262,
            "sequence_length": 64,
            "error": None,
        }

    def predict_video(self, video_path: Path, top_k: int = 5) -> dict[str, object]:
        assert video_path.read_bytes() == b"video-bytes"
        return {
            "accepted": True,
            "decision": "hat",
            "predictions": [{"rank": 1, "label_id": 97, "label": "hat", "probability": 0.99}],
            "quality": {"left_hand_coverage": 1.0, "right_hand_coverage": 0.625},
            "runtime": {"device": "cpu", "latency_ms": 10.0},
        }

    def predict_landmarks(
        self, frame_values: list[dict[str, object]], top_k: int = 5
    ) -> dict[str, object]:
        assert len(frame_values) == 16
        assert len(frame_values[0]["coordinates"]) == 51
        return {
            "accepted": True,
            "decision": "hat",
            "predictions": [{"rank": 1, "label_id": 97, "label": "hat", "probability": 0.99}],
            "quality": {"left_hand_coverage": 1.0, "right_hand_coverage": 0.625},
            "runtime": {"device": "cpu", "latency_ms": 10.0},
        }


def test_health_exposes_runtime_contract() -> None:
    client = TestClient(create_app(FakeInferenceService()))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["classes"] == 262


def test_video_upload_returns_real_service_result() -> None:
    client = TestClient(create_app(FakeInferenceService()))

    response = client.post(
        "/api/predict/video",
        files={"video": ("hat.mov", b"video-bytes", "video/quicktime")},
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "hat"


def test_video_upload_rejects_unsupported_extension() -> None:
    client = TestClient(create_app(FakeInferenceService()))

    response = client.post(
        "/api/predict/video",
        files={"video": ("not-video.txt", b"video-bytes", "text/plain")},
    )

    assert response.status_code == 415


def test_browser_landmarks_use_direct_inference_contract() -> None:
    client = TestClient(create_app(FakeInferenceService()))
    frame = {
        "coordinates": [[0.0, 0.0, 0.0] for _ in range(51)],
        "confidence": [0.0 for _ in range(51)],
        "detected": [False for _ in range(51)],
        "timestamp_ms": 0,
    }
    frames = [{**frame, "timestamp_ms": index} for index in range(16)]

    response = client.post("/api/predict/landmarks", json={"frames": frames})

    assert response.status_code == 200
    assert response.json()["decision"] == "hat"
