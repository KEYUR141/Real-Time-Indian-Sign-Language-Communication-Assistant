"""FastAPI application for the presentation frontend."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from isl_assistant.config import load_config
from isl_assistant.data.manifests import canonicalize_label, label_from_path, read_manifest
from isl_assistant.data.path_resolver import VideoPathResolver

from .inference_service import InferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = Path(
    r"E:\Hugging_Face\INCLUDE\processed\mediapipe_v1\models\full_include"
    r"\transformer_final_seed_42\selected_final.pt"
)
ALLOWED_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class BrowserLandmarkFrame(BaseModel):
    coordinates: list[list[float]]
    confidence: list[float]
    detected: list[bool]
    timestamp_ms: int = Field(ge=0)


class BrowserLandmarkSequence(BaseModel):
    frames: list[BrowserLandmarkFrame] = Field(min_length=16, max_length=600)


def create_app(service: InferenceService | None = None) -> FastAPI:
    checkpoint = Path(os.getenv("ISL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    model_config = Path(
        os.getenv("ISL_MODEL_CONFIG", str(PROJECT_ROOT / "configs" / "mediapipe_models.toml"))
    )
    inference = service or InferenceService(
        checkpoint=checkpoint,
        model_config=model_config,
        device=os.getenv("ISL_DEVICE", "auto"),
    )
    dataset_config_path = Path(
        os.getenv("ISL_DATA_CONFIG", str(PROJECT_ROOT / "configs" / "landmarks_include.toml"))
    )

    @lru_cache(maxsize=1)
    def reference_index() -> tuple[dict[str, Path], VideoPathResolver]:
        config = load_config(dataset_config_path)
        manifest = config.dataset.manifest_root / f"{config.dataset.subset}_test.txt"
        excluded = {canonicalize_label(label) for label in config.dataset.excluded_labels}
        entries: dict[str, Path] = {}
        for relative_path in read_manifest(manifest):
            label = label_from_path(relative_path)
            if label not in excluded:
                entries.setdefault(label, Path(relative_path.as_posix()))
        return entries, VideoPathResolver(config.dataset.raw_root)
    application = FastAPI(title="ISL Recognition API", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "online", "service": "isl-inference", **inference.status()}

    @application.get("/api/benchmark")
    def benchmark() -> dict[str, object]:
        return {
            "dataset": "INCLUDE",
            "provider": "AI4Bharat / IIT Madras",
            "distribution": "Zenodo v1",
            "architecture": "Compact Transformer",
            "classes": 262,
            "official_classes": 263,
            "semantic_categories": 15,
            "archive_count": 44,
            "compressed_download_gb": 56.8,
            "extracted_video_gb": 53.84,
            "verified_videos": 4284,
            "manifest_videos": 4292,
            "train_samples": 3121,
            "validation_samples": 347,
            "test_accuracy": 0.9436,
            "test_macro_f1": 0.9447,
            "correct_samples": 770,
            "test_samples": 816,
        }

    @application.get("/api/models/{model_name}", response_class=FileResponse)
    def mediapipe_model(model_name: str) -> Path:
        model_paths = {
            "hand_landmarker.task": (
                PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
            ),
            "pose_landmarker_lite.task": (
                PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_lite.task"
            ),
        }
        path = model_paths.get(model_name)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="MediaPipe model not found.")
        return path

    @application.get("/api/reference/signs")
    def reference_signs() -> dict[str, object]:
        try:
            entries, _ = reference_index()
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"count": len(entries), "signs": sorted(entries)}

    @application.get("/api/reference/{label}")
    def reference_details(label: str) -> dict[str, object]:
        try:
            entries, resolver = reference_index()
            canonical = canonicalize_label(label)
            relative_path = entries.get(canonical)
            if relative_path is None:
                raise HTTPException(status_code=404, detail="Reference sign not found.")
            video_path = resolver.resolve(relative_path.as_posix())
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "label": canonical,
            "source": "AI4Bharat INCLUDE official test split",
            "filename": video_path.name,
            "file_path": str(video_path),
            "video_url": f"/api/reference/{canonical}/video",
        }

    @application.get("/api/reference/{label}/video", response_class=FileResponse)
    def reference_video(label: str) -> Path:
        entries, resolver = reference_index()
        canonical = canonicalize_label(label)
        relative_path = entries.get(canonical)
        if relative_path is None:
            raise HTTPException(status_code=404, detail="Reference sign not found.")
        return resolver.resolve(relative_path.as_posix())

    @application.post("/api/predict/video")
    async def predict_video(video: Annotated[UploadFile, File()]) -> dict[str, object]:
        suffix = Path(video.filename or "capture.webm").suffix.casefold()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Use MOV, MP4, AVI, MKV, or WEBM video.")

        temporary_path: Path | None = None
        size = 0
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Video exceeds the 200 MB limit.",
                        )
                    temporary.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="The uploaded video is empty.")
            return await run_in_threadpool(inference.predict_video, temporary_path)
        except HTTPException:
            raise
        except Exception as error:
            import logging

            logging.getLogger(__name__).exception("Video prediction failed")
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await video.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @application.post("/api/predict/landmarks")
    async def predict_landmarks(sequence: BrowserLandmarkSequence) -> dict[str, object]:
        try:
            frames = [frame.model_dump() for frame in sequence.frames]
            return await run_in_threadpool(inference.predict_landmarks, frames)
        except Exception as error:
            import logging

            logging.getLogger(__name__).exception("Landmark prediction failed")
            raise HTTPException(status_code=422, detail=str(error)) from error

    application.state.inference_service = inference
    return application


app = create_app()
