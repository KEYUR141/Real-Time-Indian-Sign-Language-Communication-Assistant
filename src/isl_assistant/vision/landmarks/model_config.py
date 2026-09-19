"""Load and verify MediaPipe model assets from TOML configuration."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mediapipe_detector import DetectorSettings


@dataclass(frozen=True, slots=True)
class ModelAsset:
    name: str
    path: Path
    url: str
    sha256: str
    size_bytes: int
    variant: str | None = None

    def verify(self, *, verify_hash: bool = True) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.name} model file does not exist: {self.path}")
        if self.path.suffix.casefold() != ".task":
            raise ValueError(f"{self.name} model must use the .task extension: {self.path}")

        actual_size = self.path.stat().st_size
        if actual_size != self.size_bytes:
            raise ValueError(
                f"{self.name} model size mismatch: expected {self.size_bytes}, "
                f"found {actual_size}"
            )

        if verify_hash:
            actual_hash = _sha256(self.path)
            if actual_hash != self.sha256:
                raise ValueError(
                    f"{self.name} model SHA-256 mismatch: expected {self.sha256}, "
                    f"found {actual_hash}"
                )


@dataclass(frozen=True, slots=True)
class MediaPipeModelConfig:
    hand: ModelAsset
    pose: ModelAsset

    def verify(self, *, verify_hash: bool = True) -> None:
        self.hand.verify(verify_hash=verify_hash)
        self.pose.verify(verify_hash=verify_hash)

    def detector_settings(
        self,
        *,
        maximum_hands: int = 2,
        minimum_detection_confidence: float = 0.5,
        minimum_presence_confidence: float = 0.5,
        minimum_tracking_confidence: float = 0.5,
        swap_handedness_labels: bool = False,
    ) -> DetectorSettings:
        return DetectorSettings(
            hand_model_path=self.hand.path,
            pose_model_path=self.pose.path,
            maximum_hands=maximum_hands,
            minimum_detection_confidence=minimum_detection_confidence,
            minimum_presence_confidence=minimum_presence_confidence,
            minimum_tracking_confidence=minimum_tracking_confidence,
            swap_handedness_labels=swap_handedness_labels,
        )


def load_model_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify: bool = True,
    verify_hash: bool = True,
) -> MediaPipeModelConfig:
    """Load model metadata and resolve project-relative model paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as config_file:
        values = tomllib.load(config_file)

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else config_path.parent.parent
    )
    config = MediaPipeModelConfig(
        hand=_parse_asset("hand", values.get("hand", {}), root),
        pose=_parse_asset("pose", values.get("pose", {}), root),
    )

    if verify:
        config.verify(verify_hash=verify_hash)
    return config


def _parse_asset(name: str, values: dict[str, Any], project_root: Path) -> ModelAsset:
    required = ("path", "url", "sha256", "size_bytes")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing [{name}] model values: {', '.join(missing)}")

    configured_path = Path(str(values["path"])).expanduser()
    resolved_path = (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (project_root / configured_path).resolve()
    )
    sha256 = str(values["sha256"]).casefold()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError(f"[{name}].sha256 must be a 64-character hexadecimal digest")

    size_bytes = int(values["size_bytes"])
    if size_bytes <= 0:
        raise ValueError(f"[{name}].size_bytes must be greater than zero")

    return ModelAsset(
        name=name,
        path=resolved_path,
        url=str(values["url"]),
        sha256=sha256,
        size_bytes=size_bytes,
        variant=str(values["variant"]) if "variant" in values else None,
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

