"""Configuration loading for the landmark extraction pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root: Path
    subset: str
    raw_directory: Path
    manifest_directory: Path
    label_map: Path
    excluded_labels: tuple[str, ...] = ()

    @property
    def raw_root(self) -> Path:
        return self.root / self.raw_directory

    @property
    def manifest_root(self) -> Path:
        return self.root / self.manifest_directory

    @property
    def label_map_path(self) -> Path:
        return self.root / self.label_map


@dataclass(frozen=True, slots=True)
class LandmarkConfig:
    sequence_length: int
    extract_hands: bool
    extract_pose: bool
    extract_face: bool
    maximum_hands: int
    minimum_detection_confidence: float
    minimum_tracking_confidence: float


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: Path
    overwrite: bool


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    dataset: DatasetConfig
    landmarks: LandmarkConfig
    output: OutputConfig


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    try:
        return section[key]
    except KeyError as error:
        raise ValueError(f"Missing configuration value: {section_name}.{key}") from error


def load_config(path: str | Path) -> ApplicationConfig:
    """Load and validate the landmark pipeline's TOML configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as config_file:
        values = tomllib.load(config_file)

    dataset_values = values.get("dataset", {})
    landmark_values = values.get("landmarks", {})
    output_values = values.get("output", {})

    dataset = DatasetConfig(
        root=Path(_required(dataset_values, "root", "dataset")).expanduser(),
        subset=str(_required(dataset_values, "subset", "dataset")),
        raw_directory=Path(_required(dataset_values, "raw_directory", "dataset")),
        manifest_directory=Path(
            _required(dataset_values, "manifest_directory", "dataset")
        ),
        label_map=Path(_required(dataset_values, "label_map", "dataset")),
        excluded_labels=tuple(
            str(label) for label in dataset_values.get("excluded_labels", ())
        ),
    )
    landmarks = LandmarkConfig(
        sequence_length=int(_required(landmark_values, "sequence_length", "landmarks")),
        extract_hands=bool(_required(landmark_values, "extract_hands", "landmarks")),
        extract_pose=bool(_required(landmark_values, "extract_pose", "landmarks")),
        extract_face=bool(_required(landmark_values, "extract_face", "landmarks")),
        maximum_hands=int(_required(landmark_values, "maximum_hands", "landmarks")),
        minimum_detection_confidence=float(
            _required(landmark_values, "minimum_detection_confidence", "landmarks")
        ),
        minimum_tracking_confidence=float(
            _required(landmark_values, "minimum_tracking_confidence", "landmarks")
        ),
    )
    output = OutputConfig(
        root=Path(_required(output_values, "root", "output")).expanduser(),
        overwrite=bool(_required(output_values, "overwrite", "output")),
    )

    if landmarks.sequence_length <= 0:
        raise ValueError("landmarks.sequence_length must be greater than zero")
    if landmarks.maximum_hands not in {1, 2}:
        raise ValueError("landmarks.maximum_hands must be either 1 or 2")
    for field_name, confidence in (
        ("minimum_detection_confidence", landmarks.minimum_detection_confidence),
        ("minimum_tracking_confidence", landmarks.minimum_tracking_confidence),
    ):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"landmarks.{field_name} must be between 0 and 1")

    return ApplicationConfig(dataset=dataset, landmarks=landmarks, output=output)
