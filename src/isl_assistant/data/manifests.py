"""Load official INCLUDE and INCLUDE-50 manifests into typed samples."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

from .path_resolver import VideoPathResolver
from .schemas import DatasetSplit, VideoSample

CLASS_NUMBER_PREFIX = re.compile(r"^\d+\.\s*")
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def canonicalize_label(value: str) -> str:
    """Convert a class-directory name into the label-map naming format."""

    without_number = CLASS_NUMBER_PREFIX.sub("", value.strip())
    return NON_ALPHANUMERIC.sub("", without_number.casefold())


def load_label_map(path: str | Path) -> dict[str, int]:
    label_map_path = Path(path).expanduser()
    with label_map_path.open("r", encoding="utf-8") as label_file:
        raw_map = json.load(label_file)

    if not isinstance(raw_map, dict):
        raise ValueError(f"Label map must contain a JSON object: {label_map_path}")

    label_map: dict[str, int] = {}
    for label, label_id in raw_map.items():
        canonical_label = canonicalize_label(str(label))
        if canonical_label in label_map:
            raise ValueError(f"Duplicate canonical label in label map: {canonical_label}")
        label_map[canonical_label] = int(label_id)

    expected_ids = set(range(len(label_map)))
    actual_ids = set(label_map.values())
    if actual_ids != expected_ids:
        raise ValueError("Label-map IDs must be unique and contiguous from zero")

    return label_map


def exclude_and_reindex_labels(
    label_map: Mapping[str, int], excluded_labels: Iterable[str]
) -> dict[str, int]:
    """Remove excluded classes and make the remaining benchmark IDs contiguous."""

    excluded = {canonicalize_label(label) for label in excluded_labels}
    retained = sorted(
        ((label, label_id) for label, label_id in label_map.items() if label not in excluded),
        key=lambda item: item[1],
    )
    return {label: new_id for new_id, (label, _) in enumerate(retained)}


def read_manifest(path: str | Path) -> list[PurePosixPath]:
    manifest_path = Path(path).expanduser()
    entries: list[PurePosixPath] = []

    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue

        relative_path = PurePosixPath(value.replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Invalid relative path in {manifest_path} at line {line_number}: {value}"
            )
        if len(relative_path.parts) < 3:
            raise ValueError(
                f"Expected category/class/video path in {manifest_path} "
                f"at line {line_number}: {value}"
            )
        entries.append(relative_path)

    if len(entries) != len(set(entries)):
        raise ValueError(f"Manifest contains duplicate paths: {manifest_path}")

    return entries


def label_from_path(relative_path: PurePosixPath) -> str:
    """Return the canonical label represented by the class directory."""

    class_directory = relative_path.parent
    # INCLUDE stores some recordings as category/class/Extra/video.
    while class_directory.name.casefold() == "extra":
        class_directory = class_directory.parent

    return canonicalize_label(class_directory.name)


def load_split_samples(
    manifest_path: str | Path,
    split: DatasetSplit,
    label_map: Mapping[str, int],
    resolver: VideoPathResolver,
    excluded_labels: Iterable[str] = (),
    excluded_paths: Iterable[PurePosixPath] = (),
) -> list[VideoSample]:
    samples: list[VideoSample] = []
    labels_to_exclude = {canonicalize_label(label) for label in excluded_labels}
    paths_to_exclude = set(excluded_paths)

    for relative_path in read_manifest(manifest_path):
        label = label_from_path(relative_path)
        if label in labels_to_exclude or relative_path in paths_to_exclude:
            continue
        if label not in label_map:
            raise ValueError(
                f"Manifest label '{label}' is absent from the selected label map: "
                f"{relative_path}"
            )

        samples.append(
            VideoSample(
                relative_path=relative_path,
                absolute_path=resolver.resolve(relative_path),
                label=label,
                label_id=label_map[label],
                split=split,
            )
        )

    return samples


def load_dataset_samples(
    manifest_root: str | Path,
    label_map_path: str | Path,
    raw_root: str | Path,
    subset: str = "include50",
    splits: Iterable[DatasetSplit] = tuple(DatasetSplit),
    excluded_labels: Iterable[str] = (),
    excluded_paths_by_split: Mapping[DatasetSplit, Iterable[PurePosixPath]] | None = None,
) -> list[VideoSample]:
    """Load selected official splits and resolve every entry to a local video."""

    if subset not in {"include", "include50"}:
        raise ValueError("subset must be either 'include' or 'include50'")

    manifest_directory = Path(manifest_root).expanduser()
    label_map = exclude_and_reindex_labels(
        load_label_map(label_map_path), excluded_labels
    )
    resolver = VideoPathResolver(raw_root)
    samples: list[VideoSample] = []

    for split in splits:
        manifest_path = manifest_directory / f"{subset}_{split.value}.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
        samples.extend(
            load_split_samples(
                manifest_path=manifest_path,
                split=split,
                label_map=label_map,
                resolver=resolver,
                excluded_labels=excluded_labels,
                excluded_paths=(excluded_paths_by_split or {}).get(split, ()),
            )
        )

    return samples
