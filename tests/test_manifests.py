import json
from pathlib import Path, PurePosixPath

from isl_assistant.data.manifests import (
    canonicalize_label,
    exclude_and_reindex_labels,
    label_from_path,
    load_split_samples,
)
from isl_assistant.data.path_resolver import VideoPathResolver
from isl_assistant.data.schemas import DatasetSplit


def test_canonicalize_label_matches_include50_format() -> None:
    assert canonicalize_label("48. Good Morning") == "goodmorning"
    assert canonicalize_label("42. T-Shirt") == "tshirt"
    assert canonicalize_label("39. Store or Shop") == "storeorshop"


def test_excluded_label_map_is_reindexed_contiguously() -> None:
    labels = {"first": 0, "secondnumber": 1, "third": 2}

    assert exclude_and_reindex_labels(labels, ("Second (Number)",)) == {
        "first": 0,
        "third": 1,
    }


def test_label_from_path_ignores_nested_extra_directory() -> None:
    path = PurePosixPath("Adjectives/1. loud/Extra/MVI_1234.MOV")

    assert label_from_path(path) == "loud"


def test_load_split_sample(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    video = raw_root / "Greetings" / "48. Hello" / "MVI_0089.MP4"
    video.parent.mkdir(parents=True)
    video.touch()

    manifest = tmp_path / "include50_train.txt"
    manifest.write_text("Greetings/48. Hello/MVI_0089.MOV\n", encoding="utf-8")

    samples = load_split_samples(
        manifest_path=manifest,
        split=DatasetSplit.TRAIN,
        label_map={"hello": 22},
        resolver=VideoPathResolver(raw_root),
    )

    assert len(samples) == 1
    assert samples[0].absolute_path == video.resolve()
    assert samples[0].label == "hello"
    assert samples[0].label_id == 22
    assert samples[0].split is DatasetSplit.TRAIN


def test_load_split_samples_filters_label_before_resolving_missing_video(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "include_train.txt"
    manifest.write_text(
        "Days_and_Time/Second (Number)/MVI_4647.MOV\n",
        encoding="utf-8",
    )

    samples = load_split_samples(
        manifest_path=manifest,
        split=DatasetSplit.TRAIN,
        label_map={"secondnumber": 0},
        resolver=VideoPathResolver(tmp_path / "missing-raw-root"),
        excluded_labels=("Second (Number)",),
    )

    assert samples == []


def test_load_split_samples_filters_already_processed_manifest_path(
    tmp_path: Path,
) -> None:
    relative_path = PurePosixPath("Greetings/48. Hello/MVI_0089.MOV")
    manifest = tmp_path / "include_train.txt"
    manifest.write_text(f"{relative_path.as_posix()}\n", encoding="utf-8")

    samples = load_split_samples(
        manifest_path=manifest,
        split=DatasetSplit.TRAIN,
        label_map={"hello": 0},
        resolver=VideoPathResolver(tmp_path / "missing-raw-root"),
        excluded_paths=(relative_path,),
    )

    assert samples == []


def test_label_map_fixture_is_valid_json(tmp_path: Path) -> None:
    label_map_path = tmp_path / "labels.json"
    label_map_path.write_text(json.dumps({"hello": 0}), encoding="utf-8")

    assert json.loads(label_map_path.read_text(encoding="utf-8")) == {"hello": 0}
