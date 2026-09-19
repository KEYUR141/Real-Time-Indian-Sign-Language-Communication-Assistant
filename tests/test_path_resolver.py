from pathlib import Path

import pytest

from isl_assistant.data.path_resolver import VideoPathResolver


def test_resolves_manifest_mov_to_extracted_mp4(tmp_path: Path) -> None:
    class_directory = tmp_path / "Greetings" / "48. Hello"
    class_directory.mkdir(parents=True)
    expected = class_directory / "MVI_0089.MP4"
    expected.touch()

    resolver = VideoPathResolver(tmp_path)

    assert resolver.resolve("Greetings/48. Hello/MVI_0089.MOV") == expected.resolve()


def test_rejects_parent_directory_escape(tmp_path: Path) -> None:
    resolver = VideoPathResolver(tmp_path)

    with pytest.raises(ValueError, match="cannot contain"):
        resolver.resolve("../outside.MOV")


def test_missing_video_has_clear_error(tmp_path: Path) -> None:
    (tmp_path / "Greetings" / "48. Hello").mkdir(parents=True)
    resolver = VideoPathResolver(tmp_path)

    with pytest.raises(FileNotFoundError, match="No local video matches"):
        resolver.resolve("Greetings/48. Hello/MVI_0089.MOV")

