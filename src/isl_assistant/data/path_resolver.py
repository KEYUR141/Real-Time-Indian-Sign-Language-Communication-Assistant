"""Resolve official manifest entries to locally extracted video files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


SUPPORTED_VIDEO_EXTENSIONS = (".MOV", ".MP4", ".AVI", ".MKV", ".WEBM")


class VideoPathResolver:
    """Resolve manifest paths while handling extension and case differences."""

    def __init__(
        self,
        raw_root: str | Path,
        supported_extensions: tuple[str, ...] = SUPPORTED_VIDEO_EXTENSIONS,
    ) -> None:
        self.raw_root = Path(raw_root).expanduser().resolve()
        self.supported_extensions = tuple(
            extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
            for extension in supported_extensions
        )

    def resolve(self, manifest_path: str | PurePosixPath) -> Path:
        relative_path = PurePosixPath(str(manifest_path).strip().replace("\\", "/"))
        self._validate_relative_path(relative_path)

        parent = self.raw_root.joinpath(*relative_path.parent.parts)
        requested = parent / relative_path.name
        self._ensure_inside_raw_root(requested)

        if requested.is_file():
            return requested

        if not parent.is_dir():
            raise FileNotFoundError(
                f"Video directory does not exist for manifest entry: {relative_path}"
            )

        expected_stem = relative_path.stem.casefold()
        matches = [
            candidate
            for candidate in parent.iterdir()
            if candidate.is_file()
            and candidate.stem.casefold() == expected_stem
            and candidate.suffix.casefold() in self.supported_extensions
        ]

        if len(matches) == 1:
            return matches[0].resolve()
        if not matches:
            raise FileNotFoundError(f"No local video matches manifest entry: {relative_path}")

        joined_matches = ", ".join(str(match) for match in sorted(matches))
        raise RuntimeError(f"Multiple local videos match {relative_path}: {joined_matches}")

    @staticmethod
    def _validate_relative_path(relative_path: PurePosixPath) -> None:
        if relative_path.is_absolute():
            raise ValueError(f"Manifest path must be relative: {relative_path}")
        if not relative_path.name or relative_path.name in {".", ".."}:
            raise ValueError(f"Invalid manifest video path: {relative_path}")
        if ".." in relative_path.parts:
            raise ValueError(f"Manifest path cannot contain '..': {relative_path}")

    def _ensure_inside_raw_root(self, candidate: Path) -> None:
        try:
            candidate.resolve().relative_to(self.raw_root)
        except ValueError as error:
            raise ValueError(f"Resolved path escapes raw dataset root: {candidate}") from error

