import hashlib
from pathlib import Path

import pytest

from isl_assistant.vision.landmarks.model_config import load_model_config


def write_model_config(path: Path, hand_hash: str, pose_hash: str) -> None:
    path.write_text(
        "\n".join(
            (
                "[hand]",
                'path = "models/hand.task"',
                'url = "https://example.test/hand.task"',
                f'sha256 = "{hand_hash}"',
                'size_bytes = 4',
                "",
                "[pose]",
                'variant = "lite"',
                'path = "models/pose.task"',
                'url = "https://example.test/pose.task"',
                f'sha256 = "{pose_hash}"',
                'size_bytes = 4',
            )
        ),
        encoding="utf-8",
    )


def test_loads_and_verifies_model_assets(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    hand = models / "hand.task"
    pose = models / "pose.task"
    hand.write_bytes(b"hand")
    pose.write_bytes(b"pose")
    config_path = tmp_path / "models.toml"
    write_model_config(
        config_path,
        hashlib.sha256(b"hand").hexdigest(),
        hashlib.sha256(b"pose").hexdigest(),
    )

    config = load_model_config(config_path, project_root=tmp_path)

    assert config.hand.path == hand.resolve()
    assert config.pose.path == pose.resolve()
    assert config.pose.variant == "lite"
    settings = config.detector_settings()
    assert settings.hand_model_path == hand.resolve()
    assert settings.pose_model_path == pose.resolve()


def test_rejects_modified_model_file(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "hand.task").write_bytes(b"bad!")
    (models / "pose.task").write_bytes(b"pose")
    config_path = tmp_path / "models.toml"
    write_model_config(
        config_path,
        hashlib.sha256(b"hand").hexdigest(),
        hashlib.sha256(b"pose").hexdigest(),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_model_config(config_path, project_root=tmp_path)
