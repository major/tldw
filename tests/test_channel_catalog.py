"""Regression coverage for the shipped Stage Analysis channel configuration."""

from pathlib import Path

import pytest
import yaml

from tldw.config import Settings, resolve_channel_ids


@pytest.mark.parametrize(
    ("filename", "keys"),
    [
        ("channels.yaml", ("channels",)),
        ("topics.example.yaml", ("topics", "default", "channels")),
    ],
)
def test_stage_analysis_in_channel_configs(filename: str, keys: tuple[str, ...]) -> None:
    """Each shipped channel list includes Stage Analysis exactly once."""
    # Arrange
    path = Path(__file__).resolve().parents[1] / filename
    config = yaml.safe_load(path.read_text())

    # Act
    for key in keys:
        config = config[key]
    channel_ids = [entry if isinstance(entry, str) else entry["channel_id"] for entry in config]

    # Assert
    assert channel_ids.count("UCKzgts1fFoDla4ozJSn1eZQ") == 1


def test_runtime_uses_shipped_yaml_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The editable YAML catalog is the actual default subscription list."""
    # Arrange
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    catalog = yaml.safe_load((root / "channels.yaml").read_text())
    expected = [entry["channel_id"] for entry in catalog["channels"]]

    # Act
    settings = Settings()
    channel_ids = resolve_channel_ids(settings)

    # Assert
    assert settings.channel_ids_file == Path("channels.yaml")
    assert channel_ids == expected
    assert "UCMno7bbQKigk6RxiO0uv78g" not in channel_ids


def test_container_ships_only_yaml_channel_list() -> None:
    """Both container stages ship the YAML catalog, never a stale JSON copy."""
    # Arrange
    root = Path(__file__).resolve().parents[1]
    text = (root / "Containerfile").read_text()

    # Act
    copies = [line for line in text.splitlines() if line.startswith("COPY")]

    # Assert
    assert sum("channels.yaml" in line for line in copies) == 2
    assert "channels.json" not in text
    assert not (root / "channels.json").exists()
