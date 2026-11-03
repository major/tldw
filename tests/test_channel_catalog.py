"""Regression coverage for the shipped Stage Analysis channel configuration."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("filename", "keys"),
    [
        ("channels.json", ("channel_ids",)),
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
