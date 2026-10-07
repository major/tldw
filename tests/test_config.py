"""Tests for tldw.config.

These tests cover the two configuration sources for the channel list: the
``TLDW_CHANNEL_IDS`` environment variable and the JSON file named by
``TLDW_CHANNELS_FILE``. They also pin down the YouTube channel id validation
policy so malformed ids fail fast with a message the operator can act on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tldw.config import Settings, resolve_channel_ids

# Environment variables the Settings model reads. Cleared before every test so a
# developer's shell or a CI job cannot leak a value into the assertions.
_TLDW_ENV_VARS = (
    "TLDW_CALLBACK_URL",
    "TLDW_CHANNELS_FILE",
    "TLDW_CHANNEL_IDS",
    "TLDW_HUB_SECRET",
)


@pytest.fixture
def settings_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., dict[str, Any]]]:
    """Clear every TLDW_ env var and yield a factory for Settings kwargs.

    The factory returns the overrides it is given so tests can read as if they
    were calling the Settings constructor directly while still starting from a
    clean environment.
    """
    for name in _TLDW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def _kwargs(**overrides: Any) -> dict[str, Any]:
        """Return constructor kwargs for Settings merged with the overrides."""
        return dict(overrides)

    yield _kwargs


def _write_channels_file(path: Path, ids: list[str]) -> None:
    """Write a channels.json file containing the given channel ids."""
    path.write_text(json.dumps({"channel_ids": ids}), encoding="utf-8")


def test_settings_loads_from_env(
    monkeypatch: pytest.MonkeyPatch,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """Settings reads the callback URL and hub secret from TLDW_ env vars."""
    # Arrange
    monkeypatch.setenv("TLDW_CALLBACK_URL", "https://cb.example/cb")
    monkeypatch.setenv("TLDW_HUB_SECRET", "shh")

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.callback_url == "https://cb.example/cb"
    assert settings.hub_secret == "shh"
    assert settings.channel_ids_file == Path("channels.json")
    assert settings.channel_ids_env_override is None


def test_settings_defaults_when_no_env(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """With no env set every optional field keeps its default."""
    # Arrange
    # settings_kwargs has already cleared the TLDW_ env vars.

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.callback_url is None
    assert settings.channel_ids_env_override is None
    assert settings.hub_secret is None
    assert settings.channel_ids_file == Path("channels.json")


def test_resolve_uses_env_csv_when_set(
    tmp_path: Path,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A non-empty env CSV wins over the file and keeps its order."""
    # Arrange
    channels_file = tmp_path / "channels.json"
    _write_channels_file(channels_file, ["UC_FILEAAAAAAAAAAAAAA_xx"])
    settings = Settings(
        **settings_kwargs(
            channel_ids_file=channels_file,
            channel_ids_env_override=(
                "UC_ENVAAAAAAAAAAAAAAA_01,UC_ENVAAAAAAAAAAAAAAA_02"
            ),
        )
    )

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == ["UC_ENVAAAAAAAAAAAAAAA_01", "UC_ENVAAAAAAAAAAAAAAA_02"]


def test_resolve_falls_back_to_file_when_no_env(
    tmp_path: Path,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """With no env override the channel ids come from the JSON file."""
    # Arrange
    channels_file = tmp_path / "channels.json"
    _write_channels_file(
        channels_file,
        ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"],
    )
    settings = Settings(
        **settings_kwargs(
            channel_ids_file=channels_file,
            channel_ids_env_override=None,
        )
    )

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"]


def test_resolve_empty_env_falls_back_to_file(
    tmp_path: Path,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A blank env override is treated as unset and the file is used."""
    # Arrange
    channels_file = tmp_path / "channels.json"
    _write_channels_file(
        channels_file,
        ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"],
    )
    settings = Settings(
        **settings_kwargs(
            channel_ids_file=channels_file,
            channel_ids_env_override="",
        )
    )

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"]


def test_resolve_env_strips_whitespace_and_drops_blanks(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """Whitespace around env ids is trimmed and blank entries are dropped."""
    # Arrange
    settings = Settings(
        **settings_kwargs(
            channel_ids_env_override=(
                "  UC_ENVAAAAAAAAAAAAAAA_01 ,  , UC_ENVAAAAAAAAAAAAAAA_02"
            ),
        )
    )

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == ["UC_ENVAAAAAAAAAAAAAAA_01", "UC_ENVAAAAAAAAAAAAAAA_02"]


def test_resolve_missing_file_warns_and_returns_empty(
    tmp_path: Path,
    settings_kwargs: Callable[..., dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing file logs a warning naming the path and returns no ids."""
    # Arrange
    missing_file = tmp_path / "does-not-exist.json"
    settings = Settings(**settings_kwargs(channel_ids_file=missing_file))
    caplog.set_level(logging.WARNING)

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == []
    assert any(
        str(missing_file) in record.getMessage() for record in caplog.records
    )


def test_resolve_rejects_malformed_env_id(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A malformed id in the env override raises ValueError naming the id."""
    # Arrange
    settings = Settings(**settings_kwargs(channel_ids_env_override="UC_BAD"))

    # Act
    # Assert
    with pytest.raises(ValueError, match="UC_BAD"):
        resolve_channel_ids(settings)


def test_resolve_rejects_malformed_file_id(
    tmp_path: Path,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A malformed id in the JSON file raises ValueError naming the id."""
    # Arrange
    channels_file = tmp_path / "channels.json"
    _write_channels_file(channels_file, ["UC_BAD"])
    settings = Settings(**settings_kwargs(channel_ids_file=channels_file))

    # Act
    # Assert
    with pytest.raises(ValueError, match="UC_BAD"):
        resolve_channel_ids(settings)


@pytest.mark.parametrize(
    "channel_id",
    [
        pytest.param("UC_ywfvIR2JrnMuZt33y7QYQ", id="underscore-prefixed"),
        pytest.param("UCvJZEG5x-DVYZKTz--pS39w", id="hyphen-rich"),
        pytest.param("UC_-_-_-_-_-_-_-_-_-_-AA", id="all-punctuation"),
    ],
)
def test_resolve_accepts_valid_channel_ids(
    channel_id: str,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A well-formed id passes validation and is returned unchanged."""
    # Arrange
    settings = Settings(**settings_kwargs(channel_ids_env_override=channel_id))

    # Act
    result = resolve_channel_ids(settings)

    # Assert
    assert result == [channel_id]


def test_settings_channel_ids_file_is_path(
    monkeypatch: pytest.MonkeyPatch,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """TLDW_CHANNELS_FILE is parsed into a Path."""
    # Arrange
    monkeypatch.setenv("TLDW_CHANNELS_FILE", "/tmp/foo.json")

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.channel_ids_file == Path("/tmp/foo.json")
