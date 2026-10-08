"""Tests for tldw.config.

These tests cover the two configuration sources for the channel list: the
``TLDW_CHANNEL_IDS`` environment variable and the JSON file named by
``TLDW_CHANNELS_FILE``. They also pin down the YouTube channel id validation
policy so malformed ids fail fast with a message the operator can act on.

The audio pipeline settings (``TLDW_AUDIO_*``, ``TLDW_TRANSCRIBE_*``,
``TLDW_FFMPEG_TIMEOUT_SECONDS``) live in the same ``Settings`` model and are
tested in the ``test_audio_settings`` group at the bottom of this file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from helpers import make_channels_file
from tldw.config import Settings, resolve_channel_ids

# Environment variables the Settings model reads. Cleared before every test so a
# developer's shell or a CI job cannot leak a value into the assertions.
_TLDW_ENV_VARS = (
    "TLDW_CALLBACK_URL",
    "TLDW_CHANNELS_FILE",
    "TLDW_CHANNEL_IDS",
    "TLDW_HUB_SECRET",
    "TLDW_AUDIO_DOWNLOAD_DELAY_SECONDS",
    "TLDW_AUDIO_DIR",
    "TLDW_AUDIO_FORMAT",
    "TLDW_AUDIO_BITRATE",
    "TLDW_FFMPEG_TIMEOUT_SECONDS",
    "TLDW_TRANSCRIBE_MODEL",
    "TLDW_TRANSCRIBE_LANGS",
    "TLDW_TRANSCRIBE_TIMEOUT_SECONDS",
    "TLDW_INCLUDE_SHORTS",
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
    make_channels_file(
        channels_file.parent,
        ["UC_FILEAAAAAAAAAAAAAA_xx"],
        name=channels_file.name,
    )
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
    make_channels_file(
        channels_file.parent,
        ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"],
        name=channels_file.name,
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
    make_channels_file(
        channels_file.parent,
        ["UC_FILEAAAAAAAAAAAAAA_01", "UC_FILEAAAAAAAAAAAAAA_02"],
        name=channels_file.name,
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
    make_channels_file(
        channels_file.parent, ["UC_BAD"], name=channels_file.name
    )
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


# ---------------------------------------------------------------------------
# Audio pipeline settings
# ---------------------------------------------------------------------------


def test_audio_settings_defaults(settings_kwargs: Callable[..., dict[str, Any]]) -> None:
    """The audio pipeline defaults match the documented plan."""
    # Arrange
    # settings_kwargs already cleared every audio env var.

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.audio_download_delay_seconds == 300.0
    assert settings.audio_dir == Path("audio")
    assert settings.audio_format == "webm"
    assert settings.audio_bitrate == "32k"
    assert settings.ffmpeg_timeout_seconds == 900.0
    assert settings.transcribe_model == "gpt-transcribe"
    assert settings.transcribe_keywords == []
    assert settings.transcribe_timeout_seconds == 600.0


def test_audio_settings_read_from_env(
    monkeypatch: pytest.MonkeyPatch,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """Every audio env var overrides the corresponding default."""
    # Arrange
    monkeypatch.setenv("TLDW_AUDIO_DOWNLOAD_DELAY_SECONDS", "120")
    monkeypatch.setenv("TLDW_AUDIO_DIR", "/var/cache/audio")
    monkeypatch.setenv("TLDW_AUDIO_FORMAT", "m4a")
    monkeypatch.setenv("TLDW_AUDIO_BITRATE", "48k")
    monkeypatch.setenv("TLDW_FFMPEG_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("TLDW_TRANSCRIBE_MODEL", "gpt-transcribe")
    monkeypatch.setenv("TLDW_TRANSCRIBE_KEYWORDS", '["officer","DUI"]')
    monkeypatch.setenv("TLDW_TRANSCRIBE_TIMEOUT_SECONDS", "120")

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.audio_download_delay_seconds == 120.0
    assert settings.audio_dir == Path("/var/cache/audio")
    assert settings.audio_format == "m4a"
    assert settings.audio_bitrate == "48k"
    assert settings.ffmpeg_timeout_seconds == 300.0
    assert settings.transcribe_model == "gpt-transcribe"
    assert settings.transcribe_keywords == ["officer", "DUI"]
    assert settings.transcribe_timeout_seconds == 120.0


def test_audio_format_rejects_unsupported_container(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A container OpenAI will not decode fails at startup, not at first video."""
    # Arrange
    # Act / Assert
    with pytest.raises(ValidationError, match="audio_format"):
        Settings(**settings_kwargs(audio_format="ogg"))


def test_audio_format_rejects_opus_container(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """``.opus`` is not in OpenAI's accepted list; use ``.webm`` for Opus."""
    # Arrange
    # Act / Assert
    with pytest.raises(ValidationError, match="audio_format"):
        Settings(**settings_kwargs(audio_format="opus"))


@pytest.mark.parametrize(
    "container",
    ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
)
def test_audio_format_accepts_openai_supported_containers(
    container: str,
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """Every container on OpenAI's accepted list passes validation."""
    # Arrange
    # Act
    settings = Settings(**settings_kwargs(audio_format=container))

    # Assert
    assert settings.audio_format == container


def test_audio_download_delay_rejects_negative(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A negative delay would push the row into the past; reject at startup."""
    # Arrange
    # Act / Assert
    with pytest.raises(ValidationError, match="audio_download_delay_seconds"):
        Settings(**settings_kwargs(audio_download_delay_seconds=-1.0))


def test_ffmpeg_timeout_rejects_zero(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A zero timeout would kill every ffmpeg call; require ``gt=0``."""
    # Arrange
    # Act / Assert
    with pytest.raises(ValidationError, match="ffmpeg_timeout_seconds"):
        Settings(**settings_kwargs(ffmpeg_timeout_seconds=0))


def test_transcribe_timeout_rejects_zero(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """A zero timeout would kill every transcribe call; require ``gt=0``."""
    # Arrange
    # Act / Assert
    with pytest.raises(ValidationError, match="transcribe_timeout_seconds"):
        Settings(**settings_kwargs(transcribe_timeout_seconds=0))


# ---------------------------------------------------------------------------
# Shorts filter setting
# ---------------------------------------------------------------------------


def test_include_shorts_defaults_to_false(
    settings_kwargs: Callable[..., dict[str, Any]],
) -> None:
    """Shorts are filtered by default so operators focus on full-length videos."""
    # Arrange
    # settings_kwargs has already cleared TLDW_INCLUDE_SHORTS.

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.include_shorts is False


@pytest.mark.parametrize("raw_value", ["1", "true", "True", "yes"])
def test_include_shorts_reads_truthy_env(
    monkeypatch: pytest.MonkeyPatch,
    settings_kwargs: Callable[..., dict[str, Any]],
    raw_value: str,
) -> None:
    """A truthy TLDW_INCLUDE_SHORTS enables shorts delivery."""
    # Arrange
    monkeypatch.setenv("TLDW_INCLUDE_SHORTS", raw_value)

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.include_shorts is True


@pytest.mark.parametrize("raw_value", ["0", "false", "False", "no", ""])
def test_include_shorts_reads_falsy_env(
    monkeypatch: pytest.MonkeyPatch,
    settings_kwargs: Callable[..., dict[str, Any]],
    raw_value: str,
) -> None:
    """A falsy or blank TLDW_INCLUDE_SHORTS keeps shorts filtered."""
    # Arrange
    monkeypatch.setenv("TLDW_INCLUDE_SHORTS", raw_value)

    # Act
    settings = Settings(**settings_kwargs())

    # Assert
    assert settings.include_shorts is False
