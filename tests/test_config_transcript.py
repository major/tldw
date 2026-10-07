"""Tests for the transcript-pipeline settings in tldw.config.

These tests cover only the fields added for the transcript pipeline: the Discord
webhook URL, the queue and transcript paths, the line count, the worker pacing
knobs, the give-up window, the optional cookies file, and the language list.
They pin the defaults, the env override path, and the pydantic validation that
rejects a malformed value at startup. The channel-id behavior stays in
``test_config.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw.config import Settings

# Every TLDW_ env var the model reads, cleared before each test so a developer's
# shell or a CI job cannot leak a value into the assertions.
_TLDW_ENV_VARS = (
    "TLDW_CALLBACK_URL",
    "TLDW_CHANNELS_FILE",
    "TLDW_CHANNEL_IDS",
    "TLDW_HUB_SECRET",
    "TLDW_DISCORD_WEBHOOK_URL",
    "TLDW_QUEUE_FILE",
    "TLDW_TRANSCRIPT_DIR",
    "TLDW_TRANSCRIPT_LINES",
    "TLDW_POLL_BASE_SECONDS",
    "TLDW_POLL_CAP_SECONDS",
    "TLDW_GIVEUP_SECONDS",
    "TLDW_YTDLP_COOKIES_FILE",
    "TLDW_TRANSCRIPT_LANGS",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every TLDW_ env var before the test body runs."""
    for name in _TLDW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults_for_new_fields(clean_env: None) -> None:
    """With no env set every new field keeps its documented default."""
    # Arrange
    # clean_env has already cleared the TLDW_ env vars.

    # Act
    settings = Settings()

    # Assert
    assert settings.discord_webhook_url is None
    assert settings.queue_file == Path("queue.sqlite3")
    assert settings.transcript_dir == Path("transcripts")
    assert settings.transcript_lines == 10
    assert settings.poll_base_seconds == 600.0
    assert settings.poll_cap_seconds == 3600.0
    assert settings.giveup_seconds == 172800.0
    assert settings.ytdlp_cookies_file is None
    assert settings.transcript_langs == ["en", "en-orig"]


def test_settings_normalizes_empty_webhook_to_none(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """An empty TLDW_DISCORD_WEBHOOK_URL is treated as unset."""
    # Arrange
    monkeypatch.setenv("TLDW_DISCORD_WEBHOOK_URL", "")

    # Act
    settings = Settings()

    # Assert
    assert settings.discord_webhook_url is None


def test_settings_normalizes_whitespace_webhook_to_none(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A whitespace-only webhook URL is treated as unset."""
    # Arrange
    monkeypatch.setenv("TLDW_DISCORD_WEBHOOK_URL", "   ")

    # Act
    settings = Settings()

    # Assert
    assert settings.discord_webhook_url is None


@pytest.mark.parametrize(
    "env_name, raw_value, attr, expected",
    [
        pytest.param(
            "TLDW_DISCORD_WEBHOOK_URL",
            "https://discord.com/api/webhooks/123/abc",
            "discord_webhook_url",
            "https://discord.com/api/webhooks/123/abc",
            id="discord-webhook-url",
        ),
        pytest.param(
            "TLDW_QUEUE_FILE",
            "/tmp/foo/queue.sqlite3",
            "queue_file",
            Path("/tmp/foo/queue.sqlite3"),
            id="queue-file",
        ),
        pytest.param(
            "TLDW_TRANSCRIPT_DIR",
            "/tmp/foo/transcripts",
            "transcript_dir",
            Path("/tmp/foo/transcripts"),
            id="transcript-dir",
        ),
        pytest.param(
            "TLDW_TRANSCRIPT_LINES",
            "20",
            "transcript_lines",
            20,
            id="transcript-lines",
        ),
        pytest.param(
            "TLDW_POLL_BASE_SECONDS",
            "300.5",
            "poll_base_seconds",
            300.5,
            id="poll-base-seconds",
        ),
        pytest.param(
            "TLDW_POLL_CAP_SECONDS",
            "1800",
            "poll_cap_seconds",
            1800.0,
            id="poll-cap-seconds",
        ),
        pytest.param(
            "TLDW_GIVEUP_SECONDS",
            "86400.0",
            "giveup_seconds",
            86400.0,
            id="giveup-seconds",
        ),
        pytest.param(
            "TLDW_YTDLP_COOKIES_FILE",
            "/etc/yt-cookies.txt",
            "ytdlp_cookies_file",
            Path("/etc/yt-cookies.txt"),
            id="ytdlp-cookies-file",
        ),
    ],
)
def test_settings_reads_field_from_env(
    monkeypatch: pytest.MonkeyPatch,
    clean_env: None,
    env_name: str,
    raw_value: str,
    attr: str,
    expected: object,
) -> None:
    """Each TLDW_ env override lands on its field with the right type."""
    # Arrange
    monkeypatch.setenv(env_name, raw_value)

    # Act
    settings = Settings()

    # Assert
    assert getattr(settings, attr) == expected


def test_settings_rejects_non_integer_lines(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A non-numeric line count fails with a pydantic ValidationError."""
    # Arrange
    monkeypatch.setenv("TLDW_TRANSCRIPT_LINES", "banana")

    # Act / Assert
    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_negative_giveup(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A negative give-up window fails with a pydantic ValidationError."""
    # Arrange
    monkeypatch.setenv("TLDW_GIVEUP_SECONDS", "-1")

    # Act / Assert
    with pytest.raises(ValidationError):
        Settings()


def test_settings_transcript_langs_accepts_json_from_env(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """TLDW_TRANSCRIPT_LANGS is parsed as a JSON list."""
    # Arrange
    monkeypatch.setenv("TLDW_TRANSCRIPT_LANGS", '["de","fr"]')

    # Act
    settings = Settings()

    # Assert
    assert settings.transcript_langs == ["de", "fr"]
