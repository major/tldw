"""Tests for the YouTube deep-link and timestamp helpers in tldw.youtube."""

from __future__ import annotations

import pytest

from tldw.youtube import format_timestamp, youtube_deep_link


@pytest.mark.parametrize(
    ("video_id", "seconds", "expected"),
    [
        pytest.param(
            "dQw4w9WgXcQ", 0, "https://youtu.be/dQw4w9WgXcQ?t=0", id="zero"
        ),
        pytest.param(
            "dQw4w9WgXcQ", 45, "https://youtu.be/dQw4w9WgXcQ?t=45", id="sub-minute"
        ),
        pytest.param(
            "dQw4w9WgXcQ", 60, "https://youtu.be/dQw4w9WgXcQ?t=60", id="minute-boundary"
        ),
        pytest.param(
            "dQw4w9WgXcQ", 3600, "https://youtu.be/dQw4w9WgXcQ?t=3600", id="hour-boundary"
        ),
        pytest.param(
            "dQw4w9WgXcQ", 987654, "https://youtu.be/dQw4w9WgXcQ?t=987654", id="large-value"
        ),
        pytest.param(
            "dQw4w9WgXcQ", 754.9, "https://youtu.be/dQw4w9WgXcQ?t=754", id="float-truncates"
        ),
        pytest.param(
            "ab-cd_ef-12", 12, "https://youtu.be/ab-cd_ef-12?t=12", id="id-special-chars"
        ),
    ],
)
def test_youtube_deep_link(video_id: str, seconds: int | float, expected: str) -> None:
    """Deep links use the short youtu.be form and truncate fractional seconds."""
    # Arrange
    # video_id, seconds, and expected come from the parametrized matrix above.

    # Act
    link = youtube_deep_link(video_id, seconds)

    # Assert
    assert link == expected
    # The short URL form starts the time parameter with '?', never '&'.
    assert "?t=" in link
    assert "&t=" not in link


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(0, "0:00", id="zero"),
        pytest.param(59, "0:59", id="sub-minute"),
        pytest.param(60, "1:00", id="minute-boundary"),
        pytest.param(754, "12:34", id="multi-minute"),
        pytest.param(3600, "1:00:00", id="hour-boundary"),
        pytest.param(3661, "1:01:01", id="hour-and-second"),
        pytest.param(86399, "23:59:59", id="end-of-day"),
    ],
)
def test_format_timestamp(seconds: int, expected: str) -> None:
    """Timestamps use a compact H:MM:SS form with both fields padded."""
    # Arrange
    # seconds and expected come from the parametrized matrix above.

    # Act
    stamp = format_timestamp(seconds)

    # Assert
    assert stamp == expected


def test_format_timestamp_hours_are_not_padded() -> None:
    """The hours field stays unpadded while minutes and seconds keep two digits."""
    # Arrange
    # A value with a one-hour duration and single-digit minute and second.

    # Act
    stamp = format_timestamp(3661)

    # Assert
    assert stamp == "1:01:01"
    assert stamp != "1:1:1"
    assert stamp != "01:01:01"


def test_format_timestamp_minute_and_second_padding() -> None:
    """Minutes and seconds below ten are zero-padded to two digits."""
    # Arrange
    # 65 seconds is one minute and five seconds.

    # Act
    stamp = format_timestamp(65)

    # Assert
    assert stamp == "1:05"
    assert stamp != "1:5"


def test_format_timestamp_accepts_float_seconds() -> None:
    """Fractional seconds truncate to whole seconds before formatting."""
    # Arrange
    # 754.9 seconds should be treated as 754 seconds.

    # Act
    stamp = format_timestamp(754.9)

    # Assert
    assert stamp == "12:34"
