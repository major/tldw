"""Tests for the console line renderer in tldw.renderer."""

from __future__ import annotations

import pytest

from tldw.feed import VideoEntry
from tldw.renderer import format_video_line


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            VideoEntry(
                video_id="lTQHImoeuEY",
                channel_id="UC_x5XG1OV2P6uZZ5FSM9Ttw",
                title="Top 3 new model launches at Gemini Audio at Night",
                url="https://www.youtube.com/watch?v=lTQHImoeuEY",
                channel_name="Google for Developers",
                published="2024-09-24T15:00:00+00:00",
                updated="2024-09-24T15:00:00+00:00",
            ),
            "[YouTube] Google for Developers: Top 3 new model launches at "
            "Gemini Audio at Night "
            "(https://www.youtube.com/watch?v=lTQHImoeuEY)",
            id="full-entry",
        ),
    ],
)
def test_format_video_line_full_entry(entry: VideoEntry, expected: str) -> None:
    """A complete entry renders to the exact one-line console format."""
    # Arrange
    # entry and expected come from the parametrized matrix above.

    # Act
    line = format_video_line(entry)

    # Assert
    assert line == expected


def test_embedded_newlines_are_collapsed_to_spaces() -> None:
    """Newlines and carriage returns inside a title never break the line."""
    # Arrange
    entry = VideoEntry(
        video_id="v_NewlineAAAA",
        channel_id="UC_NEWLINE_AAAAAAAAAAAAAA",
        title="\nhidden\r\nstuff",
        url="https://www.youtube.com/watch?v=v_NewlineAAAA",
        channel_name="Fixture Channel",
        published=None,
        updated=None,
    )

    # Act
    line = format_video_line(entry)

    # Assert
    assert "\n" not in line
    assert "\r" not in line
    assert "hidden stuff" in line


def test_colon_inside_title_is_preserved() -> None:
    """A colon in the title is kept so the title text stays readable."""
    # Arrange
    entry = VideoEntry(
        video_id="v_ColonAAAAAA",
        channel_id="UC_COLON_AAAAAAAAAAAAAAAA",
        title="Foo: bar baz",
        url="https://www.youtube.com/watch?v=v_ColonAAAAAA",
        channel_name="Fixture Channel",
        published=None,
        updated=None,
    )

    # Act
    line = format_video_line(entry)

    # Assert
    assert ": Foo: bar baz (" in line


def test_empty_channel_name_does_not_crash() -> None:
    """An empty channel name still renders a well-formed line."""
    # Arrange
    entry = VideoEntry(
        video_id="v_NoChannelAAA",
        channel_id="",
        title="Solo Video",
        url="https://www.youtube.com/watch?v=v_NoChannelAAA",
        channel_name="",
        published=None,
        updated=None,
    )

    # Act
    line = format_video_line(entry)

    # Assert
    assert line.startswith("[YouTube] : ")
    assert line.endswith("(https://www.youtube.com/watch?v=v_NoChannelAAA)")


def test_output_is_deterministic() -> None:
    """Calling the renderer twice on one entry returns the same string."""
    # Arrange
    entry = VideoEntry(
        video_id="v_RepeatAAAAA",
        channel_id="UC_REPEAT_AAAAAAAAAAAAA",
        title="Repeatable Title",
        url="https://www.youtube.com/watch?v=v_RepeatAAAAA",
        channel_name="Fixture Channel",
        published=None,
        updated=None,
    )

    # Act
    first = format_video_line(entry)
    second = format_video_line(entry)

    # Assert
    assert first == second
