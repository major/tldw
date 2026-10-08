"""Tests for the YouTube Atom feed parser in tldw.feed."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tldw.feed import is_short_url, parse_atom


@pytest.mark.parametrize(
    ("fixture_name", "expected_count"),
    [
        ("real_atom_payload", 15),
        ("multi_entry_payload", 2),
    ],
)
def test_valid_feed_returns_one_entry_per_video(
    request: pytest.FixtureRequest, fixture_name: str, expected_count: int
) -> None:
    """A well-formed feed yields exactly one VideoEntry per <entry> element."""
    # Arrange
    payload: bytes = request.getfixturevalue(fixture_name)

    # Act
    entries = parse_atom(payload)

    # Assert
    assert len(entries) == expected_count


def test_real_payload_first_entry_fields(real_atom_payload: bytes) -> None:
    """The captured feed exposes the first video's identifying fields."""
    # Arrange
    expected_video_id = "lTQHImoeuEY"
    expected_channel_name = "Google for Developers"

    # Act
    entries = parse_atom(real_atom_payload)

    # Assert
    first = entries[0]
    assert first.video_id == expected_video_id
    assert first.channel_id == "UC_x5XG1OV2P6uZZ5FSM9Ttw"
    assert first.channel_name == expected_channel_name
    assert first.url.startswith("https://www.youtube.com/watch?v=")
    assert isinstance(first.title, str)
    assert first.title != ""


def test_multi_entry_payload_titles(multi_entry_payload: bytes) -> None:
    """Both fixture entries parse with their own video id and title."""
    # Arrange
    expected_ids = {"v_Fixture1AAAAA", "v_Fixture2AAAAA"}
    expected_titles = ["Fixture Video One", "Fixture Video Two"]

    # Act
    entries = parse_atom(multi_entry_payload)

    # Assert
    assert len(entries) == 2
    assert {entry.video_id for entry in entries} == expected_ids
    assert [entry.title for entry in entries] == expected_titles


def test_empty_feed_returns_empty_list(empty_atom_payload: bytes) -> None:
    """A feed with no entries parses to an empty list."""
    # Arrange
    payload = empty_atom_payload

    # Act
    entries = parse_atom(payload)

    # Assert
    assert entries == []


def test_entry_missing_link_raises_value_error(
    missing_link_atom_payload: bytes,
) -> None:
    """An entry without any <link> element is rejected with ValueError."""
    # Arrange
    payload = missing_link_atom_payload

    # Act / Assert
    with pytest.raises(ValueError):
        parse_atom(payload)


def test_entry_falls_back_to_media_title(
    media_title_atom_payload: bytes,
) -> None:
    """An empty Atom <title> falls back to the media:title text."""
    # Arrange
    payload = media_title_atom_payload

    # Act
    entries = parse_atom(payload)

    # Assert
    assert entries[0].title == "From Media Title"


def test_malformed_xml_raises_parse_error(malformed_atom_payload: bytes) -> None:
    """Malformed XML surfaces the parser's own ParseError."""
    # Arrange
    payload = malformed_atom_payload

    # Act / Assert
    with pytest.raises(ET.ParseError):
        parse_atom(payload)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ", id="desktop-shorts"
        ),
        pytest.param(
            "https://m.youtube.com/shorts/dQw4w9WgXcQ", id="mobile-shorts"
        ),
        pytest.param(
            "https://music.youtube.com/shorts/dQw4w9WgXcQ", id="music-shorts"
        ),
    ],
)
def test_is_short_url_recognizes_short_shapes(url: str) -> None:
    """Every YouTube Shorts URL shape is detected as a short."""
    # Arrange
    # The parametrized list covers the desktop, mobile, and music subdomains.

    # Act
    result = is_short_url(url)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", id="desktop-watch"
        ),
        pytest.param(
            "https://youtu.be/dQw4w9WgXcQ", id="youtu-be"
        ),
        pytest.param(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42", id="watch-with-t"
        ),
        pytest.param("", id="empty"),
    ],
)
def test_is_short_url_rejects_non_shorts(url: str) -> None:
    """Watch, share, and empty URLs are not shorts."""
    # Arrange
    # The parametrized list covers the canonical watch URL, the share form,
    # a query-bearing watch URL, and an empty string.

    # Act
    result = is_short_url(url)

    # Assert
    assert result is False
