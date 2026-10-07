"""Tests for the timed transcript parsers and the LLM renderer.

``parse_vtt_timed`` and ``parse_srt_timed`` preserve cue start times so the
takeaway feature can point at real cue boundaries. ``render_transcript_for_llm``
merges cues into anchored blocks. The fixtures mirror realistic YouTube
auto-captions: color tags, inline timing tags, repeated lines, a NOTE block, and
a STYLE block. Fixtures are read as plain text; they are not registered in
conftest.py because the existing Atom payload is loaded the same way.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from tldw.transcript import (
    Cue,
    parse_srt_timed,
    parse_vtt_timed,
    render_transcript_for_llm,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Each rendered line begins with a bracketed anchor such as [0:01] or [1:01:01].
_ANCHOR_PATTERN = re.compile(r"^\[(?:\d+:\d{2}|\d+:\d{2}:\d{2})\] ")


def _load(name: str) -> str:
    """Read a fixture file from tests/fixtures as text."""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# parse_vtt_timed
# ---------------------------------------------------------------------------


def test_parse_vtt_timed_fixture_structure() -> None:
    """The VTT fixture yields all ten cues with the expected time span."""
    # Arrange
    vtt = _load("sample_transcript.vtt")

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    assert len(cues) == 10
    assert cues[0].start == pytest.approx(1.0)
    # Cue stores only the start time, so the final cue's end is not on the
    # model. The fixture's last cue starts at 56.0 and ends at 60.0, so assert
    # the start sits near the end of the ~60 second span instead.
    assert cues[-1].start == pytest.approx(56.0)
    assert cues[-1].start >= 55.0


def test_parse_vtt_timed_strips_all_markup() -> None:
    """No angle-bracket markup survives into any cue's text."""
    # Arrange
    vtt = _load("sample_transcript.vtt")

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    for cue in cues:
        assert "<" not in cue.text
        assert ">" not in cue.text
        assert "-->" not in cue.text


def test_parse_vtt_timed_collapses_repeated_line_within_cue() -> None:
    """A line repeated inside one cue is kept only once."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "Repeated line\n"
        "Repeated line\n"
    )

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    assert len(cues) == 1
    assert cues[0].text == "Repeated line"


def test_parse_vtt_timed_fixture_collapses_duplicate() -> None:
    """The fixture's duplicated vacuum line collapses to a single sentence."""
    # Arrange
    vtt = _load("sample_transcript.vtt")

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    vacuum = next(cue for cue in cues if "vacuum improvements" in cue.text)
    assert vacuum.text == "Let's start with vacuum improvements."


def test_parse_vtt_timed_skips_note_and_style_blocks() -> None:
    """NOTE and STYLE blocks never become cues."""
    # Arrange
    vtt = _load("sample_transcript.vtt")

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    assert not any("caption note" in cue.text for cue in cues)
    assert not any("::cue" in cue.text for cue in cues)
    assert not any("STYLE" in cue.text for cue in cues)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "WEBVTT\n",
        "WEBVTT\n\nNOTE just a note\n",
        "WEBVTT\n\nnot-a-timestamp --> 00:00:02.000\nBad cue\n",
        "WEBVTT\n\n00:00:01.000 00:00:02.000\nMissing arrow\n",
        "WEBVTT\n\nfoo -->\nHi\n",
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c></c>\n",
    ],
)
def test_parse_vtt_timed_handles_malformed_input(text: str) -> None:
    """Empty, header-only, NOTE-only, and malformed cues yield no cues."""
    # Arrange
    # Each parametrized string is one shape of broken or empty input.

    # Act
    cues = parse_vtt_timed(text)

    # Assert
    assert cues == []


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("00:00:00.000", 0.0),
        ("00:00:01.500", 1.5),
        ("00:01:00.000", 60.0),
        ("01:00:00.000", 3600.0),
    ],
)
def test_parse_vtt_timed_converts_timestamps(timestamp: str, expected: float) -> None:
    """A VTT timestamp becomes total seconds on the cue's start field."""
    # Arrange
    vtt = f"WEBVTT\n\n{timestamp} --> {timestamp}\nHello\n"

    # Act
    cues = parse_vtt_timed(vtt)

    # Assert
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(expected)


# ---------------------------------------------------------------------------
# parse_srt_timed
# ---------------------------------------------------------------------------


def test_parse_srt_timed_fixture_structure() -> None:
    """The SRT fixture yields the same ten cues with the expected span."""
    # Arrange
    srt = _load("sample_transcript.srt")

    # Act
    cues = parse_srt_timed(srt)

    # Assert
    assert len(cues) == 10
    assert cues[0].start == pytest.approx(1.0)
    assert cues[-1].start == pytest.approx(56.0)
    assert cues[-1].start >= 55.0


def test_parse_srt_timed_handles_comma_decimals() -> None:
    """SRT's comma decimal separator is parsed like VTT's period."""
    # Arrange
    srt = "1\n00:00:01,500 --> 00:00:02,000\nHello\n"

    # Act
    cues = parse_srt_timed(srt)

    # Assert
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(1.5)


def test_parse_srt_timed_matches_vtt_fixture() -> None:
    """Both formats of the same content produce equivalent cues."""
    # Arrange
    vtt_cues = parse_vtt_timed(_load("sample_transcript.vtt"))
    srt_cues = parse_srt_timed(_load("sample_transcript.srt"))

    # Act
    srt_texts = [cue.text for cue in srt_cues]
    srt_starts = [cue.start for cue in srt_cues]

    # Assert
    assert srt_texts == [cue.text for cue in vtt_cues]
    assert srt_starts == pytest.approx([cue.start for cue in vtt_cues])


# ---------------------------------------------------------------------------
# Cue
# ---------------------------------------------------------------------------


def test_cue_is_frozen() -> None:
    """A Cue cannot be mutated after construction."""
    # Arrange
    cue = Cue(start=0.0, text="hi")

    # Act / Assert
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cue, "start", 5.0)


# ---------------------------------------------------------------------------
# render_transcript_for_llm
# ---------------------------------------------------------------------------


def test_render_transcript_for_llm_empty() -> None:
    """No cues renders as an empty string."""
    # Arrange / Act / Assert
    assert render_transcript_for_llm([]) == ""


def test_render_transcript_for_llm_fixture_lines() -> None:
    """A ~60 second fixture collapses into two or three anchored lines."""
    # Arrange
    cues = parse_vtt_timed(_load("sample_transcript.vtt"))

    # Act
    rendered = render_transcript_for_llm(cues, block_seconds=25)

    # Assert
    lines = rendered.split("\n")
    assert 2 <= len(lines) <= 3
    assert all(_ANCHOR_PATTERN.match(line) for line in lines)
    assert lines[0].startswith("[0:01] ")


def test_render_transcript_for_llm_hours_boundary() -> None:
    """A cue past one hour renders with an hours field."""
    # Arrange
    cues = [Cue(start=3661.0, text="This is the key insight about indexing.")]

    # Act
    rendered = render_transcript_for_llm(cues)

    # Assert
    assert rendered == "[1:01:01] This is the key insight about indexing."


def test_render_transcript_for_llm_groups_by_span() -> None:
    """Each block spans at most block_seconds before a new anchor begins."""
    # Arrange
    cues = [
        Cue(start=0.0, text="one"),
        Cue(start=10.0, text="two"),
        Cue(start=30.0, text="three"),
    ]

    # Act
    rendered = render_transcript_for_llm(cues, block_seconds=25)

    # Assert
    assert rendered == "[0:00] one two\n[0:30] three"
