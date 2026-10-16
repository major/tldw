"""Tests for the Discord briefing embed builders.

The builders are pure: they turn a ``Takeaways`` model into embed dicts. Tests
build the model with ``make_takeaways`` or ``Takeaways.model_validate`` so no
fixture files are needed. No test touches the network.
"""

from __future__ import annotations

import logging
import re

import pytest
from helpers import make_takeaways

from tldw import discord as discord_module
from tldw.discord import (
    _format_bullet,
    build_header_embed,
    build_takeaway_embed,
    build_takeaway_embeds,
)
from tldw.llm import Takeaway, TakeawayBullet, Takeaways

_VIDEO_ID = "dQw4w9WgXcQ"
_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_VIDEO_TITLE = "Video title"
_CHANNEL = "Test Channel"
_DEFAULT_COLOR = 0x5865F2
_DISCORD_LIMIT = 6000

# The embed tests build takeaways with "Title {i}" titles, "Short summary"
# summaries, and fixed timestamps. These constants keep the callsites readable.
_EMBED_TITLE = "Title {i}"
_EMBED_SUMMARY = "Short summary"
_TIMES_2: list[list[int | None]] = [[0, 15], [60, 75], [120, 135]]
_TIMES_3: list[list[int | None]] = [[0, 15, 30], [60, 75, 90], [120, 135, 150]]


def _briefing(
    *,
    bullet_count: int = 2,
    summary: str = _EMBED_SUMMARY,
    bullet_text: str | None = None,
    bottom_line: str = "Bottom line",
    watch: tuple[str, ...] = (),
) -> Takeaways:
    """Build a three-takeaway briefing with the shared test templates."""
    return make_takeaways(
        bullet_count=bullet_count,
        title_template=_EMBED_TITLE,
        summary_template=summary,
        bullet_text=bullet_text,
        bullet_times=_TIMES_3 if bullet_count == 3 else _TIMES_2,
        bottom_line=bottom_line,
        watch=watch,
    )


def _build_all(takeaways: Takeaways, **overrides: str) -> list[dict]:
    """Build the full four-embed briefing with the shared test video and channel."""
    kwargs = {
        "video_id": _VIDEO_ID,
        "video_url": _VIDEO_URL,
        "video_title": _VIDEO_TITLE,
        "channel_name": _CHANNEL,
    }
    kwargs.update(overrides)
    return build_takeaway_embeds(takeaways, **kwargs)


def _build_header(
    takeaways: Takeaways,
    *,
    video_title: str = _VIDEO_TITLE,
    channel_name: str = _CHANNEL,
) -> dict:
    """Build the header embed with the shared test video URL."""
    return build_header_embed(
        takeaways,
        video_url=_VIDEO_URL,
        video_title=video_title,
        channel_name=channel_name,
    )


def _embed(takeaways: Takeaways | None = None, *, index: int = 1) -> dict:
    """Build one takeaway embed from the first takeaway of the given model."""
    model = takeaways if takeaways is not None else _briefing()
    return build_takeaway_embed(
        model.items[0],
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        index=index,
    )


def _bullet_lines(embed: dict) -> list[str]:
    """Return the bullet lines of an embed description."""
    return [line for line in embed["description"].split("\n") if line.startswith("- ")]


def _bullet_counts(embeds: list[dict]) -> list[int]:
    """Return the bullet count of each takeaway embed, skipping the header."""
    return [len(_bullet_lines(embed)) for embed in embeds[1:]]


def _total_chars(embeds: list[dict]) -> int:
    """Sum title, description, and footer text over all embeds, as Discord does."""
    return sum(
        len(embed.get("title", ""))
        + len(embed["description"])
        + len(embed.get("footer", {}).get("text", ""))
        for embed in embeds
    )


def _worst_case_takeaways() -> Takeaways:
    """Build a briefing at every schema cap, the longest text the model may return."""
    bullet = {"text": "b" * 200, "timestamp_seconds": 36000}
    item = {"title": "T" * 120, "summary": "s" * 400, "bullets": [bullet] * 5}
    return Takeaways.model_validate(
        {
            "bottom_line": "x" * 300,
            "watch": ["w" * 150] * 4,
            "items": [item, item, item],
        }
    )


# ---------------------------------------------------------------------------
# build_header_embed
# ---------------------------------------------------------------------------


def test_build_header_embed_has_expected_keys() -> None:
    """The header carries title, url, description, and color, and no footer."""
    embed = _build_header(_briefing())

    assert set(embed) == {"title", "url", "description", "color"}


def test_build_header_embed_title_names_channel_and_video() -> None:
    """The header title is ``{channel}: {video title}``."""
    embed = _build_header(_briefing())

    assert embed["title"] == f"{_CHANNEL}: {_VIDEO_TITLE}"


def test_build_header_embed_url_is_video_url() -> None:
    """The header url is the video url, making the title clickable."""
    embed = _build_header(_briefing())

    assert embed["url"] == _VIDEO_URL


def test_build_header_embed_default_color() -> None:
    """The default header color is Discord blurple."""
    embed = _build_header(_briefing())

    assert embed["color"] == _DEFAULT_COLOR


def test_build_header_embed_without_watch_shows_only_bottom_line() -> None:
    """With no watch items the description is just the bottom line."""
    takeaways = _briefing(bottom_line="Stocks rally", watch=())

    embed = _build_header(takeaways)

    assert embed["description"] == "**Bottom line:** Stocks rally"


def test_build_header_embed_lists_watch_items() -> None:
    """Watch items follow the bottom line as a bulleted list under a heading."""
    takeaways = _briefing(
        bottom_line="Stocks rally",
        watch=("SPY holds 500", "CPI prints Tuesday"),
    )

    embed = _build_header(takeaways)

    assert embed["description"] == (
        "**Bottom line:** Stocks rally\n\n**Watch**\n- SPY holds 500\n- CPI prints Tuesday"
    )


def test_build_header_embed_truncates_oversized_title() -> None:
    """A header title over 256 chars is cut to 256 with an ellipsis."""
    embed = _build_header(
        _briefing(), channel_name="A" * 200, video_title="T" * 100
    )

    assert len(embed["title"]) == 256
    assert embed["title"].endswith("...")


# ---------------------------------------------------------------------------
# build_takeaway_embed
# ---------------------------------------------------------------------------


def test_build_takeaway_embed_has_expected_keys() -> None:
    """A takeaway embed carries title, url, description, and color, and no footer."""
    embed = _embed()

    assert set(embed) == {"title", "url", "description", "color"}


def test_build_takeaway_embed_url_is_video_url() -> None:
    """The embed url is the passed video url, making the title clickable."""
    embed = _embed()

    assert embed["url"] == _VIDEO_URL


def test_build_takeaway_embed_default_color() -> None:
    """The default embed color is Discord blurple."""
    embed = _embed()

    assert embed["color"] == _DEFAULT_COLOR


@pytest.mark.parametrize("index", [1, 2, 3])
def test_build_takeaway_embed_title_is_numbered(index: int) -> None:
    """The title carries the 1-based position and no channel prefix."""
    embed = _embed(index=index)

    assert embed["title"] == f"{index}. Title 0"


def test_build_takeaway_embed_description_starts_with_summary() -> None:
    """The description opens with the takeaway summary."""
    embed = _embed(_briefing(summary="A crisp summary"))

    assert embed["description"].startswith("A crisp summary")


def test_build_takeaway_embed_has_one_bullet_line_per_bullet() -> None:
    """Every bullet becomes one markdown bullet line."""
    embed = _embed(_briefing(bullet_count=2))

    assert len(_bullet_lines(embed)) == 2


def test_build_takeaway_embed_bullet_links_match_deep_link_shape() -> None:
    """Each bullet links to the video at the bullet's timestamp."""
    embed = _embed(_briefing(bullet_count=3))

    links = re.findall(r"\((https://youtu\.be/[^)]+)\)", embed["description"])
    assert len(links) == 3
    pattern = re.compile(rf"^https://youtu\.be/{_VIDEO_ID}\?t=\d+$")
    assert all(pattern.match(link) for link in links)


def test_build_takeaway_embed_truncates_oversized_description() -> None:
    """A description over 4096 chars is truncated to 4096 with an ellipsis.

    The schema caps summaries at 400 chars, so ``model_construct`` skips
    validation to build the over-cap input the builder must still handle.
    """
    takeaway = Takeaway.model_construct(
        title="Long",
        summary="x" * 5000,
        bullets=[TakeawayBullet(text="One")],
    )

    embed = build_takeaway_embed(
        takeaway, video_id=_VIDEO_ID, video_url=_VIDEO_URL, index=1
    )

    description = embed["description"]
    assert len(description) == 4096
    assert description.endswith("...")


def test_build_takeaway_embed_truncates_oversized_title() -> None:
    """A title over 256 chars is cut to 256 with an ellipsis.

    The schema caps titles at 120 chars, so ``model_construct`` skips
    validation to build the over-cap input the builder must still handle.
    """
    takeaway = Takeaway.model_construct(
        title="T" * 300,
        summary="Short",
        bullets=[TakeawayBullet(text="One")],
    )

    embed = build_takeaway_embed(
        takeaway, video_id=_VIDEO_ID, video_url=_VIDEO_URL, index=1
    )

    assert len(embed["title"]) == 256
    assert embed["title"].endswith("...")


# ---------------------------------------------------------------------------
# build_takeaway_embeds: shape
# ---------------------------------------------------------------------------


def test_build_takeaway_embeds_returns_header_then_three_takeaways() -> None:
    """The briefing is four embeds: the header, then the three takeaways."""
    embeds = _build_all(_briefing())

    assert len(embeds) == 4
    assert embeds[0]["title"] == f"{_CHANNEL}: {_VIDEO_TITLE}"
    assert [embed["title"] for embed in embeds[1:]] == [
        "1. Title 0",
        "2. Title 1",
        "3. Title 2",
    ]


def test_build_takeaway_embeds_puts_footer_only_on_last_embed() -> None:
    """The channel name is the footer of the last embed and nowhere else."""
    embeds = _build_all(_briefing())

    footers = [embed.get("footer") for embed in embeds]
    assert footers == [None, None, None, {"text": _CHANNEL}]


# ---------------------------------------------------------------------------
# build_takeaway_embeds: budget
# ---------------------------------------------------------------------------


def test_build_takeaway_embeds_worst_case_fits_message_limit() -> None:
    """A briefing at every schema cap still fits Discord's 6000 char total.

    The raw input is about 6500 chars, so this also covers trimming. Every
    takeaway keeps at least one bullet.
    """
    embeds = _build_all(
        _worst_case_takeaways(),
        video_title="V" * 200,
        channel_name="C" * 50,
    )

    assert _total_chars(embeds) <= _DISCORD_LIMIT
    assert all(count >= 1 for count in _bullet_counts(embeds))


def test_build_takeaway_embeds_does_not_trim_or_warn_when_under_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A briefing that already fits keeps every bullet and logs nothing."""
    caplog.set_level(logging.WARNING, logger="tldw.discord")

    embeds = _build_all(_briefing(bullet_count=3))

    assert _bullet_counts(embeds) == [3, 3, 3]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_build_takeaway_embeds_drops_bullets_before_truncating(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Over budget by one char, exactly one bullet is dropped and no text is cut.

    The first takeaway ties for the most bullets, so it loses the bullet.
    """
    # Arrange
    caplog.set_level(logging.WARNING, logger="tldw.discord")
    takeaways = _briefing(bullet_count=3, bullet_text="Point {i}-{n}")
    full = _total_chars(_build_all(takeaways))
    monkeypatch.setattr(discord_module, "DISCORD_MESSAGE_MAX_CHARS", full - 1)

    # Act
    embeds = _build_all(takeaways)

    # Assert
    assert _total_chars(embeds) <= full - 1
    assert _bullet_counts(embeds) == [2, 3, 3]
    assert not any(embed["description"].endswith("...") for embed in embeds)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_build_takeaway_embeds_truncates_only_after_bullets_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one bullet each, the longest description is cut to fit the limit.

    Only one description is cut, and the titles are untouched. The cut runs from
    the end of the text, so it can reach the bullet line of the cut embed.
    """
    # Arrange
    takeaways = _briefing(bullet_count=1, summary="S" * 400)
    full = _total_chars(_build_all(takeaways))
    limit = full - 100
    monkeypatch.setattr(discord_module, "DISCORD_MESSAGE_MAX_CHARS", limit)

    # Act
    embeds = _build_all(takeaways)

    # Assert
    assert _total_chars(embeds) == limit
    truncated = [embed for embed in embeds if embed["description"].endswith("...")]
    assert len(truncated) == 1
    assert [embed["title"] for embed in embeds[1:]] == [
        "1. Title 0",
        "2. Title 1",
        "3. Title 2",
    ]


# ---------------------------------------------------------------------------
# Optional timestamps (audio backend)
# ---------------------------------------------------------------------------


def test_format_bullet_renders_plain_text_when_timestamp_is_none() -> None:
    """A bullet without a timestamp renders as a plain markdown line."""
    bullet = TakeawayBullet(text="Hello world", timestamp_seconds=None)

    assert _format_bullet(bullet, video_id="abc") == "- Hello world"


def test_format_bullet_renders_deep_link_when_timestamp_is_set() -> None:
    """A bullet with a timestamp renders as a deep-linked markdown line."""
    bullet = TakeawayBullet(text="Hi", timestamp_seconds=42)

    assert (
        _format_bullet(bullet, video_id="abc")
        == "- [0:42](https://youtu.be/abc?t=42) Hi"
    )


def test_build_takeaway_embed_works_with_none_timestamps() -> None:
    """An audio-backend takeaway renders plain bullets with no deep links."""
    takeaway = Takeaway(
        title="Audio takeaway",
        summary="Short summary",
        bullets=[TakeawayBullet(text=f"Point {n}") for n in range(3)],
    )

    embed = build_takeaway_embed(
        takeaway, video_id=_VIDEO_ID, video_url=_VIDEO_URL, index=1
    )

    assert _bullet_lines(embed) == ["- Point 0", "- Point 1", "- Point 2"]
    assert "youtu.be" not in embed["description"]
