"""Tests for the Discord takeaway embed builders.

The builders are pure: they turn a ``Takeaways`` model into embed dicts. Tests
build the model inline with ``model_validate`` so no fixture files are needed.
No test touches the network.
"""

from __future__ import annotations

import re

import pytest

from tldw.discord import _format_bullet, build_takeaway_embed, build_takeaway_embeds
from tldw.llm import Takeaway, TakeawayBullet, Takeaways

_VIDEO_ID = "dQw4w9WgXcQ"
_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_CHANNEL = "Test Channel"
_DEFAULT_COLOR = 0x5865F2


def _make_takeaways(
    *,
    bullet_count: int = 2,
    bullet_text: str | None = None,
    summary: str = "Short summary",
) -> Takeaways:
    """Build a valid Takeaways with three items and configurable bullets."""
    items = []
    for i in range(3):
        bullets = [
            {
                "text": bullet_text if bullet_text is not None else f"Bullet {i}-{n}",
                "timestamp_seconds": i * 60 + n * 15,
            }
            for n in range(bullet_count)
        ]
        items.append({"title": f"Title {i}", "summary": summary, "bullets": bullets})
    return Takeaways.model_validate({"items": items})


def _embed(takeaways: Takeaways | None = None, *, index: int = 1) -> dict:
    """Build one embed from the first takeaway of the given model."""
    model = takeaways if takeaways is not None else _make_takeaways()
    return build_takeaway_embed(
        model.items[0],
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=_CHANNEL,
        index=index,
    )


# ---------------------------------------------------------------------------
# build_takeaway_embed
# ---------------------------------------------------------------------------


def test_build_takeaway_embed_has_expected_keys() -> None:
    """The embed carries title, url, description, color, and footer."""
    embed = _embed()

    assert set(embed) == {"title", "url", "description", "color", "footer"}


def test_build_takeaway_embed_url_is_video_url() -> None:
    """The embed url is the passed video url, making the title clickable."""
    embed = _embed()

    assert embed["url"] == _VIDEO_URL


def test_build_takeaway_embed_default_color() -> None:
    """The default embed color is Discord blurple."""
    embed = _embed()

    assert embed["color"] == _DEFAULT_COLOR


def test_build_takeaway_embed_footer_text() -> None:
    """The footer names the takeaway position, total, and channel."""
    embed = _embed(index=1)

    assert embed["footer"]["text"] == f"Takeaway 1 of 3 · {_CHANNEL}"


def test_build_takeaway_embed_description_starts_with_summary() -> None:
    """The description opens with the takeaway summary."""
    takeaways = _make_takeaways(summary="A crisp summary")
    embed = _embed(takeaways)

    assert embed["description"].startswith("A crisp summary")


def test_build_takeaway_embed_has_one_bullet_line_per_bullet() -> None:
    """Every bullet becomes one markdown bullet line."""
    takeaways = _make_takeaways(bullet_count=2)
    embed = _embed(takeaways)

    bullet_lines = [
        line for line in embed["description"].split("\n") if line.startswith("- [")
    ]
    assert len(bullet_lines) == 2


def test_build_takeaway_embed_bullet_links_match_deep_link_shape() -> None:
    """Each bullet links to the video at the bullet's timestamp."""
    takeaways = _make_takeaways(bullet_count=3)
    embed = _embed(takeaways)

    links = re.findall(r"\((https://youtu\.be/[^)]+)\)", embed["description"])
    assert len(links) == 3
    pattern = re.compile(rf"^https://youtu\.be/{_VIDEO_ID}\?t=\d+$")
    assert all(pattern.match(link) for link in links)


def test_build_takeaway_embed_truncates_oversized_description() -> None:
    """A description over 4096 chars is truncated to 4096 with an ellipsis."""
    takeaways = _make_takeaways(bullet_text="x" * 5000)
    embed = _embed(takeaways)

    description = embed["description"]
    assert len(description) == 4096
    assert description.endswith("...")


def test_build_takeaway_embed_title_is_prefixed_with_channel_name() -> None:
    """The embed title is ``{channel_name}: {takeaway.title}`` so the
    creator is visible in the channel timeline alongside the headline."""
    embed = _embed()

    assert embed["title"] == f"{_CHANNEL}: Title 0"


def test_build_takeaway_embed_truncates_oversized_title() -> None:
    """A title over 256 chars is truncated to 256 with an ellipsis."""
    # Takeaway.title is capped at 200 chars by the pydantic model, so the
    # longest input we can produce is a 200-char title with a channel name
    # long enough to push the formatted "channel: title" past 256 chars.
    long_channel = "A" * 100
    long_title = "T" * 200
    takeaway = Takeaway(
        title=long_title,
        summary="Short",
        bullets=[TakeawayBullet(text="One")],
    )
    embed = build_takeaway_embed(
        takeaway,
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=long_channel,
        index=1,
    )

    title = embed["title"]
    assert len(title) == 256
    assert title.endswith("...")


# ---------------------------------------------------------------------------
# build_takeaway_embeds
# ---------------------------------------------------------------------------


def test_build_takeaway_embeds_returns_one_per_item() -> None:
    """Three takeaways produce exactly three embeds."""
    embeds = build_takeaway_embeds(
        _make_takeaways(),
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=_CHANNEL,
    )

    assert len(embeds) == 3


@pytest.mark.parametrize("index", [1, 2, 3])
def test_build_takeaway_embeds_footer_uses_position(index: int) -> None:
    """Each embed's footer reports its 1-based position out of the total."""
    embeds = build_takeaway_embeds(
        _make_takeaways(),
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=_CHANNEL,
    )

    assert embeds[index - 1]["footer"]["text"] == f"Takeaway {index} of 3 · {_CHANNEL}"


def test_build_takeaway_embeds_total_chars_under_limit() -> None:
    """A typical three-takeaway message stays under Discord's 6000 char total."""
    embeds = build_takeaway_embeds(
        _make_takeaways(bullet_count=3, bullet_text="y" * 80),
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=_CHANNEL,
    )

    total = sum(
        len(embed["title"])
        + len(embed["url"])
        + len(embed["description"])
        + len(embed["footer"]["text"])
        for embed in embeds
    )
    assert total < 6000


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
        takeaway,
        video_id=_VIDEO_ID,
        video_url=_VIDEO_URL,
        channel_name=_CHANNEL,
        index=1,
    )

    bullet_lines = [
        line for line in embed["description"].split("\n") if line.startswith("- ")
    ]
    assert bullet_lines == ["- Point 0", "- Point 1", "- Point 2"]
    assert "youtu.be" not in embed["description"]
