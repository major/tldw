"""Neutral briefing contract shared by analysis and rendering."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

__all__ = [
    "TakeawayBullet",
    "Takeaway",
    "Takeaways",
    "TITLE_MAX_CHARS",
    "SUMMARY_MAX_CHARS",
    "BULLET_MAX_CHARS",
    "MAX_BULLETS",
    "BOTTOM_LINE_MAX_CHARS",
    "WATCH_ITEM_MAX_CHARS",
    "MAX_WATCH_ITEMS",
]

# Length caps that keep the whole briefing inside ONE Discord message, which
# allows at most 6000 characters across all embeds. Worst case:
#   3 x (120 + 400 + 5 x 200) = 4560 for the takeaways
#   300 + 4 x 150             =  900 for the header
# That leaves about 500 characters for embed titles and the footer.
TITLE_MAX_CHARS: int = 120
SUMMARY_MAX_CHARS: int = 400
BULLET_MAX_CHARS: int = 200
MAX_BULLETS: int = 5
BOTTOM_LINE_MAX_CHARS: int = 300
WATCH_ITEM_MAX_CHARS: int = 150
MAX_WATCH_ITEMS: int = 4


class TakeawayBullet(BaseModel):
    """One bullet in a takeaway, optionally tied to a moment in the video.

    ``timestamp_seconds`` is optional because the audio backend delivers
    plain text with no cue timings. The Discord renderer formats the bullet
    with a YouTube deep link when the timestamp is set, and as a plain line
    otherwise.
    """

    text: str = Field(min_length=1, max_length=BULLET_MAX_CHARS)
    timestamp_seconds: int | None = Field(default=None, ge=0)


class Takeaway(BaseModel):
    """One takeaway: a short title, a summary, and supporting bullets."""

    title: str = Field(min_length=1, max_length=TITLE_MAX_CHARS)
    summary: str = Field(max_length=SUMMARY_MAX_CHARS)
    # The floor is one bullet. The per-config cap is applied by the analyzer,
    # since a Pydantic Field cannot cleanly reference a runtime setting.
    bullets: list[TakeawayBullet] = Field(min_length=1, max_length=MAX_BULLETS)


class Takeaways(BaseModel):
    """The complete model output: a briefing header plus three takeaways.

    ``bottom_line`` is the one-sentence verdict and ``watch`` holds the
    concrete levels, catalysts, and conditions to track. Both render in the
    header embed of the single Discord message.
    """

    bottom_line: str = Field(min_length=1, max_length=BOTTOM_LINE_MAX_CHARS)
    watch: list[Annotated[str, Field(min_length=1, max_length=WATCH_ITEM_MAX_CHARS)]] = (
        Field(default_factory=list, max_length=MAX_WATCH_ITEMS)
    )
    items: list[Takeaway] = Field(min_length=3, max_length=3)
