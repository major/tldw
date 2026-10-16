"""LLM-backed video takeaways for tldw.

This module owns the LLM seam. ``TakeawayAnalyzer`` is the Protocol the worker
depends on, and ``OpenAIAnalyzer`` is the production implementation that
talks to the OpenAI chat completions API through pydantic-ai's
``OpenAIModel``. Keeping the seam behind a Protocol lets the worker inject a
fake in tests, and lets pydantic-ai's ``FunctionModel`` exercise the real
Agent, schema validation, and retry loop fully offline.

The worker renders a transcript and passes the resulting string here. This
module never reads subtitles and never touches Discord: it turns text into
validated :class:`Takeaways`.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

# Lazy imports (PEP 810): ``pydantic_ai.models.openai`` and
# ``pydantic_ai.providers.openai`` are only used when the production analyzer
# actually runs. Tests that inject ``model=...`` never reach the code path,
# so deferring the import keeps the cold path free of the pydantic-ai model
# graph and its transitive OpenAI plumbing.
lazy from pydantic_ai.models.openai import OpenAIResponsesModel
lazy from pydantic_ai.models.openai import OpenAIResponsesModelSettings
lazy from pydantic_ai.providers.openai import OpenAIProvider

from tldw.config import Settings

__all__ = [
    "Cue",
    "TakeawayBullet",
    "Takeaway",
    "Takeaways",
    "TakeawayAnalyzer",
    "OpenAIAnalyzer",
    "snap_timestamps",
    "SYSTEM_PROMPT",
]

logger = logging.getLogger(__name__)

# Retries when the model returns output that fails schema validation. Two gives
# the model one chance to correct a bad shape and one to recover from a flake.
DEFAULT_OUTPUT_RETRIES: int = 2

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

# The transcript may arrive with [m:ss] / [h:mm:ss] anchors, or without (the
# model does the segmenting itself). The prompt asks for timestamps when anchors
# are present and lets the model omit the field when they are not. Audio-only
# runs have no anchors, so the prompt must not demand a clear moment for every
# fact; that rule made the output vague.
SYSTEM_PROMPT = (
    "You write a trading briefing from a YouTube market video for an active "
    "trader who will not watch it. The trader wants to know what the speaker "
    "concludes, which levels and catalysts matter, and what would change the "
    "view.\n"
    "Return a bottom_line, a watch list, and exactly 3 takeaways, most important "
    "first.\n"
    "bottom_line: one sentence with the speaker's overall stance (bullish, "
    "bearish, neutral, or mixed), the main reason, and the time frame if one is "
    "given.\n"
    "watch: up to 4 short items the trader should track, each with a ticker or "
    "index plus a price level, date, data release, or event. Write conditions as "
    '"If X, then Y". Use an empty list when the video gives nothing concrete.\n'
    "Each takeaway has a title, a summary, and one to five bullets.\n"
    "Title: a claim, not a topic. Good: \"SMH bounce needs a base before any "
    'rally". Bad: "Semiconductor update".\n'
    "Summary: one or two sentences that state the conclusion first.\n"
    "Bullets: each one holds something concrete: a ticker, price level, "
    "percentage, options or gamma level (put wall, call wall), moving average, "
    "date, data release, or trade setup. Bold tickers and key levels with ** in "
    "titles, bullets, and watch items.\n"
    "Keep the speaker's stance and confidence. Mark opinions and predictions as "
    "the speaker's view. Never add trade advice, targets, or levels the speaker "
    "did not give.\n"
    'Write directly. Never write "the speaker says", "the narrator", or "in this '
    'video".\n'
    "Skip filler, sponsor reads, channel plugs, and requests to like or subscribe.\n"
    "When the transcript marks moments as [m:ss] or [h:mm:ss] anchors at the "
    "start of each block, set timestamp_seconds to the moment the bullet refers "
    "to, computed from those anchors. When the transcript has no anchors, omit "
    "timestamp_seconds. Concrete details are still wanted when there are no "
    "anchors.\n"
    "Never invent details that are not in the transcript.\n"
    "Always write every field in US English, even when the transcript is in "
    "another language."
)


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


class TakeawayAnalyzer(Protocol):
    """Seam for turning a rendered transcript into structured takeaways."""

    async def analyze(
        self, transcript: str, *, video_id: str, title: str
    ) -> Takeaways: ...


class OpenAIAnalyzer:
    """Production analyzer backed by OpenAI through pydantic-ai."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: Model[Any] | None = None,
    ) -> None:
        """Store settings and an optional model override used by tests."""
        self._settings = settings
        self._model_override = model

    def _build_client(self) -> AsyncOpenAI:
        """Build an ``AsyncOpenAI`` client from the configured credentials.

        OpenAI's API only needs the bearer token, so there are no extra
        session or version headers to thread through. The base URL is
        configurable so the same analyzer can target OpenAI-compatible
        gateways when needed.
        """
        if not self._settings.openai_api_key:
            raise ValueError("openai_api_key is required")
        return AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )

    async def analyze(
        self, transcript: str, *, video_id: str, title: str
    ) -> Takeaways:
        """Run the LLM and return validated takeaways.

        Oversized transcripts are truncated with a warning rather than
        rejected; the cap applies to the transcript only, not the title header
        sent ahead of it. The user message is ``Video title: <title>`` followed
        by a ``---`` line and the transcript, so the model sees the title. A
        fresh client is built per call: the worker is serial, so connection
        reuse is not worth leaking one video's session into another. The
        per-takeaway bullet cap is applied after validation.
        """
        if len(transcript) > self._settings.llm_max_input_chars:
            logger.warning(
                "transcript for %s is %d chars, truncating to %d",
                video_id,
                len(transcript),
                self._settings.llm_max_input_chars,
            )
            transcript = transcript[: self._settings.llm_max_input_chars]
        user_prompt = f"Video title: {title}\n---\n{transcript}"

        async with self._build_client() as client:
            model = self._model_override or OpenAIResponsesModel(
                self._settings.openai_model,
                provider=OpenAIProvider(openai_client=client),
            )
            agent = Agent(
                model,
                output_type=Takeaways,
                system_prompt=SYSTEM_PROMPT,
                # 2.54 spells this ``retries``; it controls output validation
                # retries (there is no ``output_retries`` kwarg in this version).
                retries=DEFAULT_OUTPUT_RETRIES,
                model_settings=OpenAIResponsesModelSettings(
                    max_tokens=self._settings.llm_max_output_tokens,
                ),
            )
            result = await asyncio.wait_for(
                agent.run(user_prompt, message_history=[]),
                timeout=self._settings.llm_timeout_seconds,
            )
            takeaways = result.output

        return self._cap_bullets(takeaways)

    def _cap_bullets(self, takeaways: Takeaways) -> Takeaways:
        """Return takeaways with each item's bullets capped at the config limit.

        ``model_copy`` keeps the header fields (``bottom_line`` and ``watch``)
        that a fresh ``Takeaways(items=...)`` would drop.
        """
        cap = self._settings.takeaway_max_bullets
        if cap <= 0:
            return takeaways
        capped_items = [
            item.model_copy(update={"bullets": item.bullets[:cap]})
            for item in takeaways.items
        ]
        return takeaways.model_copy(update={"items": capped_items})


@dataclass(frozen=True)
class Cue:
    """One timed subtitle cue: start time in seconds + clean text."""

    start: float
    text: str


def snap_timestamps(
    takeaways: Takeaways,
    cues: list[Cue],
    *,
    max_drift_seconds: int = 30,
) -> Takeaways:
    """Snap each bullet's timestamp_seconds to the nearest cue start.

    The LLM is asked to cite timestamps from the rendered transcript's
    ``[m:ss]`` anchors, but it can still drift. This function guarantees
    every returned timestamp lands on a real subtitle cue by snapping to
    the nearest ``cue.start`` from the parsed transcript.

    Bullets whose nearest cue is more than ``max_drift_seconds`` away are
    kept as-is and logged at WARNING level. A large drift is a hallucination
    signal (the LLM cited a moment the transcript does not cover) and is
    worth seeing in logs even though we keep the bullet (a slightly-off
    link beats a missing takeaway).

    Returns a new ``Takeaways`` with adjusted timestamps; does not mutate
    the input. If ``cues`` is empty, returns ``takeaways`` unchanged (nothing
    to snap to).
    """
    if not cues:
        return takeaways

    # Cue starts are expected in source order, but sorting here makes the
    # bisect search correct even if a parser ever emits them out of order.
    starts = sorted(cue.start for cue in cues)
    new_items: list[Takeaway] = []
    for takeaway in takeaways.items:
        new_bullets: list[TakeawayBullet] = []
        for bullet in takeaway.bullets:
            original = bullet.timestamp_seconds
            # Bullets without a timestamp are the audio-backend shape: the
            # transcript has no cues, so there is nothing to snap against.
            if original is None:
                new_bullets.append(bullet)
                continue
            # bisect_left gives the first cue at or after the timestamp. The
            # nearest cue is either that one or the one just before it.
            idx = bisect.bisect_left(starts, original)
            candidates: list[float] = []
            if idx < len(starts):
                candidates.append(starts[idx])
            if idx > 0:
                candidates.append(starts[idx - 1])
            nearest = min(candidates, key=lambda value: abs(value - original))
            drift = abs(nearest - original)
            if drift <= max_drift_seconds:
                new_bullets.append(
                    bullet.model_copy(update={"timestamp_seconds": int(nearest)})
                )
            else:
                logger.warning(
                    "takeaway bullet timestamp %ds drifts %.1fs from the "
                    "nearest cue, over the %ds limit; keeping it as-is",
                    original,
                    drift,
                    max_drift_seconds,
                )
                new_bullets.append(bullet)
        new_items.append(takeaway.model_copy(update={"bullets": new_bullets}))
    return takeaways.model_copy(update={"items": new_items})
