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
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

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

# The transcript may arrive with [m:ss] / [h:mm:ss] anchors, or without (the
# model does the segmenting itself). The prompt asks for timestamps when anchors
# are present and lets the model omit the field when they are not.
SYSTEM_PROMPT = (
    "You turn a YouTube transcript into structured takeaways for a busy reader.\n"
    "Return exactly 3 takeaways.\n"
    "Each takeaway has a short title (a few words), a one or two sentence "
    "summary, and one to five bullets.\n"
    "When the transcript marks moments as [m:ss] or [h:mm:ss] anchors at the "
    "start of each block, set timestamp_seconds to the moment the bullet "
    "refers to. Compute the seconds from those anchors. When the transcript "
    "has no anchors, omit timestamp_seconds entirely; the worker renders the "
    "bullet as a plain line in that case.\n"
    "Only include a fact when the transcript gives a clear moment for it. "
    "Omit anything you cannot ground in a specific moment.\n"
    "Prefer concrete claims, numbers, and named topics over vague filler.\n"
    "Never invent details that are not in the transcript."
)


class TakeawayBullet(BaseModel):
    """One bullet in a takeaway, optionally tied to a moment in the video.

    ``timestamp_seconds`` is optional because the audio backend delivers
    plain text with no cue timings. The Discord renderer formats the bullet
    with a YouTube deep link when the timestamp is set, and as a plain line
    otherwise.
    """

    text: str = Field(min_length=1)
    timestamp_seconds: int | None = Field(default=None, ge=0)


class Takeaway(BaseModel):
    """One takeaway: a short title, a summary, and supporting bullets."""

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(max_length=1000)
    # The floor is one bullet. The per-config cap is applied by the analyzer,
    # since a Pydantic Field cannot cleanly reference a runtime setting.
    bullets: list[TakeawayBullet] = Field(min_length=1)


class Takeaways(BaseModel):
    """The complete model output: exactly three takeaways."""

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

        Oversized input is truncated with a warning rather than rejected. A
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

        async with self._build_client() as client:
            model = self._model_override or OpenAIChatModel(
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
                model_settings=OpenAIChatModelSettings(
                    max_tokens=self._settings.llm_max_output_tokens
                ),
            )
            result = await asyncio.wait_for(
                agent.run(transcript, message_history=[]),
                timeout=self._settings.llm_timeout_seconds,
            )
            takeaways = result.output

        return self._cap_bullets(takeaways)

    def _cap_bullets(self, takeaways: Takeaways) -> Takeaways:
        """Return takeaways with each item's bullets capped at the config limit."""
        cap = self._settings.takeaway_max_bullets
        if cap <= 0:
            return takeaways
        capped_items = [
            item.model_copy(update={"bullets": item.bullets[:cap]})
            for item in takeaways.items
        ]
        return Takeaways(items=capped_items)


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
