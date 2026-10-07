"""LLM-backed video takeaways for tldw.

This module owns the LLM seam. ``TakeawayAnalyzer`` is the Protocol the worker
depends on, and ``OpencodeGoAnalyzer`` is the production implementation that
talks to the OpenCode Go gateway through the Anthropic-compatible SDK that
pydantic-ai wraps. Keeping the seam behind a Protocol lets the worker inject a
fake in tests, and lets pydantic-ai's ``FunctionModel`` exercise the real Agent,
schema validation, and retry loop fully offline.

The worker renders a transcript (see ``render_transcript_for_llm``) and passes
the resulting string here. This module never reads subtitles and never touches
Discord: it turns text into validated :class:`Takeaways`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from tldw.config import Settings

__all__ = [
    "TakeawayBullet",
    "Takeaway",
    "Takeaways",
    "TakeawayAnalyzer",
    "OpencodeGoAnalyzer",
    "session_id_for",
    "SYSTEM_PROMPT",
]

logger = logging.getLogger(__name__)

# Retries when the model returns output that fails schema validation. Two gives
# the model one chance to correct a bad shape and one to recover from a flake.
DEFAULT_OUTPUT_RETRIES: int = 2

# The transcript arrives pre-rendered with [m:ss] / [h:mm:ss] anchors at the
# start of each block. The prompt leans on those anchors so the model can pick
# accurate timestamp_seconds values instead of guessing.
SYSTEM_PROMPT = (
    "You turn a YouTube transcript into structured takeaways for a busy reader.\n"
    "Return exactly 3 takeaways.\n"
    "Each takeaway has a short title (a few words), a one or two sentence "
    "summary, and one to five bullets.\n"
    "Every bullet must carry an accurate timestamp_seconds for the moment it "
    "refers to. The transcript marks moments as [m:ss] or [h:mm:ss] anchors at "
    "the start of each block; use those anchors to compute the seconds.\n"
    "Only include a fact when the transcript gives a clear timestamp for it. "
    "Omit anything you cannot ground in a specific moment.\n"
    "Prefer concrete claims, numbers, and named topics over vague filler.\n"
    "Never invent details that are not in the transcript."
)


class TakeawayBullet(BaseModel):
    """One bullet in a takeaway, tied to a moment in the video."""

    text: str = Field(min_length=1)
    timestamp_seconds: int = Field(ge=0)


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


def session_id_for(video_id: str) -> str:
    """Return a stable per-video session id for the x-opencode-session header.

    The format is ``ses_`` plus 32 hex characters, derived deterministically
    from ``video_id`` so a retry of the same video reuses the same id.
    """
    return "ses_" + hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:32]


class TakeawayAnalyzer(Protocol):
    """Seam for turning a rendered transcript into structured takeaways."""

    async def analyze(
        self, transcript: str, *, video_id: str, title: str
    ) -> Takeaways: ...


class OpencodeGoAnalyzer:
    """Production analyzer backed by the OpenCode Go gateway via pydantic-ai."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: Model[Any] | None = None,
    ) -> None:
        """Store settings and an optional model override used by tests."""
        self._settings = settings
        self._model_override = model

    def _build_client(self, session_id: str | None = None) -> AsyncAnthropic:
        """Build an AsyncAnthropic client with the headers the gateway requires.

        Three headers matter: ``x-api-key`` (the Anthropic SDK's own convention),
        ``Authorization: Bearer`` (some gateways check this instead), and
        ``x-opencode-session`` (required by OpenCode Go). The session header is
        per-video, so it is passed in at construction: ``default_headers``
        returns a fresh merged dict on each access, which means mutating it after
        the client exists does not stick. When ``session_id`` is omitted a
        deterministic placeholder is used, which is enough for structural checks.
        """
        if not self._settings.opencode_api_key:
            raise ValueError("opencode_api_key is required")
        key = self._settings.opencode_api_key
        return AsyncAnthropic(
            api_key=key,
            base_url=self._settings.opencode_base_url,
            default_headers={
                "Authorization": f"Bearer {key}",
                "x-opencode-session": session_id or session_id_for(""),
            },
        )

    async def analyze(
        self, transcript: str, *, video_id: str, title: str
    ) -> Takeaways:
        """Run the LLM and return validated takeaways.

        Oversized input is truncated with a warning rather than rejected. A
        fresh client is built per call: the worker is serial, so connection
        reuse is not worth leaking one video's session header into another. The
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

        async with self._build_client(session_id_for(video_id)) as client:
            model = self._model_override or AnthropicModel(
                self._settings.opencode_model,
                provider=AnthropicProvider(anthropic_client=client),
            )
            agent = Agent(
                model,
                output_type=Takeaways,
                system_prompt=SYSTEM_PROMPT,
                # 2.54 spells this ``retries``; it controls output validation
                # retries (there is no ``output_retries`` kwarg in this version).
                retries=DEFAULT_OUTPUT_RETRIES,
                model_settings=AnthropicModelSettings(
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
