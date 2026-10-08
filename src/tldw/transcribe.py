"""OpenAI ``gpt-transcribe`` client for tldw.

This module is a thin async wrapper around ``client.audio.transcriptions.create``
from the official ``openai`` SDK. One call per invocation: retries and backoff
are the worker's job, not this module's.

US-English-always contract
--------------------------

The audio is ALWAYS transcribed as US English. No caller input can change the
language - ``languages=["en"]`` is hardcoded in the body sent to OpenAI. Per
OpenAI's docs for gpt-transcribe, the API uses the plural ``languages`` field
for input-language hints; the singular ``language=`` field is for older models
like whisper-1. Sending both to gpt-transcribe is rejected.

Three context knobs (per the OpenAI docs) are supported:

- ``prompt`` - free-form context about the recording (topic, setting).
  Built from the entry's ``title`` and ``channel_name`` when available.
- ``keywords`` - vocab hints that may appear in the audio (product names,
  acronyms, etc.). Empty by default; deployment can wire a domain list via
  ``Settings.transcribe_keywords``.
- ``languages`` - hardcoded to ``["en"]`` (see above).

``prompt`` and ``keywords`` are omitted entirely from the API call when
None/empty - the docs say "use these inputs only for context relevant to the
audio; don't restate the transcription task."

The worker injects this module's :func:`transcribe` function as a seam, so the
production wiring is a single factory that closes over the settings. Keeping
the seam as a plain function lets tests substitute a fake without a client.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

__all__ = ["transcribe", "TranscribeError", "DEFAULT_TIMEOUT_SECONDS"]

logger = logging.getLogger(__name__)

# Transcription can take a while for long audio, so the default is generous.
# The worker passes its configured timeout explicitly; this is the fallback.
DEFAULT_TIMEOUT_SECONDS: float = 600.0

# Hardcoded input languages for gpt-transcribe. The OpenAI docs require the
# plural ``languages`` field (singular ``language=`` is for older models),
# and tldw's contract is: US English always. No caller can override this.
TRANSCRIBE_LANGUAGES: tuple[str, ...] = ("en",)


class TranscribeError(RuntimeError):
    """Raised when transcription fails in a way the worker can classify.

    ``status_code`` carries the HTTP status from an ``APIStatusError`` (for
    example 429 for rate limiting) so the worker can choose between a
    RATE_LIMITED retry and GIVE_UP_AUDIO. It is ``None`` for local failures
    such as a missing API key.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Store the message and optional HTTP status code."""
        super().__init__(message)
        self.status_code = status_code


def build_prompt(
    *,
    title: str | None = None,
    channel_name: str | None = None,
) -> str | None:
    """Build a US-English transcription prompt from the entry metadata.

    Returns ``None`` when no metadata is available, so the API call omits
    ``prompt`` entirely (per OpenAI's "don't restate the task" guidance).
    The prompt anchors the ASR to the video's topic and forces US English
    as the transcript language.
    """
    if not title and not channel_name:
        return None
    parts: list[str] = []
    if title:
        parts.append(f"YouTube video titled {title!r}")
    if channel_name:
        parts.append(f"from channel {channel_name!r}")
    parts.append("US English transcript.")
    return " ".join(parts)


async def transcribe(
    audio_path: Path,
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float,
    prompt: str | None = None,
    keywords: list[str] | None = None,
) -> str:
    """Transcribe an audio file via OpenAI's /v1/audio/transcriptions.

    Returns the plain transcript text. One call; the worker owns retry
    scheduling. The language is hardcoded to ``en`` (see module docstring)
    - no caller input can change it. ``response_format="json"`` is the
    only format gpt-transcribe accepts.

    ``prompt`` and ``keywords`` are forwarded to the API only when set;
    ``None``/empty values are omitted so the model isn't given useless
    context (per OpenAI's docs).

    The file is opened as a binary handle and closed via ``with``; the SDK
    builds the multipart body and infers the MIME type from the filename
    extension (so a ``.webm`` file is sent as audio/webm).
    """
    # Defense in depth: the worker checks this too, but a missing key here
    # should fail fast and clearly rather than reaching the network.
    if not api_key:
        raise TranscribeError("api_key is required")

    call_kwargs: dict[str, Any] = {
        "model": model,
        "languages": list(TRANSCRIBE_LANGUAGES),
        "response_format": "json",
    }
    if prompt:
        call_kwargs["prompt"] = prompt
    if keywords:
        call_kwargs["keywords"] = list(keywords)

    # A fresh client per call matches tldw.llm.OpenAIAnalyzer._build_client:
    # the worker is serial, so connection reuse is not worth leaking one
    # video's session into another.
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        with audio_path.open("rb") as fh:
            try:
                result = await asyncio.wait_for(
                    client.audio.transcriptions.create(
                        file=fh,
                        **call_kwargs,
                    ),
                    timeout=timeout_s,
                )
            except openai.APIStatusError as exc:
                # The worker inspects the status to pick between a
                # RATE_LIMITED retry and GIVE_UP_AUDIO. Other openai errors
                # (connection, timeout, etc.) pass through for the worker.
                raise TranscribeError(
                    f"transcription failed with status {exc.status_code}",
                    status_code=exc.status_code,
                ) from exc

    return result.text
