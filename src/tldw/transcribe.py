"""OpenAI ``gpt-transcribe`` client for tldw.

This module is a thin async wrapper around ``client.audio.transcriptions.create``
from the official ``openai`` SDK. One call per invocation: retries and backoff
are the worker's job, not this module's. It uses ``languages=[]`` (the plural
array) because ``gpt-transcribe`` only accepts the plural form; sending both
``language=`` and ``languages=`` is rejected by the API.

The worker injects this module's :func:`transcribe` function as a seam, so the
production wiring is a single factory that closes over the settings. Keeping the
seam as a plain function lets tests substitute a fake without a client.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import openai
from openai import AsyncOpenAI

__all__ = ["transcribe", "TranscribeError", "DEFAULT_TIMEOUT_SECONDS"]

logger = logging.getLogger(__name__)

# Transcription can take a while for long audio, so the default is generous.
# The worker passes its configured timeout explicitly; this is the fallback.
DEFAULT_TIMEOUT_SECONDS: float = 600.0


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


async def transcribe(
    audio_path: Path,
    *,
    model: str,
    langs: list[str],
    api_key: str,
    base_url: str,
    timeout_s: float,
) -> str:
    """Transcribe an audio file via OpenAI's /v1/audio/transcriptions.

    Returns the plain transcript text. One call; the worker owns retry
    scheduling. ``languages`` is the PLURAL array (``gpt-transcribe`` only
    accepts the plural form; sending both ``language=`` and ``languages=`` is
    rejected by the API). ``response_format="json"`` is the only format
    ``gpt-transcribe`` accepts.

    The file is opened as a binary handle and closed via ``with``; the SDK
    builds the multipart body and infers the MIME type from the filename
    extension (so a ``.webm`` file is sent as audio/webm).
    """
    # Defense in depth: the worker checks this too, but a missing key here
    # should fail fast and clearly rather than reaching the network.
    if not api_key:
        raise TranscribeError("api_key is required")

    # A fresh client per call matches tldw.llm.OpenAIAnalyzer._build_client:
    # the worker is serial, so connection reuse is not worth leaking one
    # video's session into another.
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        with audio_path.open("rb") as fh:
            try:
                result = await asyncio.wait_for(
                    client.audio.transcriptions.create(
                        model=model,
                        file=fh,
                        languages=langs,
                        response_format="json",
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
