"""OpenAI ``gpt-transcribe`` client for tldw.

This module is a thin async wrapper around ``client.audio.transcriptions.create``
from the official ``openai`` SDK. Chunks are sent sequentially under one
deadline; retries and backoff are the worker's job, not this module's.

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
import re
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

from tldw.transcription_audio import (
    PREPARATION_REASONS,
    AudioPreparationError,
    audio_chunks,
)

__all__ = ["transcribe", "TranscribeError", "DEFAULT_TIMEOUT_SECONDS"]

logger = logging.getLogger(__name__)

# Transcription can take a while for long audio, so the default is generous.
# The worker passes its configured timeout explicitly; this is the fallback.
DEFAULT_TIMEOUT_SECONDS: float = 600.0

# Hardcoded input languages for gpt-transcribe. The OpenAI docs require the
# plural ``languages`` field (singular ``language=`` is for older models),
# and tldw's contract is: US English always. No caller can override this.
TRANSCRIBE_LANGUAGES: tuple[str, ...] = ("en",)

REASONS = {
    **PREPARATION_REASONS,
    "duration_limit": "Audio exceeds the provider duration limit.",
    "size_limit": "Audio exceeds the provider upload size limit.",
    "context_limit": "Audio exceeds the provider context limit.",
    "invalid_audio": "Provider could not decode the audio.",
    "unsupported_format": "Provider does not support the audio format.",
}


class TranscribeError(RuntimeError):
    """Raised when transcription fails in a way the worker can classify.

    ``status_code`` carries the HTTP status from an ``APIStatusError`` (for
    example 429 for rate limiting) so the worker can choose between a
    RATE_LIMITED retry and GIVE_UP_AUDIO. It is ``None`` for local failures
    such as a missing API key.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        reason: str | None = None,
        permanent: bool = False,
    ) -> None:
        """Store the message and optional HTTP status code."""
        super().__init__(message)
        self.status_code = status_code
        self.permanent = permanent
        self.reason = REASONS.get(reason) if reason is not None else None
        self.error_code = (
            error_code
            if error_code
            in {"credit_balance_exhausted", "insufficient_quota", "rate_limit_exceeded"}
            else None
        )


def _provider_reason(body: object) -> str | None:
    """Classify bounded provider fields without exposing their contents."""
    if not isinstance(body, dict):
        return None
    if isinstance(body.get("error"), dict):
        body = body["error"]
    codes = {
        "context_length_exceeded": "context_limit",
        "audio_duration_exceeded": "duration_limit",
        "file_too_large": "size_limit",
        "invalid_audio": "invalid_audio",
        "unsupported_format": "unsupported_format",
    }
    code = body.get("code")
    if isinstance(code, str) and code in codes:
        return codes[code]
    message = body.get("message")
    if not isinstance(message, str):
        return None
    message = message[:2000].lower()
    patterns = (
        (
            "context_limit",
            r"context[_ ]length[_ ]exceeded|context (?:length|window).{0,80}(?:exceed|too long|maximum)",
        ),
        (
            "duration_limit",
            r"(?:audio )?duration.{0,100}(?:exceed|too long|maximum|limit)|(?:maximum|max).{0,50}duration.{0,50}(?:exceed|too long)",
        ),
        (
            "size_limit",
            r"(?:file|upload|audio).{0,60}(?:too large|size.{0,40}(?:exceed|limit|maximum))",
        ),
        (
            "invalid_audio",
            r"(?:cannot|could not|unable to|failed to) decode|(?:corrupt|invalid) audio",
        ),
        (
            "unsupported_format",
            r"unsupported (?:audio )?format|(?:audio )?format.{0,40}(?:not supported|unsupported)",
        ),
    )
    return next(
        (category for category, pattern in patterns if re.search(pattern, message)),
        None,
    )


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

    Returns the plain transcript text. The worker owns retry
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

    try:
        async with asyncio.timeout(timeout_s):
            async with audio_chunks(audio_path) as chunks:
                async with AsyncOpenAI(
                    api_key=api_key, base_url=base_url, max_retries=0
                ) as client:
                    transcripts = []
                    for index, chunk in enumerate(chunks, start=1):
                        try:
                            with chunk.open("rb") as fh:
                                logger.info(
                                    "transcribing chunk %s/%s bytes=%s",
                                    index,
                                    len(chunks),
                                    chunk.stat().st_size,
                                )
                                result = await client.audio.transcriptions.create(
                                    file=fh, **call_kwargs
                                )
                        except OSError as exc:
                            raise AudioPreparationError() from exc
                        transcripts.append(result.text)
                    return "\n\n".join(transcripts)
    except AudioPreparationError as exc:
        raise TranscribeError(
            str(exc), reason=exc.reason, permanent=exc.permanent
        ) from exc
    except openai.APIStatusError as exc:
        body = exc.body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]
        code = body.get("code") if isinstance(body, dict) else None
        if not isinstance(code, str):
            code = None
        if isinstance(body, dict) and body.get("type") == "insufficient_quota":
            code = "insufficient_quota"
        category = _provider_reason(body)
        reason = REASONS.get(category) if category else None
        message = f"transcription failed with status {exc.status_code}"
        if reason:
            message += f": {reason}"
        raise TranscribeError(
            message,
            status_code=exc.status_code,
            error_code=code,
            reason=category,
        ) from exc
