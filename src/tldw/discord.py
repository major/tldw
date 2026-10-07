"""Discord webhook sender for the transcript worker.

After the worker fetches a video's subtitles it posts a short digest to a
Discord channel: the title, the channel, the watch URL, and the first few
transcript lines. This module owns the formatting, the chunking, and the POST.

Messages are plain ``{"content": "..."}`` text, not embeds. Plain text is the
whole feature here, and embeds bring their own shape limits and rendering
quirks for no benefit. Discord caps a message at 2000 characters, so longer
digests are split on newline boundaries where possible and hard-sliced only
when a single line is itself longer than the cap.

The retry policy is deliberately small. A 429 is answered with one retry after
the ``Retry-After`` header (or a default when the header is missing), and any
other failure raises immediately. Bounded retries keep a Discord outage from
turning into an infinite loop inside one send; the worker handles longer
outages through its own backoff and queue schedule.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

import httpx2

__all__ = [
    "format_message",
    "chunk_message",
    "send",
]

logger = logging.getLogger(__name__)

# Discord's hard limit per message.
DEFAULT_MAX_CHARS: int = 2000

# Asking Discord to wait makes the response carry the rate-limit headers,
# including Retry-After, instead of dropping the message silently.
DEFAULT_WEBHOOK_QUERY: str = "?wait=true"

# Used when a 429 arrives without a usable Retry-After header.
DEFAULT_RETRY_AFTER_SECONDS: float = 5.0


def format_message(
    title: str,
    channel_name: str,
    url: str,
    lines: Sequence[str],
) -> str:
    """Render the Discord message body.

    Shape::

        **{title}** ({channel_name})
        {url}

        > {line1}
        > {line2}

    Each line is stripped before it is quoted and blank lines are dropped. When
    ``lines`` is empty the blockquote section is omitted entirely. No trailing
    newline is added; Discord handles the message as-is.
    """
    parts = [f"**{title}** ({channel_name})", url]
    quoted = [f"> {line.strip()}" for line in lines if line.strip()]
    if quoted:
        parts.append("")
        parts.extend(quoted)
    return "\n".join(parts)


def chunk_message(content: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split content into chunks of at most ``max_chars`` on newline boundaries.

    Lines are accumulated while they fit, and a line that would overflow starts
    a new chunk. A single line longer than ``max_chars`` cannot be split politely,
    so it is hard-sliced into ``max_chars`` pieces. Returns ``[]`` for empty
    input and at least one chunk otherwise.
    """
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    chunks: list[str] = []
    current = ""
    for line in content.split("\n"):
        if not current:
            # An empty current also means we are at a chunk boundary, where a
            # blank line carries no information, so it is dropped here.
            current = line
        elif len(current) + 1 + len(line) <= max_chars:
            current = f"{current}\n{line}"
        else:
            chunks.append(current)
            current = line
    if current:
        chunks.append(current)

    # A chunk still over the limit means a single line was too long to split on
    # a newline; slicing mid-line is the only remaining option.
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(
                chunk[start : start + max_chars]
                for start in range(0, len(chunk), max_chars)
            )
    return result


def _retry_after_seconds(response: httpx2.Response) -> float:
    """Return the Retry-After header as seconds, or the default when unusable."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return DEFAULT_RETRY_AFTER_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_RETRY_AFTER_SECONDS


async def send(
    client: httpx2.AsyncClient,
    webhook_url: str,
    content: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """POST the chunked content to the Discord webhook.

    Posts one request per chunk to ``webhook_url`` plus ``?wait=true``. A 429 on
    a chunk is slept off for ``Retry-After`` seconds and retried exactly once;
    any other error, or a second 429, raises ``httpx2.HTTPStatusError``. Empty
    content is a no-op so callers do not have to guard it.
    """
    chunks = chunk_message(content, max_chars=max_chars)
    if not chunks:
        logger.debug("discord send skipped: empty content")
        return

    url = webhook_url + DEFAULT_WEBHOOK_QUERY
    for index, chunk in enumerate(chunks):
        attempts = 0
        while True:
            attempts += 1
            response = await client.post(url, json={"content": chunk})
            try:
                response.raise_for_status()
            except httpx2.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempts == 1:
                    retry_after = _retry_after_seconds(exc.response)
                    logger.warning(
                        "discord rate limited on chunk %d; retrying in %s seconds",
                        index,
                        retry_after,
                    )
                    await sleep(retry_after)
                    continue
                raise
            logger.debug("posted discord chunk %d (%d chars)", index, len(chunk))
            break
