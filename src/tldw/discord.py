"""Discord webhook sender for the transcript worker.

This module has two rendering modes. The regular digest path sends plain
``{"content": "..."}`` text: the title, the channel, the watch URL, and the
first few transcript lines. The LLM-takeaway path sends embeds instead, one per
takeaway, with timestamped deep-link bullets. Plain text has a 2000 character
cap, so digests are chunked; embeds have their own per-description and
per-message caps, which the embed builders respect.

Both paths share one retry policy. A 429 is answered with one retry after the
``Retry-After`` header (or a default when the header is missing), and any other
failure raises immediately. Bounded retries keep a Discord outage from turning
into an infinite loop inside one send; the worker handles longer outages through
its own backoff and queue schedule.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx2

from tldw.llm import Takeaway, TakeawayBullet, Takeaways
from tldw.youtube import format_timestamp, youtube_deep_link

__all__ = [
    "format_message",
    "chunk_message",
    "send",
    "send_embeds",
    "build_takeaway_embed",
    "build_takeaway_embeds",
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


async def _post_payload(
    client: httpx2.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """POST a JSON payload to a Discord webhook with one retry on 429.

    Honors the Retry-After header on 429 (seconds). All other non-2xx responses
    raise. Caller is responsible for payload size and shape.
    """
    attempts = 0
    while True:
        attempts += 1
        response = await client.post(url, json=payload)
        try:
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempts == 1:
                retry_after = _retry_after_seconds(exc.response)
                logger.warning(
                    "discord rate limited; retrying in %s seconds", retry_after
                )
                await sleep(retry_after)
                continue
            raise
        return


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
        await _post_payload(client, url, {"content": chunk}, sleep=sleep)
        logger.debug("posted discord chunk %d (%d chars)", index, len(chunk))


def _format_bullet(bullet: TakeawayBullet, video_id: str) -> str:
    """Render one bullet as '- [m:ss](deep-link) text' for Discord markdown."""
    ts = format_timestamp(bullet.timestamp_seconds)
    link = youtube_deep_link(video_id, bullet.timestamp_seconds)
    return f"- [{ts}]({link}) {bullet.text}"


def build_takeaway_embed(
    takeaway: Takeaway,
    *,
    video_id: str,
    video_url: str,
    channel_name: str,
    index: int,
    total: int = 3,
    color: int = 0x5865F2,
) -> dict[str, Any]:
    """Build a single Discord embed dict for one takeaway.

    Discord limits: 4096 chars per description, 6000 chars total per message
    (across all embeds), 10 embeds max. This builder truncates the description
    with an ellipsis if it would exceed 4096 chars. The caller is responsible
    for keeping the total across all embeds under 6000.
    """
    bullets_md = "\n".join(_format_bullet(b, video_id) for b in takeaway.bullets)
    description = f"{takeaway.summary}\n\n{bullets_md}" if bullets_md else takeaway.summary
    if len(description) > 4096:
        description = description[:4093] + "..."
    return {
        "title": takeaway.title,
        "url": video_url,
        "description": description,
        "color": color,
        "footer": {"text": f"Takeaway {index} of {total} · {channel_name}"},
    }


def build_takeaway_embeds(
    takeaways: Takeaways,
    *,
    video_id: str,
    video_url: str,
    channel_name: str,
) -> list[dict[str, Any]]:
    """Build all 3 takeaway embeds for a video."""
    total = len(takeaways.items)
    return [
        build_takeaway_embed(
            t,
            video_id=video_id,
            video_url=video_url,
            channel_name=channel_name,
            index=i + 1,
            total=total,
        )
        for i, t in enumerate(takeaways.items)
    ]


async def send_embeds(
    client: httpx2.AsyncClient,
    webhook_url: str,
    embeds: list[dict[str, Any]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Post embeds to a Discord webhook. Atomic single call.

    Discord returns HTTP 400 for >10 embeds or >6000 total chars; the builder
    is responsible for staying under the limits. This function just posts via
    the shared _post_payload helper (same 429-retry semantics as send()).
    """
    if not embeds:
        raise ValueError("send_embeds requires at least one embed")
    await _post_payload(
        client, webhook_url + DEFAULT_WEBHOOK_QUERY, {"embeds": embeds}, sleep=sleep
    )
