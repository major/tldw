"""Discord webhook sender for the transcript worker.

This module has two rendering modes. The regular digest path sends plain
``{"content": "..."}`` text: the title, the channel, the watch URL, and the
first few transcript lines. The LLM-takeaway path sends one message with four
embeds: a header embed (bottom line and watch list), then one embed per
takeaway with timestamped deep-link bullets. Plain text has a 2000 character
cap, so digests are chunked. The embed message has a 6000 character cap across
all of its embeds, which the embed builders enforce.

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
    "build_header_embed",
    "build_takeaway_embed",
    "build_takeaway_embeds",
]

logger = logging.getLogger(__name__)

# Discord's hard limit per message.
DEFAULT_MAX_CHARS: int = 2000

# Discord's hard limit on embeds in one message.
DISCORD_MAX_EMBEDS: int = 10

# Discord's hard limit on title, description, and footer text summed over all
# embeds in one message.
DISCORD_MESSAGE_MAX_CHARS: int = 6000

# Discord's per-embed caps for a title and a description.
_EMBED_TITLE_MAX_CHARS: int = 256
_EMBED_DESCRIPTION_MAX_CHARS: int = 4096
_ELLIPSIS: str = "..."

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


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` chars, ending in an ellipsis when cut.

    The ellipsis counts toward the limit, so the result never exceeds it.
    """
    if len(text) <= limit:
        return text
    return text[: limit - len(_ELLIPSIS)] + _ELLIPSIS


def _format_bullet(bullet: TakeawayBullet, video_id: str) -> str:
    """Render one bullet as a Discord markdown line.

    Bullets with a ``timestamp_seconds`` become ``- [m:ss](deep-link) text`` so
    the bullet links to the moment in the video. Bullets without one
    (audio-backend takeaways, where the transcript has no cues) render as a
    plain ``- text`` line.
    """
    if bullet.timestamp_seconds is None:
        return f"- {bullet.text}"
    ts = format_timestamp(bullet.timestamp_seconds)
    link = youtube_deep_link(video_id, bullet.timestamp_seconds)
    return f"- [{ts}]({link}) {bullet.text}"


def build_header_embed(
    takeaways: Takeaways,
    *,
    video_url: str,
    video_title: str,
    channel_name: str,
    color: int = 0x5865F2,
) -> dict[str, Any]:
    """Build the briefing header embed: the bottom line and the watch list.

    The title names the channel and the video, and the url makes it clickable.
    The description opens with the bottom line. When the model gave watch items,
    they follow under a bold **Watch** heading as a bullet list. The header has
    no footer because the channel name already appears in the title.
    """
    description = f"**Bottom line:** {takeaways.bottom_line}"
    if takeaways.watch:
        watch_md = "\n".join(f"- {item}" for item in takeaways.watch)
        description += f"\n\n**Watch**\n{watch_md}"
    return {
        "title": _truncate(f"{channel_name}: {video_title}", _EMBED_TITLE_MAX_CHARS),
        "url": video_url,
        "description": _truncate(description, _EMBED_DESCRIPTION_MAX_CHARS),
        "color": color,
    }


def build_takeaway_embed(
    takeaway: Takeaway,
    *,
    video_id: str,
    video_url: str,
    index: int,
    color: int = 0x5865F2,
) -> dict[str, Any]:
    """Build the Discord embed for one takeaway.

    The title is numbered, such as ``"1. Headline"``. The description holds the
    summary followed by one bullet per supporting point. Titles are capped at
    256 chars and descriptions at 4096 chars, and longer text is cut with an
    ellipsis. This builder sets no footer; ``build_takeaway_embeds`` puts the
    channel name on the last embed only.
    """
    bullets_md = "\n".join(_format_bullet(b, video_id) for b in takeaway.bullets)
    description = f"{takeaway.summary}\n\n{bullets_md}" if bullets_md else takeaway.summary
    return {
        "title": _truncate(f"{index}. {takeaway.title}", _EMBED_TITLE_MAX_CHARS),
        "url": video_url,
        "description": _truncate(description, _EMBED_DESCRIPTION_MAX_CHARS),
        "color": color,
    }


def _message_chars(embeds: Sequence[dict[str, Any]]) -> int:
    """Count the characters Discord counts toward the per-message cap.

    Discord sums the title, description, and footer text over every embed. URLs,
    colors, and bullet markup are not counted. A missing field counts as zero.
    """
    total = 0
    for embed in embeds:
        total += len(embed.get("title", ""))
        total += len(embed.get("description", ""))
        total += len(embed.get("footer", {}).get("text", ""))
    return total


def _assemble_embeds(
    header: dict[str, Any],
    items: Sequence[Takeaway],
    *,
    video_id: str,
    video_url: str,
    channel_name: str,
) -> list[dict[str, Any]]:
    """Return the header followed by one embed per takeaway, footer on the last.

    Takes the takeaways as a plain sequence so the trimming loop can rebuild the
    embeds with fewer bullets without changing the model itself.
    """
    embeds = [header]
    embeds.extend(
        build_takeaway_embed(item, video_id=video_id, video_url=video_url, index=i + 1)
        for i, item in enumerate(items)
    )
    embeds[-1]["footer"] = {"text": channel_name}
    return embeds


def _richest_index(items: Sequence[Takeaway]) -> int | None:
    """Return the index of the takeaway with the most bullets, or ``None``.

    Returns ``None`` when every takeaway is down to one bullet. A takeaway is
    never trimmed below one bullet, so there is nothing left to drop.
    """
    index = max(range(len(items)), key=lambda i: len(items[i].bullets))
    return index if len(items[index].bullets) > 1 else None


def _truncate_longest_descriptions(embeds: list[dict[str, Any]]) -> None:
    """Cut the longest descriptions in place until the message fits the cap.

    This is the last resort, used only after bullets run out. Each pass shortens
    the longest description by the remaining overflow, or by as much as it can,
    and keeps an ellipsis. It stops early when no description can shrink, which
    only happens when titles and footers alone exceed the cap.
    """
    overflow = _message_chars(embeds) - DISCORD_MESSAGE_MAX_CHARS
    while overflow > 0:
        longest = max(embeds, key=lambda embed: len(embed["description"]))
        description = longest["description"]
        cut = min(overflow, len(description) - len(_ELLIPSIS))
        if cut <= 0:
            return
        keep = len(description) - cut - len(_ELLIPSIS)
        longest["description"] = description[:keep] + _ELLIPSIS
        overflow -= cut


def build_takeaway_embeds(
    takeaways: Takeaways,
    *,
    video_id: str,
    video_url: str,
    video_title: str,
    channel_name: str,
) -> list[dict[str, Any]]:
    """Build the briefing as Discord embeds: the header, then one per takeaway.

    The result fits within ``DISCORD_MESSAGE_MAX_CHARS``. When it does not,
    bullets are dropped one at a time from the takeaway with the most bullets,
    so summaries stay intact and every takeaway keeps at least one bullet. Only
    when every takeaway is down to one bullet are descriptions cut with an
    ellipsis. A WARNING is logged whenever any trimming happens.
    """
    header = build_header_embed(
        takeaways,
        video_url=video_url,
        video_title=video_title,
        channel_name=channel_name,
    )
    items = list(takeaways.items)
    embeds = _assemble_embeds(
        header,
        items,
        video_id=video_id,
        video_url=video_url,
        channel_name=channel_name,
    )
    original_chars = _message_chars(embeds)
    if original_chars <= DISCORD_MESSAGE_MAX_CHARS:
        return embeds

    while _message_chars(embeds) > DISCORD_MESSAGE_MAX_CHARS:
        index = _richest_index(items)
        if index is None:
            break
        items[index] = items[index].model_copy(
            update={"bullets": items[index].bullets[:-1]}
        )
        embeds = _assemble_embeds(
            header,
            items,
            video_id=video_id,
            video_url=video_url,
            channel_name=channel_name,
        )

    # Edge case: every takeaway is at one bullet and the message is still over
    # the cap. This happens with a long bottom line, a long watch list, or an
    # unusually long channel name. Cutting text is the last resort.
    _truncate_longest_descriptions(embeds)

    logger.warning(
        "discord briefing for video=%s trimmed from %d to %d chars (limit %d)",
        video_id,
        original_chars,
        _message_chars(embeds),
        DISCORD_MESSAGE_MAX_CHARS,
    )
    return embeds


async def send_embeds(
    client: httpx2.AsyncClient,
    webhook_url: str,
    embeds: list[dict[str, Any]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Post all embeds to a Discord webhook as one message.

    Earlier versions sent one message per embed, because the feed preview shows
    only the top embed and a stacked message hid every takeaway after the first.
    The top embed is now the header (bottom line and watch list), so the preview
    shows the most useful part. One message is also all-or-nothing, and it needs
    one webhook call instead of one per embed.

    Raises ``ValueError`` when ``embeds`` is empty or has more than Discord's 10
    embeds per message. A 429 is retried once through the shared
    ``_post_payload`` helper. The caller keeps the 6000 char total under the
    limit, which ``build_takeaway_embeds`` enforces.
    """
    if not embeds:
        raise ValueError("send_embeds requires at least one embed")
    if len(embeds) > DISCORD_MAX_EMBEDS:
        raise ValueError(
            f"send_embeds accepts at most {DISCORD_MAX_EMBEDS} embeds, "
            f"got {len(embeds)}"
        )
    url = webhook_url + DEFAULT_WEBHOOK_QUERY
    await _post_payload(client, url, {"embeds": embeds}, sleep=sleep)
    logger.debug("posted discord embeds message (%d embeds)", len(embeds))
