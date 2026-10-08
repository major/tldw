"""Tests for the tldw Discord webhook sender.

The pure helpers (``format_message`` and ``chunk_message``) are tested directly.
``send`` is driven through an ``httpx2.MockTransport`` that records every
request and replays a scripted list of responses, so the retry and error paths
are exercised without touching the network. The real ``asyncio.sleep`` is never
used: retry tests inject a fake that records the delays.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx2
import pytest

from helpers import make_capturing_transport, recording_sleep
from tldw.discord import (
    DEFAULT_MAX_CHARS,
    DEFAULT_RETRY_AFTER_SECONDS,
    chunk_message,
    format_message,
    send,
)

_WEBHOOK = "https://discord.example/api/webhooks/123/abc"


async def _send_once(
    captured: list[httpx2.Request],
    responses: list[httpx2.Response],
    content: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Run one send against a recording MockTransport."""
    transport = make_capturing_transport(
        captured, responses, default_status=200, default_json={"id": "x"}
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        if sleep is None:
            await send(client, _WEBHOOK, content, max_chars=max_chars)
        else:
            await send(client, _WEBHOOK, content, max_chars=max_chars, sleep=sleep)


# ---------------------------------------------------------------------------
# format_message
# ---------------------------------------------------------------------------


def test_format_message_contains_title_channel_and_url() -> None:
    """The header carries the title and channel, and the URL follows."""
    # Arrange
    # A single transcript line keeps the assertion focused on the header.

    # Act
    message = format_message("My Video", "My Channel", "https://youtu.be/x", ["hi"])

    # Assert
    assert "My Video" in message
    assert "My Channel" in message
    assert "https://youtu.be/x" in message


def test_format_message_quotes_each_line_with_blockquote() -> None:
    """Every transcript line is rendered as a Discord blockquote."""
    # Arrange
    lines = ["a", "b", "c"]

    # Act
    message = format_message("T", "C", "https://youtu.be/x", lines)

    # Assert
    assert "> a" in message
    assert "> b" in message
    assert "> c" in message


def test_format_message_omits_blockquote_section_when_lines_empty() -> None:
    """No transcript lines means no blockquote section at all."""
    # Arrange
    # An empty line list is the no-transcript case.

    # Act
    message = format_message("T", "C", "https://youtu.be/x", [])

    # Assert
    assert "> " not in message
    assert "T" in message
    assert "https://youtu.be/x" in message


def test_format_message_strips_lines_before_quoting() -> None:
    """Surrounding whitespace never leaks into the blockquote."""
    # Arrange
    lines = ["  spaced  ", "\ttabbed\t"]

    # Act
    message = format_message("T", "C", "https://youtu.be/x", lines)

    # Assert
    assert "> spaced" in message
    assert "> tabbed" in message
    assert ">   spaced" not in message


# ---------------------------------------------------------------------------
# chunk_message
# ---------------------------------------------------------------------------


def test_chunk_message_returns_single_chunk_when_under_limit() -> None:
    """Content that fits is returned unchanged as one chunk."""
    # Arrange
    content = "x" * 1500

    # Act
    chunks = chunk_message(content, max_chars=2000)

    # Assert
    assert chunks == [content]


def test_chunk_message_splits_on_newline_boundary() -> None:
    """Five 500-char lines split into two chunks without breaking a line."""
    # Arrange
    content = "\n".join(["a" * 500] * 5)

    # Act
    chunks = chunk_message(content, max_chars=2000)

    # Assert
    assert len(chunks) == 2
    assert all(len(chunk) <= 2000 for chunk in chunks)
    # Rejoining on newlines must reproduce the input exactly, proving no line
    # was split.
    assert "\n".join(chunks) == content


def test_chunk_message_returns_empty_list_for_empty_input() -> None:
    """Empty content produces no chunks."""
    # Arrange
    # An empty string is the no-op case for send.

    # Act
    chunks = chunk_message("")

    # Assert
    assert chunks == []


def test_chunk_message_hard_slices_single_overlong_line() -> None:
    """A line longer than the limit is sliced into fixed-size pieces."""
    # Arrange
    content = "x" * 5000

    # Act
    chunks = chunk_message(content, max_chars=2000)

    # Assert
    assert [len(chunk) for chunk in chunks] == [2000, 2000, 1000]
    assert "".join(chunks) == content


def test_chunk_message_preserves_empty_intermediate_lines() -> None:
    """A blank line inside the content survives the accumulate step."""
    # Arrange
    # max_chars=8 forces a split while the blank line is still interior.
    content = "line1\n\nline2"

    # Act
    chunks = chunk_message(content, max_chars=8)

    # Assert
    # The blank line is kept as the trailing newline of the first chunk. A blank
    # line that lands exactly on a chunk boundary is dropped, which is harmless.
    assert chunks == ["line1\n", "line2"]


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------


async def test_send_posts_one_request_per_chunk() -> None:
    """Three chunks produce three POSTs."""
    # Arrange
    captured: list[httpx2.Request] = []
    content = "aaaa\nbbbb\ncccc"

    # Act
    await _send_once(captured, [], content, max_chars=4)

    # Assert
    assert len(captured) == 3


async def test_send_posts_to_webhook_url_with_wait_true() -> None:
    """Every request targets the webhook URL with ?wait=true appended."""
    # Arrange
    captured: list[httpx2.Request] = []

    # Act
    await _send_once(captured, [], "hello")

    # Assert
    assert len(captured) == 1
    assert str(captured[0].url).endswith("?wait=true")


async def test_send_posts_json_with_content_key() -> None:
    """Each POST body is JSON with the chunk under the content key."""
    # Arrange
    captured: list[httpx2.Request] = []
    content = "aaaa\nbbbb\ncccc"
    expected = chunk_message(content, max_chars=4)

    # Act
    await _send_once(captured, [], content, max_chars=4)

    # Assert
    bodies = [json.loads(request.content.decode("utf-8")) for request in captured]
    assert [body["content"] for body in bodies] == expected


async def test_send_returns_when_content_empty() -> None:
    """Empty content makes send a no-op with no requests."""
    # Arrange
    captured: list[httpx2.Request] = []

    # Act
    await _send_once(captured, [], "")

    # Assert
    assert captured == []


@pytest.mark.parametrize("status_code", [400, 404, 500])
async def test_send_raises_without_retry_on_non_429_errors(
    status_code: int,
) -> None:
    """A non-429 error is not retried and raises after one request."""
    # Arrange
    captured: list[httpx2.Request] = []
    responses = [httpx2.Response(status_code, text="bad")]

    # Act / Assert
    with pytest.raises(httpx2.HTTPStatusError):
        await _send_once(captured, responses, "hello")
    assert len(captured) == 1


async def test_send_retries_once_on_429_with_retry_after() -> None:
    """A 429 is retried once after sleeping for the Retry-After header."""
    # Arrange
    captured: list[httpx2.Request] = []
    responses = [
        httpx2.Response(429, headers={"Retry-After": "1.5"}),
        httpx2.Response(200, json={"id": "x"}),
    ]
    delays, sleep = recording_sleep()

    # Act
    await _send_once(captured, responses, "hello", sleep=sleep)

    # Assert
    assert len(captured) == 2
    assert delays == [1.5]


async def test_send_uses_default_retry_after_when_header_missing() -> None:
    """A 429 without Retry-After falls back to the default delay."""
    # Arrange
    captured: list[httpx2.Request] = []
    responses = [
        httpx2.Response(429),
        httpx2.Response(200, json={"id": "x"}),
    ]
    delays, sleep = recording_sleep()

    # Act
    await _send_once(captured, responses, "hello", sleep=sleep)

    # Assert
    assert len(captured) == 2
    assert delays == [DEFAULT_RETRY_AFTER_SECONDS]


async def test_send_raises_after_second_429() -> None:
    """A second 429 on the same chunk is not retried again."""
    # Arrange
    captured: list[httpx2.Request] = []
    responses = [httpx2.Response(429), httpx2.Response(429)]
    delays, sleep = recording_sleep()

    # Act / Assert
    with pytest.raises(httpx2.HTTPStatusError):
        await _send_once(captured, responses, "hello", sleep=sleep)
    assert len(captured) == 2
    assert delays == [DEFAULT_RETRY_AFTER_SECONDS]


async def test_send_uses_default_retry_after_when_header_is_garbage() -> None:
    """An unparseable Retry-After header falls back to the default delay."""
    # Arrange
    captured: list[httpx2.Request] = []
    responses = [
        httpx2.Response(429, headers={"Retry-After": "soon"}),
        httpx2.Response(200, json={"id": "x"}),
    ]
    delays, sleep = recording_sleep()

    # Act
    await _send_once(captured, responses, "hello", sleep=sleep)

    # Assert
    assert len(captured) == 2
    assert delays == [DEFAULT_RETRY_AFTER_SECONDS]
