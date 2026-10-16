"""Tests for ``send_embeds`` and the shared webhook POST helper.

Every request runs through an ``httpx2.MockTransport`` that records the request
and replays a scripted list of responses, so the retry path is exercised without
the network. Retry tests inject a fake sleep that records delays instead of
waiting.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import httpx2
import pytest

from helpers import make_capturing_transport, recording_sleep
from tldw.discord import send, send_embeds

_WEBHOOK = "https://discord.example/api/webhooks/123/abc"


def _sample_embeds() -> list[dict]:
    """Return one minimal embed dict, enough to shape the request body."""
    return [{"title": "T", "description": "D", "color": 0x5865F2}]


def _embeds(count: int) -> list[dict]:
    """Return ``count`` minimal embeds with distinguishable titles."""
    return [
        {"title": f"Embed {n}", "description": f"D{n}", "color": 0x5865F2}
        for n in range(count)
    ]


async def _send_embeds_once(
    captured: list[httpx2.Request],
    responses: list[httpx2.Response],
    embeds: list[dict],
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Run one send_embeds against a recording MockTransport."""
    transport = make_capturing_transport(captured, responses, default_status=204)
    async with httpx2.AsyncClient(transport=transport) as client:
        if sleep is None:
            await send_embeds(client, _WEBHOOK, embeds)
        else:
            await send_embeds(client, _WEBHOOK, embeds, sleep=sleep)


# ---------------------------------------------------------------------------
# send_embeds
# ---------------------------------------------------------------------------


async def test_send_embeds_posts_all_embeds_in_one_request() -> None:
    """The whole briefing goes out as one request carrying every embed."""
    # Arrange
    captured: list[httpx2.Request] = []
    embeds = _embeds(4)

    # Act
    await _send_embeds_once(captured, [], embeds)

    # Assert
    assert len(captured) == 1
    body = json.loads(captured[0].content.decode("utf-8"))
    assert body == {"embeds": embeds}


async def test_send_embeds_posts_to_webhook_url_with_wait_true() -> None:
    """The request targets the webhook URL with ?wait=true appended."""
    captured: list[httpx2.Request] = []

    await _send_embeds_once(captured, [], _embeds(4))

    assert len(captured) == 1
    assert str(captured[0].url).endswith("?wait=true")


@pytest.mark.parametrize("count", [1, 10])
async def test_send_embeds_accepts_between_one_and_ten_embeds(count: int) -> None:
    """Discord's per-message embed range, 1 to 10 inclusive, is accepted."""
    captured: list[httpx2.Request] = []

    await _send_embeds_once(captured, [], _embeds(count))

    assert len(captured) == 1


async def test_send_embeds_retries_once_on_429_with_retry_after() -> None:
    """A 429 is retried once after sleeping for the Retry-After value."""
    captured: list[httpx2.Request] = []
    responses = [
        httpx2.Response(429, headers={"Retry-After": "0"}),
        httpx2.Response(204),
    ]
    delays, sleep = recording_sleep()

    await _send_embeds_once(captured, responses, _sample_embeds(), sleep=sleep)

    assert len(captured) == 2
    assert delays == [0.0]


async def test_send_embeds_raises_after_second_429() -> None:
    """A second 429 on the same post is not retried again."""
    captured: list[httpx2.Request] = []
    responses = [httpx2.Response(429), httpx2.Response(429)]
    delays, sleep = recording_sleep()

    with pytest.raises(httpx2.HTTPStatusError):
        await _send_embeds_once(captured, responses, _sample_embeds(), sleep=sleep)
    assert len(captured) == 2


@pytest.mark.parametrize("count", [0, 11], ids=["empty", "over_limit"])
async def test_send_embeds_rejects_invalid_embed_counts(count: int) -> None:
    """Zero or more than 10 embeds is a caller bug and sends nothing."""
    captured: list[httpx2.Request] = []
    transport = make_capturing_transport(captured, [], default_status=204)

    async with httpx2.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError):
            await send_embeds(client, _WEBHOOK, _embeds(count))

    assert captured == []


# ---------------------------------------------------------------------------
# backward compatibility of the plain send() path
# ---------------------------------------------------------------------------


async def test_send_still_posts_content_body() -> None:
    """The refactored send() still posts a {"content": ...} body."""
    captured: list[httpx2.Request] = []
    transport = make_capturing_transport(captured, [], default_status=204)

    async with httpx2.AsyncClient(transport=transport) as client:
        await send(client, _WEBHOOK, "hello")

    assert len(captured) == 1
    body = json.loads(captured[0].content.decode("utf-8"))
    assert body == {"content": "hello"}
