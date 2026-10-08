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


def _three_sample_embeds() -> list[dict]:
    """Return three minimal embed dicts with distinguishable titles."""
    return [
        {"title": "First", "description": "D1", "color": 0x5865F2},
        {"title": "Second", "description": "D2", "color": 0x5865F2},
        {"title": "Third", "description": "D3", "color": 0x5865F2},
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


async def test_send_embeds_posts_one_request_per_embed() -> None:
    """The happy path posts one request per embed so each is a feed item."""
    captured: list[httpx2.Request] = []
    embeds = _three_sample_embeds()

    await _send_embeds_once(captured, [], embeds)

    assert len(captured) == len(embeds)
    for request, embed in zip(captured, embeds):
        body = json.loads(request.content.decode("utf-8"))
        assert body == {"embeds": [embed]}


async def test_send_embeds_posts_to_webhook_url_with_wait_true() -> None:
    """Every request targets the webhook URL with ?wait=true appended."""
    captured: list[httpx2.Request] = []

    await _send_embeds_once(captured, [], _three_sample_embeds())

    assert len(captured) == 3
    for request in captured:
        assert str(request.url).endswith("?wait=true")


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


async def test_send_embeds_raises_on_empty_embeds() -> None:
    """An empty embed list is a caller bug, not a no-op."""
    captured: list[httpx2.Request] = []
    transport = make_capturing_transport(captured, [], default_status=204)
    async with httpx2.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError):
            await send_embeds(client, _WEBHOOK, [])

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
