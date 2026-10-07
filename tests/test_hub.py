"""Tests for the PubSubHubbub subscriber in tldw.hub."""

from __future__ import annotations

from urllib.parse import quote

import httpx2
import pytest

from tldw.hub import (
    DEFAULT_LEASE_SECONDS,
    HUB_URL,
    build_subscribe_form,
    subscribe,
    topic_url,
)

CHANNEL_ID = "UC_ywfvIR2JrnMuZt33y7QYQ"
CALLBACK_URL = "https://cb.example/pubsub/callback"


@pytest.mark.parametrize(
    ("channel_id", "expected"),
    [
        pytest.param(
            CHANNEL_ID,
            "https://www.youtube.com/xml/feeds/videos.xml?"
            "channel_id=UC_ywfvIR2JrnMuZt33y7QYQ",
            id="youtube-channel-id",
        ),
        pytest.param(
            "UC/odd&id=1",
            "https://www.youtube.com/xml/feeds/videos.xml?channel_id=UC/odd&id=1",
            id="channel-id-passed-through",
        ),
    ],
)
def test_topic_url_includes_xml_path(channel_id: str, expected: str) -> None:
    """The topic URL keeps the load-bearing /xml/ segment and passes the id raw."""
    # Arrange
    # channel_id and expected come from the parametrized matrix above.

    # Act
    result = topic_url(channel_id)

    # Assert
    assert result == expected


def test_build_subscribe_form_required_fields_default() -> None:
    """The default form carries every required field and no secret.

    The topic is stored raw and encoded exactly once on the wire. Storing an
    already-encoded value would make httpx encode the percent signs again and
    the hub would receive a double-encoded topic it cannot parse.
    """
    # Arrange
    full_topic = topic_url(CHANNEL_ID)
    encoded_topic = quote(full_topic, safe="")

    # Act
    form = build_subscribe_form(CHANNEL_ID, CALLBACK_URL)

    # Assert
    assert isinstance(form, dict)
    for key in (
        "hub.mode",
        "hub.topic",
        "hub.callback",
        "hub.verify",
        "hub.lease_seconds",
    ):
        assert key in form
    assert form["hub.mode"] == "subscribe"
    assert form["hub.verify"] == "async"
    assert form["hub.lease_seconds"] == str(DEFAULT_LEASE_SECONDS)
    assert form["hub.lease_seconds"] == "432000"
    assert form["hub.topic"] == full_topic
    assert quote(form["hub.topic"], safe="") == encoded_topic
    assert form["hub.callback"] == CALLBACK_URL
    assert "hub.secret" not in form


def test_build_subscribe_form_includes_secret_when_provided() -> None:
    """A supplied secret is added to the form verbatim."""
    # Arrange
    secret = "topsecret"

    # Act
    form = build_subscribe_form(CHANNEL_ID, CALLBACK_URL, secret=secret)

    # Assert
    assert form["hub.secret"] == secret


def test_build_subscribe_form_can_set_unsubscribe_mode() -> None:
    """The mode field can be switched to unsubscribe."""
    # Arrange
    mode = "unsubscribe"

    # Act
    form = build_subscribe_form(CHANNEL_ID, CALLBACK_URL, mode=mode)

    # Assert
    assert form["hub.mode"] == "unsubscribe"


def test_build_subscribe_form_topic_is_url_encoded() -> None:
    """Encoding the topic value once yields the form body the hub expects."""
    # Arrange
    full_topic = topic_url(CHANNEL_ID)
    expected = quote(full_topic, safe="")

    # Act
    form = build_subscribe_form(CHANNEL_ID, CALLBACK_URL)

    # Assert
    topic_value = form["hub.topic"]
    assert isinstance(topic_value, str)
    assert quote(topic_value, safe="") == expected


async def test_subscribe_posts_form_to_hub_and_returns_202() -> None:
    """subscribe POSTs the encoded form to the hub and returns the status code."""
    # Arrange
    captured: list[httpx2.Request] = []
    full_topic = topic_url(CHANNEL_ID)
    expected_topic = quote(full_topic, safe="")

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(202)

    transport = httpx2.MockTransport(handler)

    # Act
    async with httpx2.AsyncClient(transport=transport) as client:
        status = await subscribe(client, CHANNEL_ID, CALLBACK_URL)

    # Assert
    assert status == 202
    assert len(captured) == 1
    assert captured[0].method == "POST"
    url = str(captured[0].url)
    assert url == HUB_URL
    assert "pubsubhubbub.appspot.com" in url
    assert httpx2.URL(url).path == "/subscribe"
    body = captured[0].content.decode("latin-1")
    assert "hub.mode=subscribe" in body
    assert expected_topic in body
    assert "hub.callback=https%3A%2F%2Fcb.example%2Fpubsub%2Fcallback" in body
    assert "hub.verify=async" in body
    assert "hub.lease_seconds=432000" in body
    assert "hub.secret=" not in body


async def test_subscribe_includes_secret_when_configured() -> None:
    """A configured secret travels in the POSTed form body."""
    # Arrange
    captured: list[httpx2.Request] = []
    secret = "topsecret"

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(202)

    transport = httpx2.MockTransport(handler)

    # Act
    async with httpx2.AsyncClient(transport=transport) as client:
        status = await subscribe(client, CHANNEL_ID, CALLBACK_URL, secret=secret)

    # Assert
    assert status == 202
    body = captured[0].content.decode("latin-1")
    assert "hub.secret=topsecret" in body


async def test_subscribe_raises_on_non_2xx() -> None:
    """A non-2xx hub response surfaces as an HTTPStatusError."""
    # Arrange
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, text="bad topic")

    transport = httpx2.MockTransport(handler)

    # Act
    async with httpx2.AsyncClient(transport=transport) as client:
        # Assert
        with pytest.raises(httpx2.HTTPStatusError):
            await subscribe(client, CHANNEL_ID, CALLBACK_URL)
