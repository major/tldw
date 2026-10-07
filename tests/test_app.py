"""Tests for the tldw FastAPI application factory.

These tests pin down both halves of the PubSubHubbub callback. The GET half
answers the hub verification challenge with the challenge echoed as plain text,
and anything malformed returns 404. The POST half accepts a delivery, verifies
the optional HMAC signature, parses the Atom feed, and prints one line per
video. The POST tests are sociable: they drive the real ``feed.parse_atom`` and
the real ``renderer.format_video_line`` and observe stdout, so they exercise the
whole path instead of mocking the parser or the renderer.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx2
import pytest
from fastapi.testclient import TestClient

from tldw.app import _renewal_loop, create_app, renewal_delay
from tldw.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Build Settings pointing at a channels file that does not exist."""
    # channel_ids_file carries an alias, so pass kwargs through a mapping the
    # way test_config.py does. That keeps the field name readable here without
    # tripping the type checker on the aliased constructor parameter.
    kwargs: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
    }
    return Settings(**kwargs)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Yield a TestClient for the application. Uses the context manager so lifespan runs."""
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_get_subscribe_challenge_echoed(client: TestClient) -> None:
    """A subscribe challenge is echoed exactly as plain text with status 200."""
    # Arrange
    params = {"hub.mode": "subscribe", "hub.challenge": "abc123"}

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "abc123"


def test_get_unsubscribe_challenge_echoed(client: TestClient) -> None:
    """An unsubscribe challenge is echoed exactly as plain text with status 200."""
    # Arrange
    params = {"hub.mode": "unsubscribe", "hub.challenge": "abc123"}

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "abc123"


def test_get_missing_challenge_returns_404(client: TestClient) -> None:
    """A subscribe request without a challenge is not a verification, so 404."""
    # Arrange
    params = {"hub.mode": "subscribe"}

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 404


def test_get_missing_mode_returns_404(client: TestClient) -> None:
    """A challenge without a mode is not a verification, so 404."""
    # Arrange
    params = {"hub.challenge": "abc123"}

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 404


def test_get_unknown_mode_returns_404(client: TestClient) -> None:
    """An unknown mode is rejected even when a challenge is present."""
    # Arrange
    params = {"hub.mode": "banana", "hub.challenge": "abc123"}

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 404


def test_get_extra_query_params_are_ignored(client: TestClient) -> None:
    """Topic and lease params do not change the verification response."""
    # Arrange
    params = {
        "hub.mode": "subscribe",
        "hub.challenge": "abc123",
        "hub.topic": (
            "https://www.youtube.com/xml/feeds/videos.xml"
            "?channel_id=UC_x5XG1OV2P6uZZ5FSM9Ttw"
        ),
        "hub.lease_seconds": "432000",
    }

    # Act
    response = client.get("/pubsub/callback", params=params)

    # Assert
    assert response.status_code == 200
    assert response.text == "abc123"


def test_get_no_params_returns_404(client: TestClient) -> None:
    """A bare GET without any query string is not a verification, so 404."""
    # Arrange
    # No params are sent.

    # Act
    response = client.get("/pubsub/callback")

    # Assert
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST delivery fixtures and tests.
#
# These are sociable: the request flows through the real feed.parse_atom and
# the real renderer.format_video_line and the handler prints to stdout. The
# tests assert on the captured stdout, so they pin the whole pipeline rather
# than a mocked seam.
# ---------------------------------------------------------------------------

_HMAC_SECRET = "topsecret"


def _sign(body: bytes, secret: str = _HMAC_SECRET) -> str:
    """Return the ``sha1=`` HMAC header value the hub would send for ``body``."""
    return "sha1=" + _hmac.new(secret.encode("utf-8"), body, hashlib.sha1).hexdigest()


@pytest.fixture
def settings_with_secret() -> Settings:
    """Settings configured with an HMAC secret."""
    kwargs: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
        "hub_secret": _HMAC_SECRET,
    }
    return Settings(**kwargs)


@pytest.fixture
def settings_without_secret() -> Settings:
    """Settings with hub_secret=None (the default)."""
    kwargs: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
    }
    return Settings(**kwargs)


@pytest.fixture
def client_with_secret(settings_with_secret: Settings) -> Iterator[TestClient]:
    """Yield a TestClient whose app requires an HMAC signature."""
    with TestClient(create_app(settings_with_secret)) as c:
        yield c


@pytest.fixture
def client_without_secret(settings_without_secret: Settings) -> Iterator[TestClient]:
    """Yield a TestClient whose app accepts unsigned deliveries."""
    with TestClient(create_app(settings_without_secret)) as c:
        yield c


_EMPTY_ATOM = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
)
_MALFORMED_ATOM = b"<feed><entry>"  # unclosed tag


def test_post_signed_multi_entry_prints_each_video(
    client_with_secret: TestClient,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A signed two-entry feed prints one rendered line per entry and returns 200."""
    # Arrange
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    response = client_with_secret.post(
        "/pubsub/callback", content=multi_entry_payload, headers=headers
    )

    # Assert
    out = capsys.readouterr().out
    assert response.status_code == 200
    assert "Fixture Video One" in out
    assert "Fixture Video Two" in out
    assert "[YouTube] Fixture Channel:" in out


def test_post_signed_real_payload_prints_first_video(
    client_with_secret: TestClient,
    real_atom_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The captured real feed parses and renders end to end after signing."""
    # Arrange
    headers = {"X-Hub-Signature": _sign(real_atom_payload)}

    # Act
    response = client_with_secret.post(
        "/pubsub/callback", content=real_atom_payload, headers=headers
    )

    # Assert
    out = capsys.readouterr().out
    assert response.status_code == 200
    assert "Google for Developers" in out
    assert "lTQHImoeuEY" in out


def test_post_unsigned_with_secret_returns_403_and_prints_nothing(
    client_with_secret: TestClient,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A secret is configured but no signature is sent, so the delivery is rejected."""
    # Arrange
    # No X-Hub-Signature header is sent.

    # Act
    with caplog.at_level(logging.WARNING):
        response = client_with_secret.post(
            "/pubsub/callback", content=multi_entry_payload
        )

    # Assert
    assert response.status_code == 403
    assert capsys.readouterr().out == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_post_wrong_signature_with_secret_returns_403(
    client_with_secret: TestClient,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A signature that does not match the body is rejected without printing."""
    # Arrange
    headers = {"X-Hub-Signature": "sha1=" + "deadbeef" * 5}

    # Act
    with caplog.at_level(logging.WARNING):
        response = client_with_secret.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    assert response.status_code == 403
    assert capsys.readouterr().out == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_post_unsigned_without_secret_prints_lines(
    client_without_secret: TestClient,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a configured secret an unsigned delivery is still processed."""
    # Arrange
    # No secret and no signature: the operator opted out of HMAC checking.

    # Act
    response = client_without_secret.post(
        "/pubsub/callback", content=multi_entry_payload
    )

    # Assert
    assert response.status_code == 200
    assert "Fixture Video One" in capsys.readouterr().out


def test_post_malformed_xml_returns_200_and_logs(
    client_with_secret: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed XML is logged and answered 200 so the hub does not retry forever."""
    # Arrange
    headers = {"X-Hub-Signature": _sign(_MALFORMED_ATOM)}

    # Act
    with caplog.at_level(logging.WARNING):
        response = client_with_secret.post(
            "/pubsub/callback", content=_MALFORMED_ATOM, headers=headers
        )

    # Assert
    assert response.status_code == 200
    assert any(
        record.levelno == logging.WARNING and "parse" in record.getMessage().lower()
        for record in caplog.records
    )


def test_post_empty_feed_returns_200_and_prints_nothing(
    client_with_secret: TestClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A signed feed with no entries is accepted and prints nothing."""
    # Arrange
    headers = {"X-Hub-Signature": _sign(_EMPTY_ATOM)}

    # Act
    response = client_with_secret.post(
        "/pubsub/callback", content=_EMPTY_ATOM, headers=headers
    )

    # Assert
    assert response.status_code == 200
    assert capsys.readouterr().out == ""


def test_post_signature_with_wrong_prefix_is_rejected(
    client_with_secret: TestClient,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A signature with a non-sha1 algorithm prefix is treated as missing."""
    # Arrange
    headers = {"X-Hub-Signature": "md5=" + "deadbeef" * 5}

    # Act
    with caplog.at_level(logging.WARNING):
        response = client_with_secret.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    assert response.status_code == 403
    assert capsys.readouterr().out == ""
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# Lifespan subscription fixtures and tests.
#
# The lifespan runs the real hub.subscribe at startup. The transport is the
# only seam: tests hand create_app a MockTransport whose handler records the
# wire requests. The assertions read the captured request bodies, so the tests
# pin the actual form the hub would receive instead of a mocked subscribe call.
# ---------------------------------------------------------------------------

_CHANNEL_1 = "UC_FIXTUREAAAAAAAAAAAAA1"
_CHANNEL_2 = "UC_FIXTUREAAAAAAAAAAAAA2"
_TOPIC_1 = f"https://www.youtube.com/xml/feeds/videos.xml?channel_id={_CHANNEL_1}"
_TOPIC_2 = f"https://www.youtube.com/xml/feeds/videos.xml?channel_id={_CHANNEL_2}"


@pytest.fixture
def channels_file(tmp_path: Path) -> Path:
    """Write a channels.json with two valid ids and return its path."""
    payload = {"channel_ids": [_CHANNEL_1, _CHANNEL_2]}
    path = tmp_path / "channels.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def settings_with_channels(channels_file: Path) -> Settings:
    """Settings with a callback URL, two channels, and an HMAC secret."""
    kwargs: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": channels_file,
        "hub_secret": "topsecret",
    }
    return Settings(**kwargs)


@pytest.fixture
def settings_without_callback(channels_file: Path) -> Settings:
    """Settings with channels but no callback URL."""
    kwargs: dict[str, Any] = {
        "channel_ids_file": channels_file,
        "hub_secret": None,
    }
    return Settings(**kwargs)


@pytest.fixture
def empty_channels_file(tmp_path: Path) -> Path:
    """An empty channels.json (no channel_ids key with content)."""
    path = tmp_path / "channels.json"
    path.write_text(json.dumps({"channel_ids": []}))
    return path


@pytest.fixture
def settings_empty_channels(empty_channels_file: Path) -> Settings:
    """Settings with a callback URL but an empty channel list."""
    kwargs: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": empty_channels_file,
    }
    return Settings(**kwargs)


@pytest.fixture
def captured_hub_requests() -> list[httpx2.Request]:
    """Mutable list the MockTransport handler appends to."""
    return []


def _make_capturing_transport(
    captured: list[httpx2.Request], responses: list[httpx2.Response] | None = None
) -> httpx2.MockTransport:
    """Build a MockTransport. ``responses[0]`` is returned for the first request, then 202s."""
    index = {"i": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        if responses is not None and index["i"] < len(responses):
            resp = responses[index["i"]]
            index["i"] += 1
            return resp
        return httpx2.Response(202)

    return httpx2.MockTransport(handler)


async def _never_sleep(_: float) -> None:
    """Sleep stub that raises if invoked; tests expect the loop to skip the sleep entirely."""
    raise AssertionError("renewal loop should have skipped sleeping")


async def _drive_renewal_loop(
    settings: Settings,
    *,
    sleep: Callable[[float], Awaitable[None]],
    captured: list[httpx2.Request],
    responses: list[httpx2.Response] | None = None,
) -> None:
    """Build an app + MockTransport client, run the renewal loop, close the client."""
    app = create_app(settings)
    transport = _make_capturing_transport(captured, responses=responses)
    async with httpx2.AsyncClient(transport=transport) as client:
        await _renewal_loop(app, client, sleep=sleep)


def _make_counting_sleep(
    delays: list[float], *, cancel_after: int = 2
) -> Callable[[float], Awaitable[None]]:
    """Return a sleep stub that records delays and cancels the loop after N calls."""
    calls = {"count": 0}

    async def sleep(delay: float) -> None:
        delays.append(delay)
        calls["count"] += 1
        if calls["count"] >= cancel_after:
            raise asyncio.CancelledError

    return sleep


def test_lifespan_subscribes_to_every_resolved_channel(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """Startup POSTs one subscribe per resolved channel with the right topic."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with TestClient(app):
        pass

    # Assert
    assert len(captured_hub_requests) == 2
    topics = set()
    for request in captured_hub_requests:
        assert request.method == "POST"
        assert request.url.host == "pubsubhubbub.appspot.com"
        assert request.url.path == "/subscribe"
        form = parse_qs(request.content.decode("utf-8"))
        topics.add(form["hub.topic"][0])
    assert topics == {_TOPIC_1, _TOPIC_2}


def test_lifespan_includes_hub_secret_in_body_when_configured(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """A configured hub secret is sent as hub.secret in every subscribe body."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with TestClient(app):
        pass

    # Assert
    assert len(captured_hub_requests) == 2
    for request in captured_hub_requests:
        form = parse_qs(request.content.decode("utf-8"))
        assert form["hub.secret"][0] == "topsecret"


def test_lifespan_skips_subscribe_when_callback_url_is_missing(
    settings_without_callback: Settings,
    captured_hub_requests: list[httpx2.Request],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a callback URL no subscribe is sent and the reason is logged."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_without_callback, transport=transport)

    # Act
    with caplog.at_level(logging.INFO):
        with TestClient(app):
            pass

    # Assert
    assert captured_hub_requests == []
    assert any(
        record.levelno == logging.INFO and "TLDW_CALLBACK_URL" in record.getMessage()
        for record in caplog.records
    )


def test_lifespan_skips_subscribe_when_no_channels_resolve(
    settings_empty_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty channel list sends no subscribe and logs that none resolved."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_empty_channels, transport=transport)

    # Act
    with caplog.at_level(logging.INFO):
        with TestClient(app):
            pass

    # Assert
    assert captured_hub_requests == []
    assert any(
        record.levelno == logging.INFO and "no channel ids" in record.getMessage()
        for record in caplog.records
    )


def test_lifespan_closes_http_client_on_shutdown(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """The lifespan closes the shared HTTP client when the app shuts down."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with TestClient(app) as client:
        pass

    # Assert
    assert app.state.http_client.is_closed is True
    assert client is not None


def test_lifespan_passes_default_hub_url(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """The default hub URL points at the public hub, not a local mock."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with TestClient(app):
        pass

    # Assert
    assert len(captured_hub_requests) == 2
    for request in captured_hub_requests:
        assert request.url.host == "pubsubhubbub.appspot.com"
        assert request.url.host != "localhost"


def test_lifespan_continues_when_hub_returns_non_2xx(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-2xx subscribe logs a warning for that channel and the loop continues."""
    # Arrange
    responses = [httpx2.Response(202), httpx2.Response(400, text="bad topic")]
    transport = _make_capturing_transport(captured_hub_requests, responses)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with caplog.at_level(logging.WARNING):
        with TestClient(app):
            pass

    # Assert
    assert len(captured_hub_requests) == 2
    assert any(
        record.levelno == logging.WARNING and _CHANNEL_2 in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Renewal loop fixtures and tests.
#
# The renewal loop is exercised end to end: the real _renewal_loop calls the
# real hub.subscribe against a MockTransport. The only seam is the sleep
# callable, which lets tests advance the clock without waiting on the real
# multi-day lease. subscribe is never monkeypatched.
# ---------------------------------------------------------------------------


def test_renewal_delay_uses_default_fraction() -> None:
    """renewal_delay returns 80% of the lease by default and honors an override."""
    # Arrange
    five_day_lease = 432000

    # Act
    default_delay = renewal_delay(five_day_lease)
    half_delay = renewal_delay(100, 0.5)
    full_delay = renewal_delay(100, 1.0)

    # Assert
    assert default_delay == 345600.0
    assert half_delay == 50.0
    assert full_delay == 100.0


async def test_renewal_loop_skips_when_callback_url_missing(
    settings_without_callback: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """A missing callback URL makes the renewal loop return without sleeping or posting."""
    # Arrange
    # _never_sleep raises if the loop sleeps when it should have returned early.

    # Act
    await _drive_renewal_loop(
        settings_without_callback,
        sleep=_never_sleep,
        captured=captured_hub_requests,
    )

    # Assert
    assert captured_hub_requests == []


async def test_renewal_loop_skips_when_no_channels_resolve(
    settings_empty_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """An empty channel list makes the renewal loop return without sleeping or posting."""
    # Arrange
    # _never_sleep raises if the loop sleeps when it should have returned early.

    # Act
    await _drive_renewal_loop(
        settings_empty_channels,
        sleep=_never_sleep,
        captured=captured_hub_requests,
    )

    # Assert
    assert captured_hub_requests == []


async def test_renewal_loop_sleeps_then_resubscribes(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """One renewal cycle sleeps once, then re-subscribes every resolved channel."""
    # Arrange
    delays: list[float] = []
    sleep = _make_counting_sleep(delays, cancel_after=2)

    # Act
    with pytest.raises(asyncio.CancelledError):
        await _drive_renewal_loop(
            settings_with_channels,
            sleep=sleep,
            captured=captured_hub_requests,
        )

    # Assert
    assert len(captured_hub_requests) == 2
    assert delays == [renewal_delay(432000), renewal_delay(432000)]
    topics = set()
    for request in captured_hub_requests:
        assert request.method == "POST"
        form = parse_qs(request.content.decode("utf-8"))
        assert form["hub.mode"] == ["subscribe"]
        topics.add(form["hub.topic"][0])
    assert topics == {_TOPIC_1, _TOPIC_2}


async def test_renewal_loop_logs_warning_on_non_2xx_renewal(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-2xx renewal logs a warning naming the channel and continues the loop."""
    # Arrange
    # The renewal loop is driven directly, so no startup subscribe consumes a
    # response: both responses are for the two renewal POSTs.
    responses = [httpx2.Response(400, text="bad"), httpx2.Response(400, text="bad")]
    delays: list[float] = []
    sleep = _make_counting_sleep(delays, cancel_after=2)

    # Act
    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await _drive_renewal_loop(
                settings_with_channels,
                sleep=sleep,
                captured=captured_hub_requests,
                responses=responses,
            )

    # Assert
    assert len(captured_hub_requests) == 2
    assert any(
        record.levelno == logging.WARNING
        and "renewal" in record.getMessage().lower()
        and (_CHANNEL_1 in record.getMessage() or _CHANNEL_2 in record.getMessage())
        for record in caplog.records
    )


def test_lifespan_cancels_renewal_task_on_shutdown(
    settings_with_channels: Settings,
    captured_hub_requests: list[httpx2.Request],
) -> None:
    """The lifespan starts a renewal task and cancels it on shutdown."""
    # Arrange
    transport = _make_capturing_transport(captured_hub_requests)
    app = create_app(settings_with_channels, transport=transport)

    # Act
    with TestClient(app):
        # Assert (while the lifespan is running)
        assert hasattr(app.state, "renewal_task")
        assert isinstance(app.state.renewal_task, asyncio.Task)
        assert app.state.renewal_task.done() is False

    # Assert (after shutdown)
    assert app.state.renewal_task.cancelled() is True
