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
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from urllib.parse import parse_qs

import httpx2
import pytest
from fastapi.testclient import TestClient

from helpers import make_settings, make_video_entry
from tldw import _version
from tldw.app import _renewal_loop, create_app, renewal_delay
from tldw.config import Settings
from tldw.feed import VideoEntry
from tldw.queue import TerminalState, open_store


def _read_titles(queue_file: Path) -> set[str]:
    """Read every queued title straight from the database file."""
    conn = sqlite3.connect(str(queue_file))
    try:
        return {row[0] for row in conn.execute("SELECT title FROM videos")}
    finally:
        conn.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Build Settings pointing at a channels file that does not exist."""
    return make_settings(tmp_path)


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
# /version endpoint and startup banner.
#
# The endpoint and the banner both read the module-level constants in
# tldw._version, which are captured at import time. Tests monkeypatch those
# constants so the test does not depend on the value of TLDW_GIT_SHA in the
# developer's shell.
# ---------------------------------------------------------------------------


def test_version_returns_build_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint reports the git sha, build time, and start time."""
    # Arrange
    monkeypatch.setattr(_version, "GIT_SHA", "abc1234")
    monkeypatch.setattr(_version, "BUILD_TIME", "2026-10-07T10:35:00+00:00")
    monkeypatch.setattr(_version, "STARTED_AT", "2026-10-07T10:36:00+00:00")

    # Act
    response = client.get("/version")

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "git_sha": "abc1234",
        "build_time": "2026-10-07T10:35:00+00:00",
        "started_at": "2026-10-07T10:36:00+00:00",
    }


def test_log_banner_emits_one_info_line(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The banner logs the build identity in a single INFO line."""
    # Arrange
    monkeypatch.setattr(_version, "GIT_SHA", "deadbeef")
    monkeypatch.setattr(_version, "BUILD_TIME", "2026-10-07T10:35:00+00:00")
    monkeypatch.setattr(_version, "STARTED_AT", "2026-10-07T10:36:00+00:00")
    logger = logging.getLogger("tldw.test_banner")

    # Act
    with caplog.at_level(logging.INFO, logger="tldw.test_banner"):
        _version.log_banner(logger)

    # Assert
    info_records = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(info_records) == 1
    record = info_records[0]
    assert "deadbeef" in record.getMessage()
    assert "2026-10-07T10:35:00+00:00" in record.getMessage()
    assert "2026-10-07T10:36:00+00:00" in record.getMessage()


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
def settings_with_secret(tmp_path: Path) -> Settings:
    """Settings configured with an HMAC secret."""
    return make_settings(tmp_path, hub_secret=_HMAC_SECRET)


@pytest.fixture
def settings_without_secret(tmp_path: Path) -> Settings:
    """Settings with hub_secret=None (the default)."""
    return make_settings(tmp_path)


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
def settings_with_channels(tmp_path: Path, channels_file: Path) -> Settings:
    """Settings with a callback URL, two channels, and an HMAC secret."""
    return make_settings(
        tmp_path, channel_ids_file=channels_file, hub_secret="topsecret"
    )


@pytest.fixture
def settings_without_callback(tmp_path: Path, channels_file: Path) -> Settings:
    """Settings with channels but no callback URL."""
    return make_settings(
        tmp_path,
        channel_ids_file=channels_file,
        hub_secret=None,
        callback_url=None,
    )


@pytest.fixture
def empty_channels_file(tmp_path: Path) -> Path:
    """An empty channels.json (no channel_ids key with content)."""
    path = tmp_path / "channels.json"
    path.write_text(json.dumps({"channel_ids": []}))
    return path


@pytest.fixture
def settings_empty_channels(tmp_path: Path, empty_channels_file: Path) -> Settings:
    """Settings with a callback URL but an empty channel list."""
    return make_settings(tmp_path, channel_ids_file=empty_channels_file)


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


# ---------------------------------------------------------------------------
# Transcript pipeline wiring: queue store, worker task, notify enqueue, /queue.
#
# These tests use a real lifespan through TestClient so the startup and
# shutdown ordering is exercised end to end. The queue and transcript paths
# live under tmp_path so nothing is written into the repository.
# ---------------------------------------------------------------------------


def test_lifespan_opens_queue_store(tmp_path: Path) -> None:
    """The lifespan opens the queue store and the database file exists."""
    # Arrange
    settings = make_settings(tmp_path)
    app = create_app(settings)

    # Act
    with TestClient(app):
        # Assert (while running)
        assert app.state.queue is not None

    # Assert
    assert settings.queue_file.exists()


def test_lifespan_creates_transcript_dir(tmp_path: Path) -> None:
    """The lifespan creates the transcript directory at startup."""
    # Arrange
    transcript_dir = tmp_path / "tx"
    settings = make_settings(tmp_path, transcript_dir=transcript_dir)
    app = create_app(settings)

    # Act
    with TestClient(app):
        # Assert (while running)
        assert transcript_dir.is_dir()

    # Assert
    assert transcript_dir.is_dir()


def test_lifespan_starts_transcript_task(tmp_path: Path) -> None:
    """A configured webhook and queue start the transcript worker task."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        audio_dir=tmp_path / "audio",
        openai_api_key="sk-test",
    )
    app = create_app(settings)

    # Act
    with TestClient(app):
        # Assert (while the lifespan is running)
        assert isinstance(app.state.transcript_task, asyncio.Task)
        assert app.state.transcript_task.done() is False


def test_lifespan_skips_transcript_task_without_webhook(tmp_path: Path) -> None:
    """Without a webhook URL there is nothing to send, so no worker task."""
    # Arrange
    settings = make_settings(tmp_path, discord_webhook_url=None)
    app = create_app(settings)

    # Act
    with TestClient(app):
        # Assert
        assert app.state.transcript_task is None


def test_lifespan_cancels_transcript_task_on_shutdown(tmp_path: Path) -> None:
    """The lifespan cancels the transcript worker task on shutdown."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        audio_dir=tmp_path / "audio",
        openai_api_key="sk-test",
    )
    app = create_app(settings)

    # Act
    with TestClient(app):
        assert app.state.transcript_task is not None

    # Assert
    assert app.state.transcript_task.cancelled() is True


def test_lifespan_closes_queue_on_shutdown(tmp_path: Path) -> None:
    """The lifespan closes the queue store on shutdown."""
    # Arrange
    settings = make_settings(tmp_path)
    app = create_app(settings)

    # Act
    with TestClient(app):
        assert app.state.queue is not None

    # Assert
    assert app.state.queue is not None
    with pytest.raises(RuntimeError):
        app.state.queue.enqueue(make_video_entry())


def test_lifespan_survives_queue_store_open_failure(
    tmp_path: Path,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A queue that fails to open leaves the app running and never 500s."""
    # Arrange: a regular file where the queue's parent directory should be
    # makes open_store raise during startup.
    blocking_file = tmp_path / "blocking-file"
    blocking_file.write_text("not a directory", encoding="utf-8")
    settings = make_settings(
        tmp_path, queue_file=blocking_file / "queue.sqlite3"
    )
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with TestClient(app) as client:
        # Assert: startup swallowed the open failure and kept a safe sentinel.
        assert app.state.queue is None
        post = client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )
        queue_response = client.get("/queue")

    # Assert
    assert post.status_code == 200
    assert "Fixture Video One" in capsys.readouterr().out
    assert queue_response.status_code == 503
    assert app.state.queue is None


def test_notify_enqueues_entries_into_queue_store(
    tmp_path: Path,
    multi_entry_payload: bytes,
) -> None:
    """A signed delivery enqueues every entry into the queue store."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with TestClient(app) as client:
        response = client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    assert response.status_code == 200
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 2}
    finally:
        store.close()
    assert _read_titles(settings.queue_file) == {
        "Fixture Video One",
        "Fixture Video Two",
    }


def test_notify_duplicate_delivery_enqueues_once(
    tmp_path: Path,
    multi_entry_payload: bytes,
) -> None:
    """The same delivery twice still inserts one row per video."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with TestClient(app) as client:
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 2}
    finally:
        store.close()


def test_notify_enqueue_failure_still_returns_200(
    tmp_path: Path,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queue failure is logged but the delivery still returns 200."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    def _boom(entry: VideoEntry) -> None:
        raise RuntimeError("queue is down")

    # Act
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.queue, "enqueue", _boom)
        with caplog.at_level(logging.ERROR):
            response = client.post(
                "/pubsub/callback", content=multi_entry_payload, headers=headers
            )

    # Assert
    assert response.status_code == 200
    assert "Fixture Video One" in capsys.readouterr().out
    assert any(
        record.levelno == logging.ERROR and "v_Fixture1AAAAA" in record.getMessage()
        for record in caplog.records
    )


def test_notify_does_not_enqueue_when_store_unavailable(
    tmp_path: Path,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no queue store the delivery still prints and returns 200."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    # Bypass the lifespan: TestClient without the context manager does not run
    # it, so the manual None survives and no store is opened.
    app.state.queue = None
    client = TestClient(app)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    response = client.post(
        "/pubsub/callback", content=multi_entry_payload, headers=headers
    )

    # Assert
    assert response.status_code == 200
    assert "Fixture Video One" in capsys.readouterr().out
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 0}
    finally:
        store.close()


def test_notify_duplicate_delivery_prints_once(
    tmp_path: Path,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A repeated delivery prints each video once instead of on every delivery."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with TestClient(app) as client:
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    out = capsys.readouterr().out
    assert out.count("Fixture Video One") == 1
    assert out.count("Fixture Video Two") == 1
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 2}
    finally:
        store.close()


def test_notify_duplicate_delivery_logs_body_details(
    tmp_path: Path,
    multi_entry_payload: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A duplicate delivery logs an INFO line with the size, hash, and preview."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}
    expected_hash = hashlib.sha256(multi_entry_payload).hexdigest()

    # Act
    with TestClient(app) as client:
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )
        with caplog.at_level(logging.INFO):
            client.post(
                "/pubsub/callback", content=multi_entry_payload, headers=headers
            )

    # Assert
    duplicates = [
        record
        for record in caplog.records
        if "duplicate delivery observed" in record.getMessage()
    ]
    assert duplicates
    first = duplicates[0]
    message = first.getMessage()
    assert first.levelno == logging.INFO
    assert "v_Fixture1AAAAA" in message
    assert expected_hash in message
    assert len(expected_hash) == 64
    assert str(len(multi_entry_payload)) in message


def test_notify_first_delivery_logs_no_duplicate(
    tmp_path: Path,
    multi_entry_payload: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first delivery is new, so no duplicate log line is emitted."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with TestClient(app) as client:
        with caplog.at_level(logging.INFO):
            client.post(
                "/pubsub/callback", content=multi_entry_payload, headers=headers
            )

    # Assert
    assert not any(
        "duplicate delivery observed" in record.getMessage()
        for record in caplog.records
    )


def test_notify_without_store_prints_every_delivery(
    tmp_path: Path,
    multi_entry_payload: bytes,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With no queue store there is no dedup, so every line prints and no log."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    # Bypass the lifespan: TestClient without the context manager does not run
    # it, so the manual None survives and no store is opened.
    app.state.queue = None
    client = TestClient(app)
    headers = {"X-Hub-Signature": _sign(multi_entry_payload)}

    # Act
    with caplog.at_level(logging.INFO):
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )
        client.post(
            "/pubsub/callback", content=multi_entry_payload, headers=headers
        )

    # Assert
    out = capsys.readouterr().out
    assert out.count("Fixture Video One") == 2
    assert not any(
        "duplicate delivery observed" in record.getMessage()
        for record in caplog.records
    )


def test_queue_endpoint_returns_counts_json(tmp_path: Path) -> None:
    """GET /queue reports counts grouped by state."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        audio_dir=tmp_path / "audio",
        openai_api_key="sk-test",
    )
    app = create_app(settings)

    # Act
    with TestClient(app) as client:
        app.state.queue.enqueue(make_video_entry("v_a"))
        app.state.queue.enqueue(make_video_entry("v_b"))
        app.state.queue.mark_terminal("v_a", TerminalState.DONE)
        response = client.get("/queue")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"pending": 1, "DONE": 1}


def test_queue_endpoint_returns_503_when_store_unavailable(tmp_path: Path) -> None:
    """GET /queue returns 503 when the store failed to open."""
    # Arrange
    settings = make_settings(tmp_path)
    app = create_app(settings)
    app.state.queue = None
    client = TestClient(app)

    # Act
    response = client.get("/queue")

    # Assert
    assert response.status_code == 503
    assert response.json() == {"error": "queue store unavailable"}


def test_queue_endpoint_uses_lifespan_store(
    tmp_path: Path,
    multi_entry_payload: bytes,
) -> None:
    """GET /queue reads the store the lifespan opened."""
    # Arrange
    settings = make_settings(tmp_path)
    app = create_app(settings)

    # Act
    with TestClient(app) as client:
        post = client.post("/pubsub/callback", content=multi_entry_payload)
        response = client.get("/queue")

    # Assert
    assert post.status_code == 200
    assert response.status_code == 200
    assert response.json() == {"pending": 2}


# ---------------------------------------------------------------------------
# Shorts filter: notify handler drops /shorts/ URLs unless opted in.
# ---------------------------------------------------------------------------


_SHORT_VIDEO_ID = "v_Short00000000001"
_LONG_VIDEO_ID = "v_LongVid000000001"


def _mixed_shorts_payload() -> bytes:
    """Return a tiny Atom feed with one Short and one full-length video."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:yt="http://www.youtube.com/xml/schemas/2015">'
        "<entry>"
        f"<yt:videoId>{_SHORT_VIDEO_ID}</yt:videoId>"
        "<title>One Minute Short</title>"
        f'<link rel="alternate" href="https://www.youtube.com/shorts/{_SHORT_VIDEO_ID}"/>'
        "<author><name>Fixture Channel</name></author>"
        "</entry>"
        "<entry>"
        f"<yt:videoId>{_LONG_VIDEO_ID}</yt:videoId>"
        "<title>Long Form Video</title>"
        f'<link rel="alternate" href="https://www.youtube.com/watch?v={_LONG_VIDEO_ID}"/>'
        "<author><name>Fixture Channel</name></author>"
        "</entry>"
        "</feed>"
    ).encode("utf-8")


def test_notify_drops_shorts_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without TLDW_INCLUDE_SHORTS the shorts URL is neither printed nor enqueued."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    payload = _mixed_shorts_payload()
    headers = {"X-Hub-Signature": _sign(payload)}

    # Act
    with TestClient(app) as client:
        response = client.post(
            "/pubsub/callback", content=payload, headers=headers
        )

    # Assert
    assert response.status_code == 200
    out = capsys.readouterr().out
    assert "One Minute Short" not in out
    assert "Long Form Video" in out
    store = open_store(settings.queue_file)
    try:
        counts = store.counts()
        assert counts == {"pending": 1}
    finally:
        store.close()
    assert _read_titles(settings.queue_file) == {"Long Form Video"}


def test_notify_keeps_shorts_when_opted_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With TLDW_INCLUDE_SHORTS both shorts and full-length videos are processed."""
    # Arrange
    settings = make_settings(
        tmp_path, hub_secret=_HMAC_SECRET, include_shorts=True
    )
    app = create_app(settings)
    payload = _mixed_shorts_payload()
    headers = {"X-Hub-Signature": _sign(payload)}

    # Act
    with TestClient(app) as client:
        response = client.post(
            "/pubsub/callback", content=payload, headers=headers
        )

    # Assert
    assert response.status_code == 200
    out = capsys.readouterr().out
    assert "One Minute Short" in out
    assert "Long Form Video" in out
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 2}
    finally:
        store.close()


def test_notify_real_payload_drops_shorts_by_default(
    tmp_path: Path,
    real_atom_payload: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The captured feed has 4 shorts and 11 long videos; shorts are filtered by default."""
    # Arrange
    settings = make_settings(tmp_path, hub_secret=_HMAC_SECRET)
    app = create_app(settings)
    headers = {"X-Hub-Signature": _sign(real_atom_payload)}

    # Act
    with TestClient(app) as client:
        response = client.post(
            "/pubsub/callback", content=real_atom_payload, headers=headers
        )

    # Assert
    assert response.status_code == 200
    out = capsys.readouterr().out
    assert "/shorts/" not in out
    store = open_store(settings.queue_file)
    try:
        assert store.counts() == {"pending": 11}
    finally:
        store.close()
