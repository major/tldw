"""FastAPI application factory for the tldw PubSubHubbub subscriber.

The application has a single callback path that the hub uses in two ways. It
calls the path with GET to run the verification challenge, and it POSTs
new-video notifications to the same path. This module owns the factory, the GET
verification handler, and the POST notify handler. The POST handler optionally
checks the HMAC signature, parses the Atom feed, and prints one line per video.
Keeping the factory small lets each later commit add one concern, such as the
renewal loop, without disturbing what is here.

The lifespan subscribes every resolved channel at startup using a shared HTTP
client. A background renewal loop then re-subscribes the same channels before
the hub lease expires so notifications keep flowing.

Settings are attached to ``app.state.settings`` so handlers can read them from
the request without importing global state. The shared HTTP client is attached
to ``app.state.http_client`` so tests and later commits can reach it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from xml.etree import ElementTree as ET

import httpx2
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, PlainTextResponse

from tldw import _version
from tldw import config as tldw_config
from tldw import hub as tldw_hub
from tldw import queue as tldw_queue
from tldw import worker as tldw_worker
from tldw.config import Settings
from tldw.feed import is_short_url, parse_atom
from tldw.renderer import format_video_line

__all__ = ["create_app", "renewal_delay"]

logger = logging.getLogger(__name__)

# Renew at 80% of the lease so we re-subscribe before expiry.
RENEWAL_FRACTION: float = 0.8

# Query parameter names used by the PubSubHubbub verification handshake. The
# names contain dots, so they are read from the query string map rather than as
# function parameters, which must be valid Python identifiers.
_MODE_PARAM = "hub.mode"
_CHALLENGE_PARAM = "hub.challenge"

# The hub verifies a subscription by asking the callback to echo a challenge.
# Only these two modes count as verification; anything else is not a handshake.
_VERIFY_MODES = frozenset({"subscribe", "unsubscribe"})

# The hub signs deliveries with HMAC-SHA1 and sends the digest in this header,
# prefixed with the algorithm name. Any other prefix is not a signature we can
# verify, so it is treated as if the header were absent.
_SIGNATURE_HEADER = "X-Hub-Signature"
_SIGNATURE_PREFIX = "sha1="


async def _subscribe_resolved_channels(
    app: FastAPI, client: httpx2.AsyncClient
) -> None:
    """Subscribe every resolved channel and log per-channel failures.

    A missing callback URL or an empty channel list is not an error: the app
    still starts so an operator can fix the configuration. Subscriptions run
    serially, which keeps the startup order deterministic and the tests race
    free. An HTTP error is logged at WARNING with the offending channel id, and
    any other exception is logged at EXCEPTION level; in both cases the loop
    continues so one bad channel does not block the rest.
    """
    settings: Settings = app.state.settings
    if not settings.callback_url:
        logger.info("skipping subscribe: TLDW_CALLBACK_URL is not set")
        return
    channel_ids = tldw_config.resolve_channel_ids(settings)
    if not channel_ids:
        logger.info("skipping subscribe: no channel ids resolved")
        return
    for channel_id in channel_ids:
        try:
            await tldw_hub.subscribe(
                client,
                channel_id,
                settings.callback_url,
                secret=settings.hub_secret,
            )
        except httpx2.HTTPStatusError as exc:
            logger.warning(
                "subscribe failed for channel %s with status %s",
                channel_id,
                exc.response.status_code,
            )
        except Exception:
            logger.exception("subscribe raised for channel %s", channel_id)


def renewal_delay(lease_seconds: int, fraction: float = RENEWAL_FRACTION) -> float:
    """Return the renewal sleep duration in seconds.

    The renewal loop sleeps for ``lease_seconds * fraction`` between
    re-subscribes. Using a fraction under 1.0 ensures we re-subscribe before
    the hub expires the lease.
    """
    return lease_seconds * fraction


async def _renewal_loop(
    app: FastAPI,
    client: httpx2.AsyncClient,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Re-subscribe the resolved channels forever, sleeping between cycles.

    Returns early when the callback URL is missing or no channels resolve. On
    each cycle it sleeps for ``renewal_delay(DEFAULT_LEASE_SECONDS)`` seconds,
    then re-subscribes every channel. Per-channel failures (HTTP errors, other
    exceptions) are logged but never break the loop so a single bad channel
    cannot strand the others.
    """
    settings: Settings = app.state.settings
    if not settings.callback_url:
        return
    channel_ids = tldw_config.resolve_channel_ids(settings)
    if not channel_ids:
        return
    delay = renewal_delay(tldw_hub.DEFAULT_LEASE_SECONDS)
    while True:
        await sleep(delay)
        for channel_id in channel_ids:
            try:
                await tldw_hub.subscribe(
                    client,
                    channel_id,
                    settings.callback_url,
                    secret=settings.hub_secret,
                )
            except httpx2.HTTPStatusError as exc:
                logger.warning(
                    "renewal subscribe failed for channel %s with status %s",
                    channel_id,
                    exc.response.status_code,
                )
            except Exception:
                logger.exception(
                    "renewal subscribe raised for channel %s", channel_id
                )


def create_app(
    settings: Settings,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the FastAPI application for the subscriber.

    The settings are stored on ``app.state.settings`` so handlers can read them
    from the request. The lifespan opens a shared HTTP client, subscribes every
    resolved channel at startup, starts the renewal loop, and cancels the loop
    and closes the client on shutdown. Passing a transport lets tests hand in an
    ``httpx2.MockTransport`` so the real ``hub.subscribe`` runs against a
    recorded in-memory wire.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Open the queue, subscribe, run the tasks, and clean up on exit."""
        async with httpx2.AsyncClient(transport=transport) as client:
            app.state.http_client = client
            settings: Settings = app.state.settings
            settings.transcript_dir.mkdir(parents=True, exist_ok=True)
            try:
                app.state.queue = tldw_queue.open_store(settings.queue_file)
            except Exception:
                # A malformed or unreadable queue file must not stop the app
                # from starting; notifications then degrade to print-only.
                logger.exception(
                    "failed to open queue store at %s; continuing without it",
                    settings.queue_file,
                )
                app.state.queue = None
            await _subscribe_resolved_channels(app, client)
            app.state.renewal_task = asyncio.create_task(_renewal_loop(app, client))
            if app.state.queue is None or not settings.discord_webhook_url:
                # No queue or no usable webhook means the worker has nothing to
                # drain or nowhere to send; skip the task entirely. The
                # falsy check also catches an empty string, which is what the
                # shipped manifests set by default.
                app.state.transcript_task = None
            else:
                app.state.transcript_task = asyncio.create_task(
                    tldw_worker.transcript_loop(app, client)
                )
            try:
                yield
            finally:
                # Cancel both background tasks before the client and the queue
                # close so neither task can touch a closed resource.
                for task in (app.state.transcript_task, app.state.renewal_task):
                    if task is None:
                        continue
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if app.state.queue is not None:
                    app.state.queue.close()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings

    @app.get("/version")
    async def version() -> dict[str, str]:
        """Return the build identity as JSON.

        Kubernetes probes and operators use this to confirm which commit and
        build is live without scraping container logs. The values mirror what
        the startup banner logs on stdout, so both surfaces agree.
        """
        return _version.snapshot()

    @app.get("/pubsub/callback")
    async def verify_subscription(request: Request) -> PlainTextResponse:
        """Echo the hub challenge when the request is a real verification.

        The hub sends ``hub.mode`` (subscribe or unsubscribe) and
        ``hub.challenge``. A known mode with a non-empty challenge is answered
        with the challenge as text/plain and status 200, which is how the hub
        learns the callback is live. Anything else returns 404 so a stray or
        malformed request never looks like a successful verification. Other
        handshake fields such as the topic and lease are deliberately ignored.
        """
        mode = request.query_params.get(_MODE_PARAM)
        challenge = request.query_params.get(_CHALLENGE_PARAM)
        if mode in _VERIFY_MODES and challenge:
            return PlainTextResponse(content=challenge, status_code=200)
        return PlainTextResponse(status_code=404)

    @app.post("/pubsub/callback")
    async def notify(request: Request) -> PlainTextResponse:
        """Handle one hub delivery and print a line per new video.

        The raw body is read once so the same bytes feed both the HMAC check
        and the parser. When a secret is configured the ``X-Hub-Signature``
        header must carry a matching ``sha1=`` digest, or the delivery is
        rejected with 403. A missing or malformed feed is logged and answered
        with 200 rather than an error: the hub retries failed deliveries, so
        raising would only cause it to hammer the callback.

        The handler stays async so the request body can be awaited without
        blocking the event loop.
        """
        body = await request.body()

        settings: Settings = app.state.settings

        signatures = request.headers.getlist(_SIGNATURE_HEADER)
        header = signatures[0].strip() if signatures else None
        if header is not None and not header.startswith(_SIGNATURE_PREFIX):
            # An unknown algorithm prefix cannot be verified, so treat the
            # header as if it were missing.
            header = None

        secret = settings.hub_secret
        if secret:
            if header is None:
                logger.warning("hub delivery is missing a valid %s header", _SIGNATURE_HEADER)
                return PlainTextResponse(status_code=403)
            expected = hmac.new(
                secret.encode("utf-8"), body, hashlib.sha1
            ).hexdigest()
            provided = header[len(_SIGNATURE_PREFIX):]
            if not hmac.compare_digest(expected, provided):
                logger.warning("hub delivery signature does not match the body")
                return PlainTextResponse(status_code=403)

        try:
            entries = parse_atom(body)
        except ET.ParseError:
            logger.warning("failed to parse hub delivery body as Atom XML")
            return PlainTextResponse(status_code=200)

        # Shorts are filtered by default so the digest focuses on full-length
        # videos. Set TLDW_INCLUDE_SHORTS=true (or pass include_shorts=True to
        # the Settings constructor) to keep them.
        if not settings.include_shorts:
            kept = [entry for entry in entries if not is_short_url(entry.url)]
            dropped = len(entries) - len(kept)
            if dropped:
                logger.info(
                    "filtered shorts from delivery: dropped=%d kept=%d "
                    "total=%d",
                    dropped,
                    len(kept),
                    len(entries),
                )
            entries = kept

        store = getattr(app.state, "queue", None)
        for entry in entries:
            if store is not None:
                try:
                    # The audio backend debounces the first download so YouTube's
                    # pipeline has time to finish producing the audio track. A 0
                    # here means "go immediately", which is what an operator who
                    # set the delay to 0 wants.
                    inserted = store.enqueue(
                        entry,
                        delay_seconds=settings.audio_download_delay_seconds,
                    )
                except Exception:
                    # Best effort: the hub already got its 200, so a queue
                    # failure must not turn into a 500 and an endless retry.
                    # Treat the entry as shown so operators still see the video.
                    logger.exception("queue enqueue failed for %s", entry.video_id)
                    inserted = True
            else:
                # Without a store there is no way to detect duplicates, so
                # every delivery is treated as new and printed.
                inserted = True
            if inserted:
                print(format_video_line(entry), flush=True)
            else:
                # The hub re-delivered a video already in the queue. Log the
                # body details at INFO so an operator can grep and diff later
                # deliveries without treating this as an error.
                logger.info(
                    "duplicate delivery observed: video_id=%s body_size=%d "
                    "body_sha256=%s body_preview=%r",
                    entry.video_id,
                    len(body),
                    hashlib.sha256(body).hexdigest(),
                    body[:200].decode("utf-8", errors="replace").replace("\n", " "),
                )

        return PlainTextResponse(status_code=200)

    @app.get("/queue")
    async def queue_counts() -> JSONResponse:
        """Return the queue's record counts grouped by state.

        Operators poll this to see backlog depth and how many videos have been
        given up on. When the queue store failed to open, returns 503.
        """
        store = getattr(app.state, "queue", None)
        if store is None:
            return JSONResponse(
                {"error": "queue store unavailable"}, status_code=503
            )
        return JSONResponse(store.counts())

    return app
