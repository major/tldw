"""Shared factory helpers for the tldw test suite.

These are plain functions (not fixtures) so each test file imports only what
it needs. They live in a top-level module rather than ``conftest.py`` so the
boundary between auto-injected fixtures and explicit helpers stays clear.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from tldw.config import Settings
from tldw.feed import VideoEntry
from tldw.llm import Takeaways


def make_video_entry(
    video_id: str = "dQw4w9WgXcQ", **overrides: object
) -> VideoEntry:
    """Build a VideoEntry with the standard test channel and title pattern.

    The defaults match the values the queue and worker tests have used
    historically, so swapping a local ``_entry`` / ``_video_entry`` helper for
    this call site changes nothing observable.
    """
    defaults: dict[str, object] = {
        "video_id": video_id,
        "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
        "title": f"Test {video_id}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel_name": "Test Channel",
        "published": None,
        "updated": None,
    }
    defaults.update(overrides)
    return VideoEntry(**defaults)  # type: ignore[arg-type]


def make_settings(tmp_path: Path | None = None, **overrides: Any) -> Settings:
    """Build Settings with safe defaults for the tldw test suite.

    When ``tmp_path`` is provided, ``queue_file`` and ``transcript_dir`` are
    placed under it so tests do not touch real on-disk state. When it is
    omitted, the caller is expected to override any path fields they need
    (used by the llm tests that never touch disk).
    """
    defaults: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
    }
    if tmp_path is not None:
        defaults["transcript_dir"] = tmp_path / "transcripts"
        defaults["queue_file"] = tmp_path / "queue.sqlite3"
    defaults.update(overrides)
    return Settings(**defaults)


def make_capturing_transport(
    captured: list[httpx2.Request],
    responses: list[httpx2.Response] | None = None,
    *,
    default_status: int = 200,
    default_json: dict[str, object] | None = None,
    default_text: str | None = None,
) -> httpx2.MockTransport:
    """Build an ``httpx2.MockTransport`` that records each request.

    ``responses`` is replayed in order. When the list is exhausted, every
    additional request receives a default response of ``default_status`` with
    ``default_json`` as the JSON body or ``default_text`` as the text body
    (both ``None`` produces a bare status response with no body).

    This consolidates three near-identical helpers that previously lived in
    ``test_app.py``, ``test_discord.py``, and ``test_send_embeds.py`` (which
    defaulted to 202, 200, and 204 respectively) and the inline transport
    handlers in ``test_hub.py``.
    """
    queue = list(responses or [])
    index = {"i": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        if index["i"] < len(queue):
            response = queue[index["i"]]
            index["i"] += 1
            return response
        if default_json is not None:
            return httpx2.Response(default_status, json=default_json)
        if default_text is not None:
            return httpx2.Response(default_status, text=default_text)
        return httpx2.Response(default_status)

    return httpx2.MockTransport(handler)


def recording_sleep() -> tuple[
    list[float], Callable[[float], Awaitable[None]]
]:
    """Return ``(delays, sleep)`` where ``sleep`` records instead of waiting.

    The returned ``delays`` list is mutated in place as ``sleep`` is called,
    so tests can assert the recorded delay sequence after the await.
    """
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    return delays, sleep


async def noop_sleep(_delay: float) -> None:
    """Sleep stub that returns immediately. Used in worker pipeline tests."""


async def never_sleep(_delay: float) -> None:
    """Sleep stub that raises if called.

    Tests that inject this prove the loop returned before reaching any
    sleep, so an unhandled call here indicates a regression.
    """
    raise AssertionError("sleep should not be called")


def counting_sleep(
    delays: list[float], cancel_after: int
) -> Callable[[float], Awaitable[None]]:
    """Sleep stub that records delays and raises ``CancelledError`` after N.

    Used to drive the worker loop's cancellation path with deterministic
    pacing.
    """
    counter = {"n": 0}

    async def sleep(delay: float) -> None:
        delays.append(delay)
        counter["n"] += 1
        if counter["n"] >= cancel_after:
            raise asyncio.CancelledError

    return sleep


def make_takeaways(
    item_count: int = 3,
    bullet_count: int = 1,
    *,
    title_template: str = "Takeaway {i}",
    summary_template: str | None = "Summary {i}",
    bullet_text: str | None = "Bullet {i}-{n}",
    bullet_times: list[list[int | None]] | None = None,
) -> Takeaways:
    """Build a valid ``Takeaways`` model for tests.

    The defaults reproduce the three-item / one-bullet shape that the
    worker pipeline tests use. To build the variants the LLM and embed
    tests exercise, pass ``bullet_times=[[t1, t2, ...], ...]``
    (per-item, per-bullet) for custom timestamps, or pass a custom
    ``bullet_text`` template.

    ``TakeawayBullet.text`` requires a non-empty string, so this factory
    never emits ``text=None`` bullets. Callers that want the worker /
    audio bullet text pass ``bullet_text="Bullet {i}"`` explicitly.
    ``bullet_text=None`` falls back to the default ``Bullet {i}-{n}``
    template.

    For full control over the payload, callers can
    ``Takeaways.model_validate`` directly -- this factory targets the
    common case and the variants above.
    """
    resolved_bullet_text = (
        "Bullet {i}-{n}" if bullet_text is None else bullet_text
    )
    items: list[dict[str, object]] = []
    for i in range(item_count):
        bullets: list[dict[str, object]] = []
        for n in range(bullet_count):
            bullet: dict[str, object] = {
                "text": resolved_bullet_text.format(i=i, n=n)
            }
            if bullet_times is not None and i < len(bullet_times):
                per_item = bullet_times[i]
                if n < len(per_item):
                    bullet["timestamp_seconds"] = per_item[n]
            bullets.append(bullet)
        item: dict[str, object] = {
            "title": title_template.format(i=i),
            "bullets": bullets,
        }
        if summary_template is not None:
            item["summary"] = summary_template.format(i=i)
        items.append(item)
    return Takeaways.model_validate({"items": items})
