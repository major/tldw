"""Tests for the tldw transcript worker.

The loop is exercised with every I/O seam injected: a fake probe, a fake send,
and a fake sleep. The queue is a real ``QueueStore`` on a ``tmp_path`` database
so the terminal-state and schedule assertions read real persisted rows. No test
touches the network or waits on a real sleep. ``_process_record`` is tested
directly for the per-record state machine, and ``transcript_loop`` is tested for
the guard, the pacing, and the serial-processing guarantee.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI

from tldw import worker
from tldw.config import Settings
from tldw.feed import VideoEntry
from tldw.queue import QueueRecord, QueueStore, TerminalState, open_store
from tldw.transcript import ProbeResult, ProbeState
from tldw.worker import (
    DEFAULT_EMPTY_QUEUE_SLEEP,
    DEFAULT_RECORD_PACING,
    _process_record,
    transcript_loop,
)


def _entry(video_id: str = "dQw4w9WgXcQ", **overrides: object) -> VideoEntry:
    """Build a VideoEntry with sensible defaults for worker tests."""
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


def _make_settings(**overrides: Any) -> Settings:
    """Build Settings with the transcript pipeline configured."""
    defaults: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
        "discord_webhook_url": "https://discord.com/api/webhooks/x/y",
        "transcript_dir": Path("/tmp/test-transcripts"),
        "queue_file": Path("/tmp/test-queue.sqlite3"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_app(settings: Settings, store: QueueStore) -> FastAPI:
    """Build a minimal app carrying the two state attributes the loop reads."""
    app = FastAPI()
    app.state.settings = settings
    app.state.queue = store
    return app


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """Open a real QueueStore backed by a temporary database."""
    queue_store = open_store(tmp_path / "queue.sqlite3")
    yield queue_store
    queue_store.close()


def make_probe(
    state: ProbeState = ProbeState.READY,
    transcript_path: str | None = "/tmp/test-transcripts/dQw4w9WgXcQ.en.vtt",
    detail: str | None = None,
) -> Callable[..., ProbeResult]:
    """Build a probe fake that always returns the given result."""

    def _fake(url: str, opts: object, ydl_class: object, dest_dir: Path) -> ProbeResult:
        return ProbeResult(state=state, transcript_path=transcript_path, detail=detail)

    return _fake


def make_probe_raises(exc: Exception | None = None) -> Callable[..., ProbeResult]:
    """Build a probe fake that raises, or asserts if it should not be called."""

    def _fake(url: str, opts: object, ydl_class: object, dest_dir: Path) -> ProbeResult:
        if exc is not None:
            raise exc
        raise AssertionError("probe should not have been called")

    return _fake


class CountingSend:
    """Send fake that counts calls and records the message bodies."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls = 0
        self.contents: list[str] = []
        self.raises = raises

    async def __call__(
        self, client: object, url: str, content: str, **kwargs: object
    ) -> None:
        self.calls += 1
        self.contents.append(content)
        if self.raises is not None:
            raise self.raises


async def _noop_sleep(_delay: float) -> None:
    """Sleep stub for per-record tests that never need to wait."""
    return None


async def _never_sleep(_delay: float) -> None:
    """Sleep stub that raises if called; pins a path that must not sleep."""
    raise AssertionError("loop should have returned before sleeping")


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


def _write_vtt(tmp_path: Path, video_id: str) -> Path:
    """Write a small VTT file for the video and return its path."""
    path = tmp_path / f"{video_id}.en.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello transcript\n",
        encoding="utf-8",
    )
    return path


def _enqueue_with_path(
    store: QueueStore, tmp_path: Path, video_id: str, now: float
) -> Path:
    """Enqueue a record and pre-set its cached transcript path."""
    path = _write_vtt(tmp_path, video_id)
    store.enqueue(_entry(video_id), now=now)
    store.mark_attempt(
        video_id, now=now, next_attempt_at=now, transcript_path=str(path)
    )
    return path


def _fetch(store: QueueStore, now: float) -> QueueRecord:
    """Fetch the single pending record, asserting it exists."""
    record = store.next_due(now=now)
    assert record is not None
    return record


# ---------------------------------------------------------------------------
# Loop guards
# ---------------------------------------------------------------------------


async def test_loop_returns_early_without_webhook_url(
    store: QueueStore, tmp_path: Path
) -> None:
    """No webhook URL means the worker returns before it does any work."""
    # Arrange
    settings = _make_settings(
        discord_webhook_url=None, transcript_dir=tmp_path / "transcripts"
    )
    app = _make_app(settings, store)

    # Act
    # _never_sleep raises if the loop reaches a sleep, so a clean return is the
    # assertion that the guard fired first.
    await transcript_loop(
        app,
        httpx2.AsyncClient(),
        sleep=_never_sleep,
        probe=make_probe_raises(),
        send=CountingSend(),
    )

    # Assert
    assert not (tmp_path / "transcripts").exists()


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------


async def test_ready_record_sends_and_marks_done(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached transcript is sent once and the record is marked DONE."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    _enqueue_with_path(store, tmp_path, "dQw4w9WgXcQ", now)
    record = _fetch(store, now)
    sender = CountingSend()

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(),
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert sender.calls == 1
    assert store.counts() == {"pending": 0, "DONE": 1}


async def test_ready_message_contains_title_and_quoted_lines(
    store: QueueStore, tmp_path: Path
) -> None:
    """The sent message carries the title and blockquoted transcript lines.

    This exercises the probe-READY path: the probe points at a real VTT file,
    the worker caches that path, then reads and sends it.
    """
    # Arrange
    settings = _make_settings()
    now = 1000.0
    vtt = _write_vtt(tmp_path, "dQw4w9WgXcQ")
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)
    assert record.transcript_path is None
    sender = CountingSend()

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe(ProbeState.READY, str(vtt)),
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert sender.calls == 1
    assert "Test dQw4w9WgXcQ" in sender.contents[0]
    assert "> Hello transcript" in sender.contents[0]
    assert store.counts() == {"pending": 0, "DONE": 1}


async def test_not_ready_record_rearms_with_backoff(store: QueueStore) -> None:
    """A NOT_READY probe stays pending and is rescheduled by the base delay."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe(ProbeState.NOT_READY, None),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert abs((updated.next_attempt_at - now) - settings.poll_base_seconds) <= 1.0


async def test_unavailable_marks_give_up_immediately(store: QueueStore) -> None:
    """An UNAVAILABLE probe is permanent and marks the record terminal."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe(ProbeState.UNAVAILABLE, None),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 0, "GIVE_UP_UNAVAILABLE": 1}


async def test_rate_limited_uses_short_base_and_remains_pending(
    store: QueueStore,
) -> None:
    """A first 429 uses the short 300 second base and stays pending."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe(ProbeState.RATE_LIMITED, None),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert abs((updated.next_attempt_at - now) - 300.0) <= 1.0


async def test_rate_limited_exhausts_budget_and_marks_give_up(
    store: QueueStore,
) -> None:
    """The eighth consecutive 429 marks the record GIVE_UP_RATE_LIMITED_DEAD."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    for _ in range(7):
        store.mark_attempt("dQw4w9WgXcQ", now=now, next_attempt_at=now)
    record = _fetch(store, now)
    assert record.attempts == 7

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe(ProbeState.RATE_LIMITED, None),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 0, "GIVE_UP_RATE_LIMITED_DEAD": 1}


async def test_record_past_giveup_marks_give_up_never_without_probing(
    store: QueueStore,
) -> None:
    """A record past the give-up window is retired without a probe."""
    # Arrange
    settings = _make_settings(giveup_seconds=0.0)
    now = 1000.0
    store.enqueue(_entry(), now=now - 1)
    record = _fetch(store, now)

    # Act
    # make_probe_raises has no exception, so it asserts if the probe is reached.
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 0, "GIVE_UP_NEVER": 1}


async def test_transcript_path_present_skips_probe_and_sends_only(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached transcript path skips the probe and goes straight to send."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    _enqueue_with_path(store, tmp_path, "dQw4w9WgXcQ", now)
    record = _fetch(store, now)
    sender = CountingSend()

    # Act
    # make_probe_raises asserts if the probe is reached, pinning the skip.
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(),
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert sender.calls == 1


async def test_send_failure_rearms_without_terminal_state(
    store: QueueStore, tmp_path: Path
) -> None:
    """A Discord failure re-arms the record and never marks it terminal."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    _enqueue_with_path(store, tmp_path, "dQw4w9WgXcQ", now)
    record = _fetch(store, now)
    request = httpx2.Request("POST", "https://discord.example/api/webhooks/x/y")
    response = httpx2.Response(500, request=request)
    sender = CountingSend(
        raises=httpx2.HTTPStatusError("boom", request=request, response=response)
    )

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(),
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert sender.calls == 1
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert updated.next_attempt_at > now


async def test_probe_exception_logged_and_record_rearmed(store: QueueStore) -> None:
    """A probe exception leaves the record pending and rescheduled."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(RuntimeError("boom")),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert updated.next_attempt_at > now


async def test_mark_attempt_runs_before_probe_so_crash_safety_holds(
    store: QueueStore,
) -> None:
    """The attempt is recorded once before the probe, even if the probe fails."""
    # Arrange
    settings = _make_settings()
    now = 1000.0
    store.enqueue(_entry(), now=now)
    record = _fetch(store, now)

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=make_probe_raises(RuntimeError("boom")),
        send=CountingSend(),
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert updated.last_attempt_at == now
    assert updated.attempts == 1


# ---------------------------------------------------------------------------
# Loop pacing and serial processing
# ---------------------------------------------------------------------------


async def test_loop_calls_pacing_sleep_between_records(
    store: QueueStore, tmp_path: Path
) -> None:
    """After each processed record the loop sleeps for the record pacing."""
    # Arrange
    settings = _make_settings(transcript_dir=tmp_path / "transcripts")
    store.enqueue(_entry("a"))
    store.enqueue(_entry("b"))
    app = _make_app(settings, store)
    delays: list[float] = []
    sleep = _make_counting_sleep(delays, cancel_after=3)

    # Act
    with pytest.raises(asyncio.CancelledError):
        await transcript_loop(
            app,
            httpx2.AsyncClient(),
            sleep=sleep,
            probe=make_probe(ProbeState.UNAVAILABLE, None),
            send=CountingSend(),
        )

    # Assert
    assert DEFAULT_RECORD_PACING in delays


async def test_loop_sleeps_long_when_queue_empty(
    store: QueueStore, tmp_path: Path
) -> None:
    """An empty queue sleeps for the long empty-queue interval."""
    # Arrange
    settings = _make_settings(transcript_dir=tmp_path / "transcripts")
    app = _make_app(settings, store)
    delays: list[float] = []
    sleep = _make_counting_sleep(delays, cancel_after=2)

    # Act
    with pytest.raises(asyncio.CancelledError):
        await transcript_loop(
            app,
            httpx2.AsyncClient(),
            sleep=sleep,
            probe=make_probe_raises(),
            send=CountingSend(),
        )

    # Assert
    assert delays[0] == DEFAULT_EMPTY_QUEUE_SLEEP


async def test_due_backlog_probed_serially_with_pacing(
    store: QueueStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A due backlog is probed one at a time with pacing between probes."""
    # Arrange
    pacing = 0.02
    monkeypatch.setattr(worker, "DEFAULT_RECORD_PACING", pacing)
    settings = _make_settings(transcript_dir=tmp_path / "transcripts")
    for video_id in ("a", "b", "c"):
        store.enqueue(_entry(video_id))
    app = _make_app(settings, store)

    state = {"in_flight": 0, "max_in_flight": 0}
    call_times: list[float] = []

    def probe(url: str, opts: object, ydl_class: object, dest_dir: Path) -> ProbeResult:
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        call_times.append(time.monotonic())
        # Hold the in-flight slot briefly; a concurrency bug would overlap here.
        time.sleep(0.005)
        state["in_flight"] -= 1
        return ProbeResult(ProbeState.UNAVAILABLE, None, None)

    recorded: list[float] = []

    async def sleep(delay: float) -> None:
        recorded.append(delay)
        await asyncio.sleep(delay)
        if len(recorded) >= 3:
            raise asyncio.CancelledError

    # Act
    with pytest.raises(asyncio.CancelledError):
        await transcript_loop(
            app,
            httpx2.AsyncClient(),
            sleep=sleep,
            probe=probe,
            send=CountingSend(),
        )

    # Assert
    assert state["max_in_flight"] == 1
    assert len(call_times) == 3
    gaps = [later - earlier for earlier, later in zip(call_times, call_times[1:])]
    assert all(gap >= pacing for gap in gaps)
