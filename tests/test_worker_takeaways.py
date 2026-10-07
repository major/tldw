"""Tests for the worker's LLM takeaway path.

Every I/O seam is injected: a fake probe, a fake plain ``send``, a fake
``send_embeds``, and a fake ``analyze``. The queue is a real ``QueueStore`` on a
``tmp_path`` database so the terminal-state and detail assertions read real
persisted rows. No test touches the network or waits on a real sleep.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from tldw.config import Settings
from tldw.feed import VideoEntry
from tldw.llm import OpencodeGoAnalyzer, Takeaways
from tldw.queue import QueueRecord, QueueStore, TerminalState, open_store
from tldw.transcript import ProbeResult, ProbeState
from tldw.worker import _process_record

_VIDEO_ID = "dQw4w9WgXcQ"


def _entry(video_id: str = _VIDEO_ID, **overrides: object) -> VideoEntry:
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


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Build Settings with the transcript pipeline and an API key configured."""
    defaults: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
        "discord_webhook_url": "https://discord.com/api/webhooks/x/y",
        "transcript_dir": tmp_path / "transcripts",
        "queue_file": tmp_path / "queue.sqlite3",
        "opencode_api_key": "test-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """Open a real QueueStore backed by a temporary database."""
    queue_store = open_store(tmp_path / "queue.sqlite3")
    yield queue_store
    queue_store.close()


def _write_vtt(tmp_path: Path, video_id: str) -> Path:
    """Write a small VTT file with one cue and return its path."""
    path = tmp_path / f"{video_id}.en.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello transcript\n",
        encoding="utf-8",
    )
    return path


def _write_vtt_without_cues(tmp_path: Path, video_id: str) -> Path:
    """Write a VTT file that parses to zero cues (a header plus a NOTE)."""
    path = tmp_path / f"{video_id}.en.vtt"
    path.write_text(
        "WEBVTT\n\nNOTE this comment carries no transcript text\n",
        encoding="utf-8",
    )
    return path


def _enqueue_with_path(
    store: QueueStore, path: Path, video_id: str, now: float
) -> None:
    """Enqueue a record and pre-set its cached transcript path."""
    store.enqueue(_entry(video_id), now=now)
    store.mark_attempt(
        video_id, now=now, next_attempt_at=now, transcript_path=str(path)
    )


def _fetch(store: QueueStore, now: float) -> QueueRecord:
    """Fetch the single pending record, asserting it exists."""
    record = store.next_due(now=now)
    assert record is not None
    return record


def _read_row(tmp_path: Path, video_id: str) -> sqlite3.Row:
    """Read a videos row straight from the file, including terminal rows."""
    conn = sqlite3.connect(str(tmp_path / "queue.sqlite3"))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def _takeaways() -> Takeaways:
    """Build a valid three-item Takeaways model for the fake analyzer."""
    return Takeaways.model_validate(
        {
            "items": [
                {
                    "title": f"Takeaway {i}",
                    "summary": f"Summary {i}",
                    "bullets": [
                        {"text": f"Bullet {i}", "timestamp_seconds": i * 30}
                    ],
                }
                for i in range(3)
            ]
        }
    )


def _probe_should_not_run(*args: object, **kwargs: object) -> ProbeResult:
    """Probe fake that fails the test if it is ever called."""
    raise AssertionError("probe should not have been called")


class CountingSend:
    """Plain-send fake that counts calls and records the message bodies."""

    def __init__(self) -> None:
        self.calls = 0
        self.contents: list[str] = []

    async def __call__(
        self, client: object, url: str, content: str, **kwargs: object
    ) -> None:
        self.calls += 1
        self.contents.append(content)


class CountingSendEmbeds:
    """Embed-send fake that counts calls and records the embed lists."""

    def __init__(self) -> None:
        self.calls = 0
        self.embeds: list[list[dict[str, Any]]] = []

    async def __call__(
        self, client: object, url: str, embeds: list[dict[str, Any]], **kwargs: object
    ) -> None:
        self.calls += 1
        self.embeds.append(embeds)


class FakeAnalyze:
    """Analyze fake that returns a canned result or raises."""

    def __init__(
        self,
        *,
        result: Takeaways | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.calls = 0
        self.result = result
        self.raises = raises

    async def __call__(self, rendered: str, video_id: str, title: str) -> Takeaways:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result


async def _noop_sleep(_delay: float) -> None:
    """Sleep stub for per-record tests that never need to wait."""
    return None


async def _run(
    store: QueueStore,
    tmp_path: Path,
    *,
    settings: Settings,
    now: float,
    analyze: FakeAnalyze,
    sender: CountingSend,
    embeds_sender: CountingSendEmbeds,
) -> None:
    """Drive _process_record for one cached-transcript record."""
    record = _fetch(store, now)
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=_probe_should_not_run,
        send=sender,
        send_embeds=embeds_sender,
        analyze=analyze,
        sleep=_noop_sleep,
        now=now,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


async def test_llm_success_sends_embeds_and_marks_done(
    store: QueueStore, tmp_path: Path
) -> None:
    """A successful analysis posts three embeds and marks DONE with llm_embeds."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue_with_path(store, _write_vtt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(result=_takeaways())

    # Act
    await _run(
        store,
        tmp_path,
        settings=settings,
        now=now,
        analyze=analyzer,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert analyzer.calls == 1
    assert embeds_sender.calls == 1
    assert len(embeds_sender.embeds[0]) == 3
    assert sender.calls == 0
    row = _read_row(tmp_path, _VIDEO_ID)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] == "llm_embeds"


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------


async def test_no_api_key_skips_analysis_and_sends_plain(
    store: QueueStore, tmp_path: Path
) -> None:
    """Without an API key the analyzer never runs and the plain digest is sent."""
    # Arrange
    settings = _make_settings(tmp_path, opencode_api_key=None)
    now = 1000.0
    _enqueue_with_path(store, _write_vtt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=AssertionError("analyze should not have run"))

    # Act
    await _run(
        store,
        tmp_path,
        settings=settings,
        now=now,
        analyze=analyzer,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert analyzer.calls == 0
    assert sender.calls == 1
    assert embeds_sender.calls == 0
    row = _read_row(tmp_path, _VIDEO_ID)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] is None


async def test_empty_transcript_skips_analysis_and_sends_plain(
    store: QueueStore, tmp_path: Path
) -> None:
    """A transcript with no cues skips the LLM and sends the plain digest."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    path = _write_vtt_without_cues(tmp_path, _VIDEO_ID)
    _enqueue_with_path(store, path, _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=AssertionError("analyze should not have run"))

    # Act
    await _run(
        store,
        tmp_path,
        settings=settings,
        now=now,
        analyze=analyzer,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert analyzer.calls == 0
    assert sender.calls == 1
    assert embeds_sender.calls == 0
    row = _read_row(tmp_path, _VIDEO_ID)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] is None


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


def _validation_error() -> ValidationError:
    """Produce a real ValidationError from a deliberately invalid model."""
    try:
        Takeaways.model_validate({"items": []})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected Takeaways validation to fail")


def _api_error() -> anthropic.APIError:
    """Build a real anthropic APIError without touching the network."""
    request = httpx2.Request("POST", "https://opencode.ai/zen/go")
    return anthropic.APIError("boom", request=request, body=None)


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        _validation_error(),
        _api_error(),
    ],
    ids=["timeout", "validation", "api_error"],
)
async def test_llm_failure_falls_back_to_plain_digest(
    store: QueueStore, tmp_path: Path, exc: BaseException
) -> None:
    """A timeout, validation error, or API error falls back without bumping attempts."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue_with_path(store, _write_vtt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=exc)

    # Act
    await _run(
        store,
        tmp_path,
        settings=settings,
        now=now,
        analyze=analyzer,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert analyzer.calls == 1
    assert sender.calls == 1
    assert embeds_sender.calls == 0
    row = _read_row(tmp_path, _VIDEO_ID)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] == "llm_fallback"
    # An LLM failure is not a probe, so attempts stays at the cached-path value.
    assert row["attempts"] == 1


# ---------------------------------------------------------------------------
# Live smoke test (opt-in, hits the real gateway)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("TLDW_OPENCODE_API_KEY"),
    reason="TLDW_OPENCODE_API_KEY not set",
)
async def test_live_opencode_analyzer_returns_takeaways() -> None:
    """One real gateway request returns three validated takeaways."""
    settings = Settings(opencode_api_key=os.environ["TLDW_OPENCODE_API_KEY"])
    analyzer = OpencodeGoAnalyzer(settings)
    transcript = (
        "[0:00] Welcome to the show.\n"
        "[0:10] Today we talk about testing.\n"
        "[0:20] Write the test first, then the code."
    )
    result = await analyzer.analyze(
        transcript, video_id="live-smoke", title="Live smoke test"
    )
    assert len(result.items) == 3
