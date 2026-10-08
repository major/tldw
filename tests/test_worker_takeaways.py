"""Tests for the worker's LLM takeaway path.

Every I/O seam is injected: a fake download, compress, and transcribe (none of
which should run for a cached transcript), a fake plain ``send``, a fake
``send_embeds``, and a fake ``analyze``. The queue is a real ``QueueStore`` on a
``tmp_path`` database so the terminal-state and detail assertions read real
persisted rows. No test touches the network or waits on a real sleep.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import openai
import httpx2
import pydantic_ai.exceptions
import pytest
from pydantic import ValidationError

from helpers import make_settings, make_video_entry, noop_sleep
from tldw.audio import DownloadResult
from tldw.config import Settings
from tldw.llm import OpenAIAnalyzer, Takeaways
from tldw.queue import QueueRecord, QueueStore
from tldw.worker import _process_record_audio

_VIDEO_ID = "dQw4w9WgXcQ"


def _write_txt(tmp_path: Path, video_id: str) -> Path:
    """Write a small ``.txt`` transcript file and return its path."""
    path = tmp_path / f"{video_id}.txt"
    path.write_text("Hello transcript\n", encoding="utf-8")
    return path


def _write_empty_txt(tmp_path: Path, video_id: str) -> Path:
    """Write an empty ``.txt`` transcript so the LLM stage sees no text."""
    path = tmp_path / f"{video_id}.txt"
    path.write_text("", encoding="utf-8")
    return path


def _enqueue_with_path(
    queue_store: QueueStore, path: Path, video_id: str, now: float
) -> None:
    """Enqueue a record and pre-set its cached transcript path."""
    queue_store.enqueue(make_video_entry(video_id), now=now)
    queue_store.mark_attempt(
        video_id, now=now, next_attempt_at=now, transcript_path=str(path)
    )


def _fetch(queue_store: QueueStore, now: float) -> QueueRecord:
    """Fetch the single pending record, asserting it exists."""
    record = queue_store.next_due(now=now)
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


def _download_should_not_run(*args: object, **kwargs: object) -> DownloadResult:
    """Download fake that fails the test if it is ever called."""
    raise AssertionError("download should not have been called")


def _compress_should_not_run(*args: object, **kwargs: object) -> Path:
    """Compress fake that fails the test if it is ever called."""
    raise AssertionError("compress should not have been called")


async def _transcribe_should_not_run(*args: object, **kwargs: object) -> str:
    """Transcribe fake that fails the test if it is ever called."""
    raise AssertionError("transcribe should not have been called")


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


async def _run(
    queue_store: QueueStore,
    tmp_path: Path,
    *,
    settings: Settings,
    now: float,
    analyze: FakeAnalyze,
    sender: CountingSend,
    embeds_sender: CountingSendEmbeds,
) -> None:
    """Drive _process_record_audio for one cached-transcript record."""
    record = _fetch(queue_store, now)
    await _process_record_audio(
        record,
        settings,
        queue_store,
        httpx2.AsyncClient(),
        download_audio=_download_should_not_run,
        compress_audio=_compress_should_not_run,
        transcribe=_transcribe_should_not_run,
        analyze=analyze,
        send=sender,
        send_embeds=embeds_sender,
        sleep=noop_sleep,
        now=now,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


async def test_llm_success_sends_embeds_and_marks_done(
    queue_store: QueueStore, tmp_path: Path
) -> None:
    """A successful analysis posts three embeds and marks DONE with llm_embeds."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        openai_api_key="test-key",
    )
    now = 1000.0
    _enqueue_with_path(queue_store, _write_txt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(result=_takeaways())

    # Act
    await _run(
        queue_store,
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
    queue_store: QueueStore, tmp_path: Path
) -> None:
    """Without an API key the analyzer never runs and the plain digest is sent."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        openai_api_key=None,
    )
    now = 1000.0
    _enqueue_with_path(queue_store, _write_txt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=AssertionError("analyze should not have run"))

    # Act
    await _run(
        queue_store,
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
    queue_store: QueueStore, tmp_path: Path
) -> None:
    """An empty transcript skips the LLM and sends the plain digest."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        openai_api_key="test-key",
    )
    now = 1000.0
    path = _write_empty_txt(tmp_path, _VIDEO_ID)
    _enqueue_with_path(queue_store, path, _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=AssertionError("analyze should not have run"))

    # Act
    await _run(
        queue_store,
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
    assert row["detail"] == "empty_transcript"


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


def _api_error() -> openai.APIError:
    """Build a real openai APIError without touching the network."""
    request = httpx2.Request("POST", "https://api.openai.com/v1")
    return openai.APIError("boom", request=request, body=None)


def _unexpected_model_behavior() -> pydantic_ai.exceptions.UnexpectedModelBehavior:
    """Build a real UnexpectedModelBehavior without touching the network."""
    return pydantic_ai.exceptions.UnexpectedModelBehavior(
        "Model token limit (2048) exceeded before any response was generated."
    )


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        _validation_error(),
        _api_error(),
        _unexpected_model_behavior(),
    ],
    ids=["timeout", "validation", "api_error", "unexpected_model"],
)
async def test_llm_failure_falls_back_to_plain_digest(
    queue_store: QueueStore, tmp_path: Path, exc: BaseException
) -> None:
    """A timeout, validation, API, or unexpected-model error falls back without bumping attempts."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        openai_api_key="test-key",
    )
    now = 1000.0
    _enqueue_with_path(queue_store, _write_txt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(raises=exc)

    # Act
    await _run(
        queue_store,
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
    not os.environ.get("TLDW_OPENAI_API_KEY"),
    reason="TLDW_OPENAI_API_KEY not set",
)
async def test_live_openai_analyzer_returns_takeaways() -> None:
    """One real OpenAI request returns three validated takeaways."""
    settings = Settings(openai_api_key=os.environ["TLDW_OPENAI_API_KEY"])
    analyzer = OpenAIAnalyzer(settings)
    transcript = (
        "[0:00] Welcome to the show.\n"
        "[0:10] Today we talk about testing.\n"
        "[0:20] Write the test first, then the code."
    )
    result = await analyzer.analyze(
        transcript, video_id="live-smoke", title="Live smoke test"
    )
    assert len(result.items) == 3


# ---------------------------------------------------------------------------
# Optional timestamps (audio backend)
# ---------------------------------------------------------------------------


def _takeaways_without_timestamps() -> Takeaways:
    """Build a valid three-item Takeaways with no bullet timestamps."""
    return Takeaways.model_validate(
        {
            "items": [
                {
                    "title": f"Takeaway {i}",
                    "summary": f"Summary {i}",
                    "bullets": [{"text": f"Bullet {i}"}],
                }
                for i in range(3)
            ]
        }
    )


async def test_takeaways_without_timestamps_render_plain(
    queue_store: QueueStore, tmp_path: Path
) -> None:
    """Audio-backend takeaways render plain bullet lines, not deep links."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_webhook_url="https://discord.com/api/webhooks/x/y",
        openai_api_key="test-key",
    )
    now = 1000.0
    _enqueue_with_path(queue_store, _write_txt(tmp_path, _VIDEO_ID), _VIDEO_ID, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyzer = FakeAnalyze(result=_takeaways_without_timestamps())

    # Act
    await _run(
        queue_store,
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
    descriptions = [embed["description"] for embed in embeds_sender.embeds[0]]
    assert all("youtu.be" not in description for description in descriptions)
    assert all("- Bullet" in description for description in descriptions)
