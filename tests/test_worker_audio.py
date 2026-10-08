"""Tests for the worker's audio backend pipeline.

Every I/O seam is injected: a fake download, a fake compress, a fake transcribe,
a fake analyze, fake sends, and a fake sleep. The queue is a real ``QueueStore``
on a ``tmp_path`` database so terminal-state, schedule, and artifact assertions
read real persisted rows. No test touches the network or waits on a real sleep.
``_process_record_audio`` is tested directly for the per-record state machine,
and ``transcript_loop`` is tested for the enqueue delay.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI

from helpers import make_video_entry
from tldw import worker
from tldw.audio import CompressError, DownloadResult, ProbeState
from tldw.config import Settings
from tldw.llm import Takeaways
from tldw.queue import QueueRecord, QueueStore, open_store
from tldw.transcribe import TranscribeError
from tldw.worker import (
    DEFAULT_EMPTY_QUEUE_SLEEP,
    DEFAULT_RECORD_PACING,
    _process_record_audio,
    transcript_loop,
)

_VIDEO_ID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Build Settings with the audio pipeline configured under ``tmp_path``."""
    defaults: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
        "discord_webhook_url": "https://discord.com/api/webhooks/x/y",
        "transcript_dir": tmp_path / "transcripts",
        "audio_dir": tmp_path / "audio",
        "queue_file": tmp_path / "queue.sqlite3",
        "openai_api_key": "sk-test",
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


def _takeaways() -> Takeaways:
    """Build a valid three-item Takeaways model with no bullet timestamps.

    The audio backend has no cues to snap to, so bullets carry no timestamps.
    """
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


def _fetch(store: QueueStore, now: float) -> QueueRecord:
    """Fetch the single pending record, asserting it exists."""
    record = store.next_due(now=now)
    assert record is not None
    return record


def _read_row(tmp_path: Path, video_id: str = _VIDEO_ID) -> sqlite3.Row:
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


def _enqueue(store: QueueStore, now: float, video_id: str = _VIDEO_ID) -> None:
    """Enqueue a single pending record due at ``now``."""
    store.enqueue(make_video_entry(video_id), now=now)


def _enqueue_cached(
    store: QueueStore,
    tmp_path: Path,
    now: float,
    *,
    video_id: str = _VIDEO_ID,
    audio_path: Path | None = None,
    transcript_path: Path | None = None,
) -> None:
    """Enqueue a record and pre-set its cached audio or transcript path."""
    store.enqueue(make_video_entry(video_id), now=now)
    store.mark_attempt(
        video_id,
        now=now,
        next_attempt_at=now,
        audio_path=str(audio_path) if audio_path is not None else None,
        transcript_path=str(transcript_path) if transcript_path is not None else None,
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDownload:
    """Download fake that returns canned results in order.

    The sequence is consumed in call order; exhausting it raises so a test that
    expects a single download cannot silently pass on an extra call.
    """

    def __init__(
        self,
        results: Sequence[DownloadResult],
        *,
        order: list[str] | None = None,
    ) -> None:
        self._results = list(results)
        self.call_count = 0
        self.last_url: str | None = None
        self.last_opts: dict[str, object] | None = None
        self.last_dest_dir: Path | None = None
        self._order = order

    def __call__(
        self,
        url: str,
        *,
        opts: dict[str, object],
        ydl_class: object | None = None,
        dest_dir: Path,
    ) -> DownloadResult:
        self.call_count += 1
        self.last_url = url
        self.last_opts = opts
        self.last_dest_dir = dest_dir
        if self._order is not None:
            self._order.append("download")
        if not self._results:
            raise AssertionError("FakeDownload has no result left")
        return self._results.pop(0)


class FakeCompress:
    """Compress fake that records its arguments and optionally creates ``dst``."""

    def __init__(
        self,
        *,
        creates: bool = False,
        raises: Exception | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.call_count = 0
        self.last_src: Path | None = None
        self.last_dst: Path | None = None
        self.last_bitrate: str | None = None
        self.last_timeout_s: float | None = None
        self.raises = raises
        self._creates = creates
        self._order = order

    def __call__(
        self,
        src: Path,
        dst: Path,
        *,
        bitrate: str = "32k",
        timeout_s: float = 900.0,
    ) -> Path:
        self.call_count += 1
        self.last_src = src
        self.last_dst = dst
        self.last_bitrate = bitrate
        self.last_timeout_s = timeout_s
        if self._order is not None:
            self._order.append("compress")
        if self.raises is not None:
            raise self.raises
        if self._creates:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(b"compressed")
        return Path(dst)


class FakeTranscribe:
    """Transcribe fake that returns canned transcripts in order."""

    def __init__(
        self,
        *,
        transcripts: Sequence[str] | None = None,
        raises: Exception | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.transcripts = (
            list(transcripts) if transcripts is not None else ["hello world"]
        )
        self.call_count = 0
        self.last_path: Path | None = None
        self.raises = raises
        self._order = order

    async def __call__(self, path: Path, **_kwargs: Any) -> str:
        self.call_count += 1
        self.last_path = path
        if self._order is not None:
            self._order.append("transcribe")
        if self.raises is not None:
            raise self.raises
        if not self.transcripts:
            raise AssertionError("FakeTranscribe has no transcript left")
        return self.transcripts.pop(0)


class FakeAnalyze:
    """Analyze fake that returns a canned result or raises."""

    def __init__(
        self,
        *,
        result: Takeaways | None = None,
        raises: BaseException | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.calls = 0
        self.result = result
        self.raises = raises
        self._order = order

    async def __call__(self, text: str, video_id: str, title: str) -> Takeaways:
        self.calls += 1
        if self._order is not None:
            self._order.append("analyze")
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result


class CountingSend:
    """Plain-send fake that counts calls and records the message bodies."""

    def __init__(self, *, order: list[str] | None = None) -> None:
        self.calls = 0
        self.contents: list[str] = []
        self._order = order

    async def __call__(
        self, client: object, url: str, content: str, **kwargs: object
    ) -> None:
        self.calls += 1
        self.contents.append(content)
        if self._order is not None:
            self._order.append("send_plain")


class CountingSendEmbeds:
    """Embed-send fake that counts calls and records the embed lists."""

    def __init__(self, *, order: list[str] | None = None) -> None:
        self.calls = 0
        self.embeds: list[list[dict[str, Any]]] = []
        self._order = order

    async def __call__(
        self,
        client: object,
        url: str,
        embeds: list[dict[str, Any]],
        **kwargs: object,
    ) -> None:
        self.calls += 1
        self.embeds.append(embeds)
        if self._order is not None:
            self._order.append("send_embeds")


# ---------------------------------------------------------------------------
# Sleep stubs and the per-record driver
# ---------------------------------------------------------------------------


async def _noop_sleep(_delay: float) -> None:
    """Sleep stub for per-record tests that never need to wait."""
    return None


async def _never_sleep(_delay: float) -> None:
    """Sleep stub that raises if called; pins a path that must not sleep."""
    raise AssertionError("loop should have returned before sleeping")


def _counting_sleep(
    delays: list[float], *, cancel_after: int
) -> Callable[[float], Awaitable[None]]:
    """Return a sleep stub that records delays and cancels the loop after N calls."""
    calls = {"count": 0}

    async def sleep(delay: float) -> None:
        delays.append(delay)
        calls["count"] += 1
        if calls["count"] >= cancel_after:
            raise asyncio.CancelledError

    return sleep


async def _run_audio(
    store: QueueStore,
    settings: Settings,
    *,
    now: float,
    download: FakeDownload,
    compress: FakeCompress,
    transcribe: FakeTranscribe,
    analyze: FakeAnalyze,
    sender: CountingSend,
    embeds_sender: CountingSendEmbeds,
) -> None:
    """Drive _process_record_audio for the single pending record."""
    record = _fetch(store, now)
    await _process_record_audio(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        download_audio=download,
        compress_audio=compress,
        transcribe=transcribe,
        analyze=analyze,
        send=sender,
        send_embeds=embeds_sender,
        sleep=_noop_sleep,
        now=now,
    )


def _ready_raw(tmp_path: Path, name: str = "raw.webm") -> Path:
    """Write a raw audio file and return its path."""
    path = tmp_path / name
    path.write_bytes(b"raw-audio")
    return path


# ---------------------------------------------------------------------------
# Happy path and caching
# ---------------------------------------------------------------------------


async def test_process_record_audio_happy_path_calls_every_stage_in_order(
    store: QueueStore, tmp_path: Path
) -> None:
    """The happy path runs download, compress, transcribe, analyze, embeds."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    order: list[str] = []
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)],
        order=order,
    )
    compress = FakeCompress(creates=True, order=order)
    transcribe = FakeTranscribe(transcripts=["hello world"], order=order)
    analyze = FakeAnalyze(result=_takeaways(), order=order)
    sender = CountingSend(order=order)
    embeds_sender = CountingSendEmbeds(order=order)

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=transcribe,
        analyze=analyze,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert order == ["download", "compress", "transcribe", "analyze", "send_embeds"]
    assert download.call_count == 1
    assert compress.call_count == 1
    assert transcribe.call_count == 1
    assert analyze.calls == 1
    assert embeds_sender.calls == 1
    assert sender.calls == 0
    row = _read_row(tmp_path)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] == "llm_embeds"


async def test_process_record_audio_skips_download_when_audio_path_cached(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached audio_path skips download and compress but still transcribes."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    cached = tmp_path / "audio" / f"{_VIDEO_ID}.compressed.webm"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"cached")
    _enqueue_cached(store, tmp_path, now, audio_path=cached)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    compress = FakeCompress()
    transcribe = FakeTranscribe(transcripts=["hello world"])

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert download.call_count == 0
    assert compress.call_count == 0
    assert transcribe.call_count == 1
    assert transcribe.last_path == cached
    assert _read_row(tmp_path)["terminal_state"] == "DONE"


async def test_process_record_audio_re_downloads_when_cached_audio_path_missing(
    store: QueueStore, tmp_path: Path
) -> None:
    """A stale audio_path that no longer points at a file must trigger re-download.

    The compressed audio is best-effort-deleted after a successful transcribe,
    but the row keeps the old ``audio_path`` until the next reschedule
    successfully writes a fresh path. If the LLM step failed downstream on
    a previous attempt, the file is gone but the row still references it.
    Without the existence check, the worker would skip download+compress and
    feed transcribe a vanished path on every retry.
    """
    # Arrange: enqueue a row whose audio_path points at a file we never create.
    settings = _make_settings(tmp_path)
    now = 1000.0
    stale = tmp_path / "audio" / f"{_VIDEO_ID}.compressed.webm"
    stale.parent.mkdir(parents=True, exist_ok=True)
    # NB: deliberately do NOT write the file.
    _enqueue_cached(store, tmp_path, now, audio_path=stale)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    compress = FakeCompress(creates=True)
    transcribe = FakeTranscribe(transcripts=["hello world"])

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert: download and compress ran because the cached file was gone, the
    # new compressed path is persisted on the row, and the run finished DONE.
    assert download.call_count == 1
    assert compress.call_count == 1
    assert transcribe.call_count == 1
    fresh = settings.audio_dir / f"{_VIDEO_ID}.compressed.{settings.audio_format}"
    assert compress.last_dst == fresh
    assert _read_row(tmp_path)["audio_path"] == str(fresh)
    assert _read_row(tmp_path)["terminal_state"] == "DONE"


async def test_process_record_audio_skips_download_and_compress_when_transcript_path_cached(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached transcript_path skips download, compress, and transcribe."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    txt = tmp_path / "transcripts" / f"{_VIDEO_ID}.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("cached transcript text", encoding="utf-8")
    _enqueue_cached(store, tmp_path, now, transcript_path=txt)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    compress = FakeCompress()
    transcribe = FakeTranscribe()
    analyze = FakeAnalyze(result=_takeaways())

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=transcribe,
        analyze=analyze,
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert download.call_count == 0
    assert compress.call_count == 0
    assert transcribe.call_count == 0
    assert analyze.calls == 1
    assert _read_row(tmp_path)["terminal_state"] == "DONE"


# ---------------------------------------------------------------------------
# Download outcome classification
# ---------------------------------------------------------------------------


async def test_process_record_audio_classifies_429_as_rate_limited_and_streaks(
    store: QueueStore, tmp_path: Path
) -> None:
    """A RATE_LIMITED download bumps the streak and uses the 300s base."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.RATE_LIMITED, None, "429 from youtube")]
    )
    transcribe = FakeTranscribe()

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(),
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert abs((updated.next_attempt_at - now) - 300.0) <= 1.0
    assert updated.rate_limit_streak == 1
    assert transcribe.call_count == 0


async def test_process_record_audio_unavailable_marks_give_up(
    store: QueueStore, tmp_path: Path
) -> None:
    """An UNAVAILABLE download is permanent and marks the record terminal."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.UNAVAILABLE, None, "video is gone")]
    )

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(),
        transcribe=FakeTranscribe(),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 0, "GIVE_UP_UNAVAILABLE": 1}


async def test_process_record_audio_not_ready_reschedules_with_poll_backoff(
    store: QueueStore, tmp_path: Path
) -> None:
    """A NOT_READY download re-arms with the poll base and resets the streak."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload([DownloadResult(ProbeState.NOT_READY, None, None)])

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(),
        transcribe=FakeTranscribe(),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert abs((updated.next_attempt_at - now) - settings.poll_base_seconds) <= 1.0
    assert updated.rate_limit_streak == 0
    # One download means exactly one attempt.
    assert updated.attempts == 1


# ---------------------------------------------------------------------------
# Compress failure
# ---------------------------------------------------------------------------


async def test_process_record_audio_compress_error_reschedules_with_300s_backoff(
    store: QueueStore, tmp_path: Path
) -> None:
    """A CompressError reschedules with the 300 second base."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    transcribe = FakeTranscribe()

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(raises=CompressError("ffmpeg exploded")),
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert abs((updated.next_attempt_at - now) - 300.0) <= 1.0
    assert transcribe.call_count == 0


async def test_process_record_audio_compress_error_does_not_set_audio_path(
    store: QueueStore, tmp_path: Path
) -> None:
    """A failed compress leaves audio_path unset so the next pass re-downloads."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(raises=CompressError("ffmpeg exploded")),
        transcribe=FakeTranscribe(),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert _read_row(tmp_path)["audio_path"] is None


# ---------------------------------------------------------------------------
# Transcription failure classification
# ---------------------------------------------------------------------------


async def test_process_record_audio_4xx_api_error_marks_give_up_audio(
    store: QueueStore, tmp_path: Path
) -> None:
    """A permanent 4xx transcription error marks GIVE_UP_AUDIO."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    transcribe = FakeTranscribe(raises=TranscribeError("bad audio", status_code=400))

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(creates=True),
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 0, "GIVE_UP_AUDIO": 1}


async def test_process_record_audio_429_api_error_reschedules_with_audio_path_preserved(
    store: QueueStore, tmp_path: Path
) -> None:
    """A 429 transcription error reschedules and keeps the compressed audio."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    transcribe = FakeTranscribe(raises=TranscribeError("slow down", status_code=429))
    compressed = settings.audio_dir / f"{_VIDEO_ID}.compressed.{settings.audio_format}"

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(creates=True),
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert updated.audio_path == str(compressed)
    # The next pass skips download and compress, so the streak-free retry
    # cannot re-download the audio.
    assert updated.next_attempt_at > now


# ---------------------------------------------------------------------------
# Transcript artifacts
# ---------------------------------------------------------------------------


async def test_process_record_audio_empty_transcript_marks_done_with_detail_empty_transcript(
    store: QueueStore, tmp_path: Path
) -> None:
    """An empty transcript sends the plain digest with the empty_transcript marker."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    transcribe = FakeTranscribe(transcripts=["   \n  "])
    analyze = FakeAnalyze(raises=AssertionError("analyze should not run"))
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(creates=True),
        transcribe=transcribe,
        analyze=analyze,
        sender=sender,
        embeds_sender=embeds_sender,
    )

    # Assert
    assert analyze.calls == 0
    assert sender.calls == 1
    assert embeds_sender.calls == 0
    row = _read_row(tmp_path)
    assert row["terminal_state"] == "DONE"
    assert row["detail"] == "empty_transcript"


async def test_process_record_audio_writes_transcript_txt_and_persists_transcript_path(
    store: QueueStore, tmp_path: Path
) -> None:
    """The transcript is written to a .txt and its path is persisted."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    txt = settings.transcript_dir / f"{_VIDEO_ID}.txt"

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(creates=True),
        transcribe=FakeTranscribe(transcripts=["hello world"]),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert txt.read_text(encoding="utf-8") == "hello world"
    assert _read_row(tmp_path)["transcript_path"] == str(txt)


async def test_process_record_audio_deletes_compressed_audio_after_transcription(
    store: QueueStore, tmp_path: Path
) -> None:
    """The compressed audio is deleted once the .txt transcript is written."""
    # Arrange
    settings = _make_settings(tmp_path)
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    compress = FakeCompress(creates=True)
    compressed = settings.audio_dir / f"{_VIDEO_ID}.compressed.{settings.audio_format}"

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=FakeTranscribe(transcripts=["hello world"]),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert compress.call_count == 1
    assert not compressed.exists()


# ---------------------------------------------------------------------------
# Give-up and settings wiring
# ---------------------------------------------------------------------------


async def test_process_record_audio_48h_give_up_short_circuits_before_download(
    store: QueueStore, tmp_path: Path
) -> None:
    """A record past the give-up window is retired without a download."""
    # Arrange
    settings = _make_settings(tmp_path, giveup_seconds=0.0)
    now = 1000.0
    store.enqueue(make_video_entry(), now=now - 1)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=FakeCompress(),
        transcribe=FakeTranscribe(),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert download.call_count == 0
    assert store.counts() == {"pending": 0, "GIVE_UP_NEVER": 1}


async def test_process_record_audio_uses_audio_settings_for_dest_dir_and_format(
    store: QueueStore, tmp_path: Path
) -> None:
    """The download dest dir and compress output come from the audio settings."""
    # Arrange
    custom_dir = tmp_path / "custom-audio"
    settings = _make_settings(
        tmp_path, audio_dir=custom_dir, audio_format="mp3", audio_bitrate="24k"
    )
    now = 1000.0
    _enqueue(store, now)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    compress = FakeCompress()

    # Act
    await _run_audio(
        store,
        settings,
        now=now,
        download=download,
        compress=compress,
        transcribe=FakeTranscribe(transcripts=["hello world"]),
        analyze=FakeAnalyze(result=_takeaways()),
        sender=CountingSend(),
        embeds_sender=CountingSendEmbeds(),
    )

    # Assert
    assert download.last_dest_dir == custom_dir
    assert compress.last_dst == custom_dir / f"{_VIDEO_ID}.compressed.mp3"
    assert compress.last_bitrate == "24k"


# ---------------------------------------------------------------------------
# transcript_loop guards and pacing
# ---------------------------------------------------------------------------


async def test_transcript_loop_requires_openai_api_key(
    store: QueueStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker returns before any work when the OpenAI key is missing."""
    # Arrange
    settings = _make_settings(tmp_path, openai_api_key=None)
    app = _make_app(settings, store)

    async def fake_audio(*args: object, **kwargs: object) -> None:
        raise AssertionError("processor should not run")

    monkeypatch.setattr(worker, "_process_record_audio", fake_audio)

    # Act
    # _never_sleep raises if the loop reaches a sleep, so a clean return proves
    # the guard fired before the loop started.
    await transcript_loop(app, httpx2.AsyncClient(), sleep=_never_sleep)

    # Assert
    assert not (tmp_path / "audio").exists()


async def test_transcript_loop_respects_enqueue_delay(
    store: QueueStore, tmp_path: Path
) -> None:
    """A record enqueued with a delay is not downloaded until the delay elapses."""
    # Arrange
    settings = _make_settings(tmp_path)
    app = _make_app(settings, store)
    now = time.time()
    store.enqueue(make_video_entry("v1"), now=now, delay_seconds=300.0)
    download = FakeDownload(
        [DownloadResult(ProbeState.READY, str(_ready_raw(tmp_path)), None)]
    )
    delays: list[float] = []

    # Act
    with pytest.raises(asyncio.CancelledError):
        await transcript_loop(
            app,
            httpx2.AsyncClient(),
            sleep=_counting_sleep(delays, cancel_after=1),
            download_audio=download,
            compress_audio=FakeCompress(),
            transcribe=FakeTranscribe(),
            analyze=FakeAnalyze(result=_takeaways()),
        )

    # Assert
    assert download.call_count == 0
    assert delays == [DEFAULT_EMPTY_QUEUE_SLEEP]
    assert DEFAULT_RECORD_PACING != DEFAULT_EMPTY_QUEUE_SLEEP
