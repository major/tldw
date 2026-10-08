"""Roll-back hardening tests for the transcript worker.

The audio backend is the default, but ``TLDW_TRANSCRIPT_BACKEND=vtt`` must stay
operational as a switchable roll-back path. These tests pin the edge cases that
only show up after a backend flip: an audio record whose transcript already
landed as a plain ``.txt`` and is then observed by a vtt-mode worker.

Every test uses a real ``QueueStore`` on a ``tmp_path`` database so the
terminal-state and schedule assertions read real persisted rows. No test touches
the network or waits on a real sleep.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest

from tldw.audio import DownloadResult
from tldw.config import Settings
from tldw.feed import VideoEntry
from tldw.llm import Takeaway, TakeawayBullet, Takeaways
from tldw.queue import QueueRecord, QueueStore, open_store
from tldw.transcript import ProbeResult, build_ydl_opts
from tldw.worker import _process_record, _process_record_audio

_VIDEO_ID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Builders and fixtures
# ---------------------------------------------------------------------------


def _entry(video_id: str = _VIDEO_ID, **overrides: object) -> VideoEntry:
    """Build a VideoEntry with sensible defaults for roll-back tests."""
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


def _make_settings(tmp_path: Path, *, backend: str = "vtt", **overrides: Any) -> Settings:
    """Build Settings with the transcript pipeline configured.

    ``backend`` selects the roll-back path per test: ``vtt`` for the historical
    subtitle path, ``audio`` for the default audio pipeline.
    """
    defaults: dict[str, Any] = {
        "callback_url": "https://cb.example/pubsub/callback",
        "channel_ids_file": Path("/nonexistent.json"),
        "discord_webhook_url": "https://discord.com/api/webhooks/x/y",
        "transcript_dir": tmp_path / "transcripts",
        "audio_dir": tmp_path / "audio",
        "queue_file": tmp_path / "queue.sqlite3",
        "transcript_backend": backend,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """Open a real QueueStore backed by a temporary database."""
    queue_store = open_store(tmp_path / "queue.sqlite3")
    yield queue_store
    queue_store.close()


def _takeaways() -> Takeaways:
    """Build a valid three-item Takeaways model with timestamped bullets."""
    return Takeaways(
        items=[
            Takeaway(
                title=f"T{i}",
                summary="S",
                bullets=[TakeawayBullet(text="b", timestamp_seconds=10)],
            )
            for i in range(3)
        ]
    )


def _fetch(store: QueueStore, now: float) -> QueueRecord:
    """Fetch the single pending record, asserting it exists."""
    record = store.next_due(now=now)
    assert record is not None
    return record


def _read_row(tmp_path: Path, video_id: str = _VIDEO_ID) -> dict[str, Any]:
    """Read a videos row straight from the file, including terminal rows."""
    conn = sqlite3.connect(str(tmp_path / "queue.sqlite3"))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        assert row is not None
        return dict(row)
    finally:
        conn.close()


def _enqueue_cached_txt(
    store: QueueStore,
    settings: Settings,
    video_id: str,
    now: float,
    text: str,
) -> Path:
    """Enqueue a record and pre-seed a plain ``.txt`` cached transcript."""
    store.enqueue(_entry(video_id), now=now)
    settings.transcript_dir.mkdir(parents=True, exist_ok=True)
    path = settings.transcript_dir / f"{video_id}.txt"
    path.write_text(text, encoding="utf-8")
    store.mark_attempt(
        video_id, now=now, next_attempt_at=now, transcript_path=str(path)
    )
    return path


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


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
        self,
        client: object,
        url: str,
        embeds: list[dict[str, Any]],
        **kwargs: object,
    ) -> None:
        self.calls += 1
        self.embeds.append(embeds)


class FakeAnalyze:
    """Analyze fake that returns a canned result and counts its calls."""

    def __init__(self, *, result: Takeaways | None = None) -> None:
        self.calls = 0
        self.result = result

    async def __call__(self, text: str, video_id: str, title: str) -> Takeaways:
        self.calls += 1
        assert self.result is not None
        return self.result


class FakeDownload:
    """Download fake that fails if it is called (cached transcript case)."""

    def __init__(self, results: Sequence[DownloadResult] = ()) -> None:
        self._results = list(results)
        self.call_count = 0

    def __call__(
        self,
        url: str,
        *,
        opts: dict[str, object],
        ydl_class: object | None = None,
        dest_dir: Path,
    ) -> DownloadResult:
        self.call_count += 1
        if not self._results:
            raise AssertionError("download_audio should not have been called")
        return self._results.pop(0)


class FakeCompress:
    """Compress fake that fails if it is called (cached transcript case)."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(
        self,
        src: Path,
        dst: Path,
        *,
        bitrate: str = "32k",
        timeout_s: float = 900.0,
    ) -> Path:
        self.call_count += 1
        raise AssertionError("compress_audio should not have been called")


class FakeTranscribe:
    """Transcribe fake that fails if it is called (cached transcript case)."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self, path: Path) -> str:
        self.call_count += 1
        raise AssertionError("transcribe should not have been called")


async def _noop_sleep(_delay: float) -> None:
    """Sleep stub for per-record tests that never need to wait."""
    return None


def _probe_unused(*_args: object, **_kwargs: object) -> ProbeResult:
    """Probe fake that asserts if reached; a cached path must skip the probe."""
    raise AssertionError("probe should not run when a transcript_path is cached")


# ---------------------------------------------------------------------------
# VTT roll-back: cached .txt rows
# ---------------------------------------------------------------------------


async def test_vtt_process_record_handles_txt_cached_path(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached .txt is delivered via the first_lines fallback and marked DONE.

    This is the flip scenario: the audio backend cached a .txt transcript, then
    the operator set TLDW_TRANSCRIPT_BACKEND=vtt and restarted. The vtt worker
    sees the cached path, skips the probe, and reads the .txt as raw lines.
    """
    # Arrange
    settings = _make_settings(tmp_path, backend="vtt", openai_api_key=None)
    now = 1000.0
    txt = _enqueue_cached_txt(
        store, settings, _VIDEO_ID, now, "line one\nline two\nline three\n"
    )
    record = _fetch(store, now)
    sender = CountingSend()

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=_probe_unused,
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert sender.calls == 1
    message = sender.contents[0]
    assert "**Test dQw4w9WgXcQ** (Test Channel)" in message
    assert record.url in message
    assert "> line one" in message
    assert "> line two" in message
    assert "> line three" in message
    row = _read_row(tmp_path)
    # The cached path round-trips unchanged.
    assert row["transcript_path"] == str(txt)
    assert row["terminal_state"] == "DONE"
    # The no-LLM plain digest carries no detail marker.
    assert row["detail"] is None


async def test_vtt_process_record_handles_txt_cached_path_with_llm(
    store: QueueStore, tmp_path: Path
) -> None:
    """A cached .txt still delivers the plain digest when an LLM key is set.

    A ``.txt`` carries no timed cues, so ``_parse_cues_for_llm`` returns an
    empty list and the vtt LLM branch is skipped even with a key configured.
    The record is delivered through the same ``first_lines`` fallback, which is
    the roll-back guarantee for an audio record whose transcript already landed.
    """
    # Arrange
    settings = _make_settings(tmp_path, backend="vtt", openai_api_key="sk-test")
    now = 1000.0
    _enqueue_cached_txt(
        store, settings, _VIDEO_ID, now, "line one\nline two\nline three\n"
    )
    record = _fetch(store, now)
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()
    analyze = FakeAnalyze(result=_takeaways())

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=_probe_unused,
        send=sender,
        send_embeds=embeds_sender,
        analyze=analyze,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert: no cues means no LLM, so the plain digest is sent, not embeds.
    assert analyze.calls == 0
    assert embeds_sender.calls == 0
    assert sender.calls == 1
    assert "> line one" in sender.contents[0]
    assert _read_row(tmp_path)["terminal_state"] == "DONE"


async def test_vtt_process_record_handles_missing_transcript_file(
    store: QueueStore, tmp_path: Path
) -> None:
    """A missing cached file leaves the record retryable, never terminal.

    The unreadable path yields no cues, so the LLM branch is skipped. The plain
    digest then fails to read the file and the outer ``except Exception`` in
    ``_process_record`` reschedules the record. This pins the observable
    roll-back guarantee (the record stays retryable) rather than the internal
    crash path, which is the safer interpretation.
    """
    # Arrange
    settings = _make_settings(tmp_path, backend="vtt", openai_api_key="sk-test")
    now = 1000.0
    store.enqueue(_entry(_VIDEO_ID), now=now)
    missing = settings.transcript_dir / f"{_VIDEO_ID}.txt"
    store.mark_attempt(
        _VIDEO_ID, now=now, next_attempt_at=now, transcript_path=str(missing)
    )
    record = _fetch(store, now)
    analyze = FakeAnalyze(result=_takeaways())

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=_probe_unused,
        send=CountingSend(),
        send_embeds=CountingSendEmbeds(),
        analyze=analyze,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert analyze.calls == 0
    assert store.counts() == {"pending": 1}
    updated = store.next_due(now=now + 100_000)
    assert updated is not None
    assert updated.next_attempt_at > now
    assert updated.terminal_state is None


async def test_audio_process_record_falls_back_to_first_lines_for_txt_path(
    store: QueueStore, tmp_path: Path
) -> None:
    """The audio processor treats a cached .txt as ready text and skips stages.

    A transcript_path already on disk means download, compress, and transcribe
    are all skipped; the .txt content goes straight to the send stage.
    """
    # Arrange
    settings = _make_settings(tmp_path, backend="audio", openai_api_key=None)
    now = 1000.0
    _enqueue_cached_txt(store, settings, _VIDEO_ID, now, "alpha\nbeta\ngamma\n")
    record = _fetch(store, now)
    download = FakeDownload()
    compress = FakeCompress()
    transcribe = FakeTranscribe()
    sender = CountingSend()
    embeds_sender = CountingSendEmbeds()

    # Act
    await _process_record_audio(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        download_audio=download,
        compress_audio=compress,
        transcribe=transcribe,
        analyze=FakeAnalyze(result=_takeaways()),
        send=sender,
        send_embeds=embeds_sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    assert download.call_count == 0
    assert compress.call_count == 0
    assert transcribe.call_count == 0
    assert sender.calls == 1
    assert embeds_sender.calls == 0
    assert "> alpha" in sender.contents[0]
    assert _read_row(tmp_path)["terminal_state"] == "DONE"


# ---------------------------------------------------------------------------
# VTT roll-back: regression guards
# ---------------------------------------------------------------------------


def test_vtt_process_record_uses_new_player_client_ladder(tmp_path: Path) -> None:
    """The vtt path uses the new player-client ladder, not the broken one.

    ``tv_embedded`` was removed in 2026; a roll-back must not resurrect it.
    """
    # Arrange / Act
    opts = build_ydl_opts(tmp_path)
    extractor_args = cast(dict[str, Any], opts["extractor_args"])
    youtube = cast(dict[str, Any], extractor_args["youtube"])

    # Assert
    assert youtube["player_client"] == [
        "visionos",
        "web_safari",
        "tv",
        "mweb",
        "web_embedded",
    ]


async def test_vtt_process_record_yields_lines_count_setting(
    store: QueueStore, tmp_path: Path
) -> None:
    """The transcript_lines cap is respected even for a raw .txt digest."""
    # Arrange
    settings = _make_settings(
        tmp_path, backend="vtt", openai_api_key=None, transcript_lines=5
    )
    now = 1000.0
    text = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
    _enqueue_cached_txt(store, settings, _VIDEO_ID, now, text)
    record = _fetch(store, now)
    sender = CountingSend()

    # Act
    await _process_record(
        record,
        settings,
        store,
        httpx2.AsyncClient(),
        opts={},
        probe=_probe_unused,
        send=sender,
        sleep=_noop_sleep,
        now=now,
    )

    # Assert
    quoted = [line for line in sender.contents[0].splitlines() if line.startswith("> ")]
    assert quoted == [f"> line {i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# Enqueue delay survives a restart
# ---------------------------------------------------------------------------


def test_transcript_backend_setting_persists_across_restart(tmp_path: Path) -> None:
    """The audio enqueue delay is persisted and survives a store reopen.

    Regression for the PR 2 enqueue-with-delay path: a record inserted with the
    audio backend's first-download delay must still carry that schedule after
    the process restarts.
    """
    # Arrange
    settings = _make_settings(tmp_path, backend="audio")
    now = 1000.0
    delay = settings.audio_download_delay_seconds

    # Act: first session enqueues with the audio delay, then closes.
    first = open_store(settings.queue_file)
    first.enqueue(_entry(_VIDEO_ID), now=now, delay_seconds=delay)
    first.close()

    # Act: reopen the same database, simulating a restart.
    reopened = open_store(settings.queue_file)
    try:
        # Assert
        assert reopened.next_due(now=now) is None
        record = reopened.next_due(now=now + delay)
        assert record is not None
        assert record.next_attempt_at == now + delay
    finally:
        reopened.close()
