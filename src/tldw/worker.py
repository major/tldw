"""Background worker that drains the transcript queue.

This module is the heartbeat of the transcript pipeline. It runs forever,
pulling one record at a time out of the SQLite queue, downloading the video's
audio track with yt-dlp, compressing it with ffmpeg, transcribing it with
OpenAI, and posting the takeaways to Discord.

The design is deliberately serial. One worker, one record, one YouTube request
at a time is the main defense against rate limiting: a burst of notifications
for a freshly published batch of videos cannot thunder the extractor. A small
pacing sleep between records adds a second layer for a full backlog.

Crash safety comes from ordering. Before the blocking download runs, the
record's attempt count and last-attempt time are written to disk, and each
stage's artifact (compressed audio, then the ``.txt`` transcript) is persisted
before the next stage runs. If the process is killed mid-stage, the row is
already rescheduled and will resume at the last completed artifact instead of
being stuck as if the attempt never happened.

The queue caches the transcript path on the record. Once the ``.txt``
transcript is on disk, later passes skip download, compress, and transcribe and
go straight to the LLM and send. That means a Discord outage costs retries
against Discord, never against the YouTube request budget.

Two terminal outcomes are not failures of the process. A video whose enqueue
time is older than ``giveup_seconds`` (48 hours by default) is marked
GIVE_UP_NEVER. A video that hits the rate limit
``DEFAULT_MAX_RATE_LIMIT_ATTEMPTS`` times in a row is marked
GIVE_UP_RATE_LIMITED_DEAD so a dead rate-limit budget stops consuming the loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import openai
import httpx2
from fastapi import FastAPI
from pydantic import ValidationError
import pydantic_ai.exceptions

from tldw.audio import (
    CompressError,
    DownloadResult,
    ProbeState,
    build_audio_ydl_opts,
    compress_audio,
)
from tldw.audio import download_audio as audio_download
from tldw.backoff import backoff_delay
from tldw.config import Settings
from tldw.discord import build_takeaway_embeds, format_message
from tldw.discord import send as discord_send
from tldw.discord import send_embeds as discord_send_embeds
from tldw.llm import OpenAIAnalyzer, TakeawayAnalyzer, Takeaways
from tldw.queue import QueueRecord, QueueStore, TerminalState
from tldw.transcribe import TranscribeError, transcribe

__all__ = ["transcript_loop"]

logger = logging.getLogger(__name__)

# Seconds to sleep when the queue is empty. Long enough to avoid busy-waiting,
# short enough to feel responsive when a new video arrives.
DEFAULT_EMPTY_QUEUE_SLEEP: float = 60.0
# Per-record pacing. The serial design is the main rate-limit defense; this is
# the secondary pacing for a full backlog.
DEFAULT_RECORD_PACING: float = 5.0
# After this many consecutive RATE_LIMITED results for one record, give up
# rather than drain the rate-limit budget. With backoff base 300s and cap
# 3600s, seven waits total roughly 4 hours before the eighth 429 trips.
DEFAULT_MAX_RATE_LIMIT_ATTEMPTS: int = 8

# Type aliases for the injectable seams. The download and compress signatures
# are left open (``...``) because the real implementations use keyword-only
# arguments; tests inject callables with the same shape.
SleepFn = Callable[[float], Awaitable[None]]
SendFn = Callable[..., Awaitable[None]]
SendEmbedsFn = Callable[..., Awaitable[None]]
AnalyzeFn = Callable[[str, str, str], Awaitable[Takeaways]]
"""Signature: (rendered_transcript, video_id, title) -> Takeaways."""
DownloadAudioFn = Callable[..., DownloadResult]
"""Signature mirrors the real download; the real call is kwarg-only, tests inject the same shape."""
CompressAudioFn = Callable[..., Path]
"""Signature mirrors the real compress; the real call is kwarg-only, tests inject the same shape."""
TranscribeFn = Callable[[Path], Awaitable[str]]
"""Signature: (audio_path) -> transcript text."""


def _default_analyze_factory(settings: Settings) -> AnalyzeFn:
    """Build the production analyzer from settings. Returns a closure."""
    analyzer: TakeawayAnalyzer = OpenAIAnalyzer(settings)

    async def _analyze(rendered: str, video_id: str, title: str) -> Takeaways:
        return await analyzer.analyze(rendered, video_id=video_id, title=title)

    return _analyze


def _default_compress_audio() -> CompressAudioFn:
    """Bind compress_audio for worker use; closure for the asyncio.to_thread call."""

    def _compress(
        src: Path, dst: Path, *, bitrate: str = "32k", timeout_s: float = 900.0
    ) -> Path:
        return compress_audio(src, dst, bitrate=bitrate, timeout_s=timeout_s)

    return _compress


def _default_transcribe_factory(settings: Settings) -> TranscribeFn:
    """Bind transcribe() with settings; the worker injects the seam."""

    async def _transcribe(path: Path) -> str:
        return await transcribe(
            path,
            model=settings.transcribe_model,
            langs=settings.transcribe_langs,
            api_key=settings.openai_api_key or "",
            base_url=settings.openai_base_url,
            timeout_s=settings.transcribe_timeout_seconds,
        )

    return _transcribe


async def transcript_loop(
    app: FastAPI,
    client: httpx2.AsyncClient,
    *,
    sleep: SleepFn = asyncio.sleep,
    send: SendFn = discord_send,
    send_embeds: SendEmbedsFn = discord_send_embeds,
    analyze: AnalyzeFn | None = None,
    download_audio: DownloadAudioFn | None = None,
    compress_audio: CompressAudioFn | None = None,
    transcribe: TranscribeFn | None = None,
) -> None:
    """Drain the queue forever, one record at a time, until cancelled.

    Mirrors the lifecycle of ``_renewal_loop`` in app.py: it returns immediately
    when the pipeline is not configured, then runs forever in a ``while True``,
    catching and logging per-record exceptions so a single bad record never
    kills the loop.

    The default ``send`` and ``send_embeds`` are the real modules; tests inject
    fakes through the keyword arguments. The default ``sleep`` is
    ``asyncio.sleep``. ``analyze``, ``transcribe``, ``download_audio``, and
    ``compress_audio`` are resolved once from settings when not injected, so the
    seams can be faked in tests without touching the network or the filesystem.
    """
    settings: Settings = app.state.settings
    if not settings.discord_webhook_url:
        logger.info("transcript worker skipping: TLDW_DISCORD_WEBHOOK_URL is not set")
        return
    # The audio backend has no plain-digest fallback without a key: without it,
    # transcription can never run, so the worker refuses to start.
    if not settings.openai_api_key:
        logger.error("audio backend requires TLDW_OPENAI_API_KEY; worker not starting")
        return
    if analyze is None:
        analyze = _default_analyze_factory(settings)
    if transcribe is None:
        transcribe = _default_transcribe_factory(settings)
    if download_audio is None:
        download_audio = audio_download
    if compress_audio is None:
        compress_audio = _default_compress_audio()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    await _run_loop(
        app,
        client,
        settings,
        sleep=sleep,
        download_audio=download_audio,
        compress_audio=compress_audio,
        transcribe=transcribe,
        analyze=analyze,
        send=send,
        send_embeds=send_embeds,
    )


async def _run_loop(
    app: FastAPI,
    client: httpx2.AsyncClient,
    settings: Settings,
    *,
    sleep: SleepFn,
    download_audio: DownloadAudioFn,
    compress_audio: CompressAudioFn,
    transcribe: TranscribeFn,
    analyze: AnalyzeFn,
    send: SendFn,
    send_embeds: SendEmbedsFn,
) -> None:
    """Run the serial drain loop.

    The store is resolved once from the app state. Each iteration pulls the
    oldest due record and hands it to ``_process_record_audio`` with the shared
    seams, then sleeps the per-record pacing. An empty queue sleeps the long
    empty-queue interval. A per-record exception is logged and the loop moves
    on, so one bad record can never kill the worker.
    """
    store: QueueStore = app.state.queue

    while True:
        try:
            record = store.next_due()
        except Exception:
            logger.exception("queue next_due raised")
            await sleep(DEFAULT_EMPTY_QUEUE_SLEEP)
            continue

        if record is None:
            await sleep(DEFAULT_EMPTY_QUEUE_SLEEP)
            continue

        try:
            await _process_record_audio(
                record,
                settings,
                store,
                client,
                download_audio=download_audio,
                compress_audio=compress_audio,
                transcribe=transcribe,
                analyze=analyze,
                send=send,
                send_embeds=send_embeds,
                sleep=sleep,
            )
        except Exception:
            # A record that fails outside its own stage try blocks must not
            # kill the loop; log it and move on to the next record.
            logger.exception("processing raised for %s", record.video_id)
        await sleep(DEFAULT_RECORD_PACING)


def first_lines(text: str, n: int) -> list[str]:
    """Return the first ``n`` non-empty lines of ``text``, each stripped.

    Raises ValueError when ``n`` is negative, since that is a caller bug rather
    than an empty result.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            lines.append(stripped)
    return lines[:n]


# ---------------------------------------------------------------------------
# Audio backend
# ---------------------------------------------------------------------------


async def _process_record_audio(
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    client: httpx2.AsyncClient,
    *,
    download_audio: DownloadAudioFn,
    compress_audio: CompressAudioFn,
    transcribe: TranscribeFn,
    analyze: AnalyzeFn,
    send: SendFn,
    send_embeds: SendEmbedsFn,
    sleep: SleepFn,
    now: float | None = None,
) -> None:
    """Process one record through the audio pipeline.

    Stage sequence:
      1. 48h give-up.
      2. If transcript_path is set, read it and skip to LLM/send.
      3. If audio_path is set, skip download+compress and go to transcribe.
      4. Otherwise: mark_attempt(now), download_audio (to_thread).
         - RATE_LIMITED streak / give-up
         - UNAVAILABLE give-up
         - NOT_READY backoff
         - READY: compress_audio (to_thread); on CompressError, reschedule
         - On compress success, delete raw, persist audio_path.
      5. transcribe (await); on 4xx APIStatusError → GIVE_UP_AUDIO.
      6. Write .txt transcript; delete compressed audio; reschedule
         transcript_path= (so a worker crash resumes at the LLM stage).
      7. analyze (LLM); on failure fall back to plain digest (existing
         behavior, adapted for the .txt path).
      8. send_embeds; mark_terminal(DONE, detail="llm_embeds" or "plain").

    Each stage's success is persisted before the next stage runs, so a crash
    resumes at the last completed artifact. ``now`` is the clock seam for tests
    and defaults to ``time.time()``. ``sleep`` is accepted for symmetry with
    the other seams and so future per-step pacing can use it without a
    signature change.
    """
    if now is None:
        now = time.time()

    # 48 hour give-up. Measured from enqueued_at, not the video's publish time:
    # the cutoff is about how long we have been trying, not how old the video is.
    if now - record.enqueued_at > settings.giveup_seconds:
        store.mark_terminal(
            record.video_id,
            TerminalState.GIVE_UP_NEVER,
            detail=f"no transcript after {settings.giveup_seconds}s",
        )
        return

    text: str
    if record.transcript_path is not None:
        # The .txt transcript is already cached, so skip download, compress, and
        # transcribe and go straight to the LLM/send stage.
        txt_path = Path(record.transcript_path)
        try:
            text = txt_path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("cached transcript unreadable for %s", record.video_id)
            delay = backoff_delay(
                record.attempts + 1,
                base=settings.poll_base_seconds,
                cap=settings.poll_cap_seconds,
                jitter=0.0,
            )
            store.reschedule(record.video_id, next_attempt_at=now + delay)
            return
        logger.info(
            "using cached transcript: video=%s path=%s", record.video_id, txt_path
        )
    else:
        if record.audio_path is not None:
            # The compressed audio is cached; skip download and compress.
            compressed = Path(record.audio_path)
            logger.info(
                "using cached audio: video=%s path=%s", record.video_id, compressed
            )
        else:
            raw = await _download_stage(
                record, settings, store, download_audio=download_audio, now=now
            )
            if raw is None:
                return
            compressed = await _compress_stage(
                raw, record, settings, store, compress_audio=compress_audio, now=now
            )
            if compressed is None:
                return
            # Persist before transcribing so a crash resumes at transcribe.
            store.reschedule(record.video_id, audio_path=str(compressed))
        transcribed = await _transcribe_stage(
            compressed, record, settings, store, transcribe=transcribe, now=now
        )
        if transcribed is None:
            return
        text = transcribed

    await _llm_and_send_stage(
        text,
        record,
        settings,
        store,
        client,
        analyze=analyze,
        send=send,
        send_embeds=send_embeds,
        now=now,
    )


async def _download_stage(
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    *,
    download_audio: DownloadAudioFn,
    now: float,
) -> Path | None:
    """Download the raw audio for one record.

    Returns the raw file path, or None when the worker should stop this pass
    (a terminal state was set or the record was rescheduled). The attempt is
    counted before the blocking download, so a process killed mid-download
    still records the attempt.
    """
    store.mark_attempt(record.video_id, now=now)
    opts = build_audio_ydl_opts(
        settings.audio_dir, cookies_file=settings.ytdlp_cookies_file
    )
    try:
        result = await asyncio.to_thread(
            download_audio,
            record.url,
            opts=opts,
            ydl_class=None,
            dest_dir=settings.audio_dir,
        )
    except Exception:
        logger.exception("audio download raised for %s", record.video_id)
        delay = backoff_delay(
            record.attempts + 1,
            base=settings.poll_base_seconds,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return None

    if result.state is ProbeState.READY:
        if result.path is None:
            # A READY result without a path should not happen. Reschedule with
            # backoff rather than leaving the record immediately due again.
            logger.info("audio ready: video=%s path=None", record.video_id)
            delay = backoff_delay(
                record.attempts + 1,
                base=settings.poll_base_seconds,
                cap=settings.poll_cap_seconds,
                jitter=0.0,
            )
            store.reschedule(record.video_id, next_attempt_at=now + delay)
            return None
        path = Path(result.path)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        logger.info(
            "audio ready: video=%s path=%s size_bytes=%d",
            record.video_id,
            path,
            size,
        )
        return path

    if result.state is ProbeState.RATE_LIMITED:
        # Count consecutive 429s on the row. Any other outcome resets this to 0,
        # so the give-up budget only trips on a real streak.
        new_streak = record.rate_limit_streak + 1
        if new_streak >= DEFAULT_MAX_RATE_LIMIT_ATTEMPTS:
            store.mark_terminal(
                record.video_id,
                TerminalState.GIVE_UP_RATE_LIMITED_DEAD,
                detail=result.detail,
            )
        else:
            delay = backoff_delay(
                new_streak,
                base=300.0,
                cap=settings.poll_cap_seconds,
                jitter=0.0,
            )
            store.reschedule(
                record.video_id,
                next_attempt_at=now + delay,
                rate_limit_streak=new_streak,
            )
        return None

    if result.state is ProbeState.UNAVAILABLE:
        store.mark_terminal(
            record.video_id,
            TerminalState.GIVE_UP_UNAVAILABLE,
            detail=result.detail,
        )
        return None

    # NOT_READY: the audio is not available yet. Back off and reset the streak.
    delay = backoff_delay(
        record.attempts + 1,
        base=settings.poll_base_seconds,
        cap=settings.poll_cap_seconds,
        jitter=0.0,
    )
    store.reschedule(
        record.video_id,
        next_attempt_at=now + delay,
        rate_limit_streak=0,
    )
    return None


async def _compress_stage(
    raw: Path,
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    *,
    compress_audio: CompressAudioFn,
    now: float,
) -> Path | None:
    """Compress the raw download into the small Opus file the API wants.

    Returns the compressed path, or None when ffmpeg failed and the record was
    rescheduled. On success the raw file is deleted (best effort) so only the
    compressed artifact stays on disk.
    """
    dst = settings.audio_dir / f"{record.video_id}.{settings.audio_format}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(
            compress_audio,
            raw,
            dst,
            bitrate=settings.audio_bitrate,
            timeout_s=settings.ffmpeg_timeout_seconds,
        )
    except CompressError:
        logger.exception("compress failed for %s", record.video_id)
        delay = backoff_delay(
            record.attempts + 1,
            base=300.0,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return None
    # Best effort: a failed unlink must not fail the record; the compressed
    # file is the artifact that matters from here on.
    try:
        raw.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete raw audio %s", raw)
    return dst


async def _transcribe_stage(
    compressed: Path,
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    *,
    transcribe: TranscribeFn,
    now: float,
) -> str | None:
    """Transcribe the compressed audio and persist the .txt transcript.

    Returns the transcript text, or None when the record was resolved or
    rescheduled. A permanent 4xx (except 429) marks GIVE_UP_AUDIO; a 429,
    connection error, or timeout reschedules with backoff. On success the .txt
    is written, the compressed audio is deleted, and the record is rescheduled
    with ``transcript_path`` set so a crash resumes at the LLM stage.
    """
    try:
        text = await transcribe(compressed)
    except TranscribeError as exc:
        status = exc.status_code
        if status is not None and 400 <= status < 500 and status != 429:
            # A permanent client error: retrying cannot fix a bad upload.
            store.mark_terminal(
                record.video_id, TerminalState.GIVE_UP_AUDIO, detail=str(exc)
            )
            return None
        # 429 and 5xx are transient; fall through to the backoff reschedule.
        logger.warning("transcription retryable for %s: %s", record.video_id, exc)
        delay = backoff_delay(
            record.attempts + 1,
            base=settings.poll_base_seconds,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return None
    except (
        openai.RateLimitError,
        openai.APIConnectionError,
        asyncio.TimeoutError,
    ) as exc:
        # The production transcribe wraps APIStatusError, but a fake seam or a
        # future direct call can still surface these; treat them as transient.
        logger.warning("transcription retryable for %s: %s", record.video_id, exc)
        delay = backoff_delay(
            record.attempts + 1,
            base=settings.poll_base_seconds,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return None
    except openai.APIError as exc:
        store.mark_terminal(
            record.video_id, TerminalState.GIVE_UP_AUDIO, detail=str(exc)
        )
        return None
    except Exception as exc:
        logger.warning("transcription raised for %s: %s", record.video_id, exc)
        delay = backoff_delay(
            record.attempts + 1,
            base=settings.poll_base_seconds,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return None

    txt_path = settings.transcript_dir / f"{record.video_id}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    # Best effort: the .txt is now the artifact the LLM stage reads.
    try:
        compressed.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete compressed audio %s", compressed)
    store.reschedule(
        record.video_id,
        next_attempt_at=now + DEFAULT_RECORD_PACING,
        transcript_path=str(txt_path),
    )
    return text


async def _llm_and_send_stage(
    text: str,
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    client: httpx2.AsyncClient,
    *,
    analyze: AnalyzeFn,
    send: SendFn,
    send_embeds: SendEmbedsFn,
    now: float,
) -> None:
    """Run the LLM takeaway path over plain text, then fall back to a digest.

    The audio backend has no timed cues, so the LLM gets the raw transcript and
    ``snap_timestamps`` is not called: bullets may or may not carry timestamps.
    An empty transcript, a missing key, or an LLM failure sends the plain
    digest. A send failure reschedules without bumping attempts.
    """

    async def _send_plain(detail: str | None) -> None:
        """Send the plain text digest and mark the record DONE."""
        lines = first_lines(text, settings.transcript_lines)
        message = format_message(record.title, record.channel_name, record.url, lines)
        logger.info(
            "calling discord: video=%s kind=plain digest_lines=%d",
            record.video_id,
            len(lines),
        )
        _t0 = time.monotonic()
        await send(client, settings.discord_webhook_url, message)
        logger.info(
            "called discord: video=%s kind=plain duration_s=%.2f",
            record.video_id,
            time.monotonic() - _t0,
        )
        store.mark_terminal(record.video_id, TerminalState.DONE, detail=detail)

    try:
        if not text.strip():
            # An empty transcript still notifies; it is not a failure, so it
            # carries its own detail marker rather than the fallback one.
            logger.info(
                "empty transcript for %s, sending plain digest", record.video_id
            )
            await _send_plain(detail="empty_transcript")
            return
        if settings.openai_api_key:
            logger.info(
                "handing off to LLM: video=%s model=%s max_output_tokens=%d "
                "transcript_chars=%d",
                record.video_id,
                settings.openai_model,
                settings.llm_max_output_tokens,
                len(text),
            )
            _llm_t0 = time.monotonic()
            try:
                takeaways = await analyze(text, record.video_id, record.title)
            except (
                TimeoutError,
                ValidationError,
                openai.APIError,
                openai.APIConnectionError,
                pydantic_ai.exceptions.UnexpectedModelBehavior,
            ) as exc:
                # APIConnectionError is an APIError subclass; listing both is
                # explicit about the failures we expect from the OpenAI API.
                logger.warning(
                    "LLM analysis failed for %s, falling back to plain digest: %s",
                    record.video_id,
                    exc,
                )
                await _send_plain(detail="llm_fallback")
                return
            _bullets = sum(len(t.bullets) for t in takeaways.items)
            logger.info(
                "LLM returned: video=%s takeaways=%d bullets=%d duration_s=%.2f",
                record.video_id,
                len(takeaways.items),
                _bullets,
                time.monotonic() - _llm_t0,
            )
            # No cues to snap to: the LLM may return bullets with or without
            # timestamps, and both render correctly downstream.
            embeds = build_takeaway_embeds(
                takeaways,
                video_id=record.video_id,
                video_url=record.url,
                channel_name=record.channel_name,
            )
            logger.info(
                "calling discord: video=%s kind=embeds embed_count=%d",
                record.video_id,
                len(embeds),
            )
            _discord_t0 = time.monotonic()
            await send_embeds(client, settings.discord_webhook_url, embeds)
            logger.info(
                "called discord: video=%s kind=embeds duration_s=%.2f",
                record.video_id,
                time.monotonic() - _discord_t0,
            )
            store.mark_terminal(
                record.video_id,
                TerminalState.DONE,
                detail="llm_embeds",
            )
            return
        await _send_plain(detail=None)
    except Exception:
        logger.exception("send raised for %s", record.video_id)
        delay = backoff_delay(
            record.attempts + 1,
            base=300.0,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        # A send failure is not a download, so it must not bump attempts. The
        # cached transcript_path means the retry will not re-download audio.
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return
