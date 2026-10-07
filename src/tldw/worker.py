"""Background worker that drains the transcript queue.

This module is the heartbeat of the transcript pipeline. It runs forever,
pulling one record at a time out of the SQLite queue, asking yt-dlp whether the
transcript is ready, and posting the digest to Discord when it is.

The design is deliberately serial. One worker, one record, one YouTube request
at a time is the main defense against rate limiting: a burst of notifications
for a freshly published batch of videos cannot thunder the extractor. A small
pacing sleep between records adds a second layer for a full backlog.

Crash safety comes from ordering. Before the blocking probe runs, the record's
attempt count and last-attempt time are written to disk. If the process is
killed mid-probe, the row is already rescheduled and will be picked up again
instead of being stuck as if the attempt never happened.

The queue caches the transcript path on the record. Once a subtitle file is on
disk, later passes skip the probe and go straight to reading, parsing, and
sending. That means a Discord outage costs retries against Discord, never
against the YouTube request budget.

Two terminal outcomes are not failures of the process. A video whose enqueue
time is older than ``giveup_seconds`` (48 hours by default) is marked
GIVE_UP_NEVER: to yt-dlp, "captions still pending" and "captions will never
come" look identical, and the cutoff is how we tell them apart. A video that
hits the rate limit ``DEFAULT_MAX_RATE_LIMIT_ATTEMPTS`` times in a row is marked
GIVE_UP_RATE_LIMITED_DEAD so a dead rate-limit budget stops consuming the loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx2
from fastapi import FastAPI

from tldw.backoff import backoff_delay
from tldw.config import Settings
from tldw.discord import format_message
from tldw.discord import send as discord_send
from tldw.queue import QueueRecord, QueueStore, TerminalState
from tldw.transcript import (
    ProbeResult,
    ProbeState,
    build_ydl_opts,
    first_lines,
    parse_srt_timed,
    parse_vtt_timed,
)
from tldw.transcript import probe_and_fetch as transcript_probe

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

# Type aliases for the injectable seams. The probe and send signatures are left
# open (``...``) because the real implementations use keyword-only arguments;
# tests inject callables with the same shape.
SleepFn = Callable[[float], Awaitable[None]]
ProbeFn = Callable[..., ProbeResult]
SendFn = Callable[..., Awaitable[None]]


async def transcript_loop(
    app: FastAPI,
    client: httpx2.AsyncClient,
    *,
    sleep: SleepFn = asyncio.sleep,
    probe: ProbeFn = transcript_probe,
    send: SendFn = discord_send,
) -> None:
    """Drain the queue forever, one record at a time, until cancelled.

    Mirrors the lifecycle of ``_renewal_loop`` in app.py: it returns immediately
    when the pipeline is not configured, then runs forever in a ``while True``,
    catching and logging per-record exceptions so a single bad record never
    kills the loop.

    The default ``probe`` and ``send`` are the real modules; tests inject fakes
    through the keyword arguments. The default ``sleep`` is ``asyncio.sleep``.
    """
    settings: Settings = app.state.settings
    if not settings.discord_webhook_url:
        logger.info("transcript worker skipping: TLDW_DISCORD_WEBHOOK_URL is not set")
        return
    settings.transcript_dir.mkdir(parents=True, exist_ok=True)
    opts = build_ydl_opts(
        settings.transcript_dir,
        cookies_file=settings.ytdlp_cookies_file,
        langs=tuple(settings.transcript_langs),
    )
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
            await _process_record(
                record,
                settings,
                store,
                client,
                opts=opts,
                probe=probe,
                send=send,
                sleep=sleep,
            )
        except Exception:
            # A record that fails outside the probe/send try blocks must not
            # kill the loop; log it and move on to the next record.
            logger.exception("processing raised for %s", record.video_id)
        await sleep(DEFAULT_RECORD_PACING)


def _digest_lines(path: Path, n: int) -> list[str]:
    """Return up to ``n`` blockquote-ready transcript lines from a subtitle file.

    Parses timed cues first so VTT headers and timing lines never reach the
    digest. Unknown extensions, and any parse error, fall back to ``first_lines``.
    """
    raw_text = path.read_text(encoding="utf-8")
    try:
        if path.suffix == ".vtt":
            cues = parse_vtt_timed(raw_text)
        elif path.suffix == ".srt":
            cues = parse_srt_timed(raw_text)
        else:
            return first_lines(raw_text, n)
    except Exception:
        # A malformed subtitle file must not crash the worker; degrade to the
        # old raw-line behavior so the digest is still sent.
        logger.warning("timed parse failed for %s, falling back to raw lines", path)
        return first_lines(raw_text, n)
    return [cue.text for cue in cues[:n]]


async def _process_record(
    record: QueueRecord,
    settings: Settings,
    store: QueueStore,
    client: httpx2.AsyncClient,
    *,
    opts: dict[str, object],
    probe: ProbeFn,
    send: SendFn,
    sleep: SleepFn,
    now: float | None = None,
) -> None:
    """Process one queue record: probe (if needed), parse, send, mark.

    ``attempts`` is incremented exactly once per probe, by the pre-probe
    ``mark_attempt`` call. Every later update goes through ``reschedule`` so a
    single probe never counts as two attempts. The rate-limit streak is bumped
    only on a RATE_LIMITED outcome and reset to zero on any other outcome,
    which is what makes the give-up budget mean "consecutive 429s".

    ``now`` is the clock seam for tests and defaults to ``time.time()``.
    ``sleep`` is accepted for symmetry with the other seams and so future
    per-step pacing can use it without changing the signature.
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

    record_path: str | None
    if record.transcript_path is None:
        # Count the attempt BEFORE the blocking probe. This is the only place
        # attempts is incremented, so one probe means one attempt. If the
        # process is killed mid-probe the row is already marked as attempted
        # and will be picked up again. jitter=0.0 keeps the schedule
        # deterministic: the serial single worker needs no fleet de-sync.
        store.mark_attempt(record.video_id, now=now)
        try:
            result = await asyncio.to_thread(
                probe,
                record.url,
                opts=opts,
                ydl_class=None,
                dest_dir=settings.transcript_dir,
            )
        except Exception:
            logger.exception("probe raised for %s", record.video_id)
            delay = backoff_delay(
                record.attempts + 1,
                base=settings.poll_base_seconds,
                cap=settings.poll_cap_seconds,
                jitter=0.0,
            )
            store.reschedule(record.video_id, next_attempt_at=now + delay)
            return

        if result.state is ProbeState.READY:
            store.reschedule(
                record.video_id,
                next_attempt_at=now + DEFAULT_RECORD_PACING,
                transcript_path=result.transcript_path,
                rate_limit_streak=0,
            )
            record_path = result.transcript_path
        elif result.state is ProbeState.RATE_LIMITED:
            # Count consecutive 429s on the row. Any other outcome below resets
            # this to 0, so the give-up budget only trips on a real streak.
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
            return
        elif result.state is ProbeState.UNAVAILABLE:
            store.mark_terminal(
                record.video_id,
                TerminalState.GIVE_UP_UNAVAILABLE,
                detail=result.detail,
            )
            return
        else:  # NOT_READY
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
            return
    else:
        # The subtitle file is already cached, so skip the probe entirely.
        record_path = record.transcript_path

    if record_path is None:
        # A READY probe without a path should not happen; log and leave the
        # record scheduled rather than crash on Path(None).
        logger.warning("record %s has no transcript path to send", record.video_id)
        return

    # Read the subtitle file, parse the cues first, slice, format, send. Parsing
    # before slicing keeps VTT headers and cue timing lines out of the digest.
    try:
        path = Path(record_path)
        lines = _digest_lines(path, settings.transcript_lines)
        message = format_message(record.title, record.channel_name, record.url, lines)
        await send(client, settings.discord_webhook_url, message)
    except Exception:
        logger.exception("send raised for %s", record.video_id)
        delay = backoff_delay(
            record.attempts + 1,
            base=300.0,
            cap=settings.poll_cap_seconds,
            jitter=0.0,
        )
        # A send failure is not a probe, so it must not bump attempts. The
        # cached transcript_path means the retry will not re-probe YouTube.
        store.reschedule(record.video_id, next_attempt_at=now + delay)
        return

    # Success.
    store.mark_terminal(record.video_id, TerminalState.DONE)
