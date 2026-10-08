"""yt-dlp probe-and-fetch for the transcript worker.

The queue hands the worker one pending video at a time, and the worker asks
YouTube whether the transcript is ready. This module owns that question. It
hides yt-dlp behind a small seam (``YoutubeDLProtocol``) so the worker and the
tests never touch the library's globals.

Probing and fetching are one call on purpose. A separate "is it ready?" probe
plus a later fetch would spend two YouTube requests per video, and lookups are
exactly what gets us rate limited. ``extract_info(url, download=True)`` fetches
the player response and writes the subtitle file in a single request, so the
worker spends its request budget once.

The subtitle language list is ``en,en-orig`` and never ``en.*``. yt-dlp treats
a trailing ``.*`` as a ``re.fullmatch`` pattern, which also matches translated
variants such as ``en-de-DE`` and ``en-ja``. Those extra requests are what
triggered the 429 storms that motivated this rule. Do not loosen it.

The retry boundary matters: ``NOT_READY`` means "no captions yet, ask again
later" and is retryable, while ``UNAVAILABLE`` means the video is gone or bot
checked and will never succeed. Collapsing the two would either retry forever
on dead videos or give up on videos still waiting for ASR. A 48 hour cutoff,
applied by the worker, is what separates "ASR still pending" from "captions
never", because to yt-dlp those two cases are byte-for-byte identical.

Threading contract: every function here is blocking. ``probe_and_fetch`` must be
called through ``asyncio.to_thread()`` so the event loop keeps serving requests
while yt-dlp waits on the network. The pure helpers are safe anywhere.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast
from urllib.parse import parse_qs, urlparse

__all__ = [
    "ProbeState",
    "ProbeResult",
    "build_ydl_opts",
    "probe_and_fetch",
    "classify_error",
    "parse_vtt",
    "parse_vtt_timed",
    "parse_srt_timed",
    "render_transcript_for_llm",
    "first_lines",
    "Cue",
]

logger = logging.getLogger(__name__)

# These are the exact yt-dlp error fragments we have seen live, in match
# priority order. A 429 is transient, an unavailable video is permanent, and a
# missing PO token means the request was rejected before subtitles were even
# considered, so it is retryable rather than a hard failure.
_RE_429 = re.compile(r"HTTP Error 429|Too Many Requests|RequestBlocked")
_RE_UNAVAILABLE = re.compile(
    r"This video is unavailable|Private video|"
    r"Sign in to confirm you're not a bot|terminated|removed"
)
_RE_PO_TOKEN = re.compile(r"PO Token was not provided|PO token was not provided")

# A WebVTT cue timing line, for example "00:00:01.000 --> 00:00:04.500" with
# optional cue settings such as "align:start position:0%" after it.
_RE_TIMESTAMP = re.compile(
    r"\d\d:\d\d:\d\d\.\d\d\d --> \d\d:\d\d:\d\d\.\d\d\d"
)

# Auto-captions wrap words in tags like <c.colorCCCCCC> and carry inline
# timing tags like <00:00:01.000>. Both are markup, not transcript text.
_RE_TAG = re.compile(r"<[^>]+>")

# Header lines that introduce a WebVTT block and are never transcript text.
_HEADER_PREFIXES = ("WEBVTT", "NOTE", "Kind:", "Language:")


class YoutubeDLProtocol(Protocol):
    """Subset of yt_dlp.YoutubeDL the probe uses. Lets tests inject fakes."""

    def __enter__(self) -> YoutubeDLProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...

    def extract_info(self, url: str, download: bool = True) -> dict[str, object]: ...


class ProbeState(StrEnum):
    """Outcome of one probe, from the worker's point of view.

    READY, NOT_READY, and RATE_LIMITED are all retryable in their own way, while
    UNAVAILABLE is permanent. The worker maps these onto queue terminal states.
    """

    READY = "READY"  # transcript file written, requested_subtitles non-empty
    NOT_READY = "NOT_READY"  # no subs; either pending or never (worker decides)
    UNAVAILABLE = "UNAVAILABLE"  # video private/removed/bot-checked; permanent
    RATE_LIMITED = "RATE_LIMITED"  # HTTP 429; transient


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The state of one probe plus an optional path and human detail."""

    state: ProbeState
    transcript_path: str | None
    detail: str | None


def build_ydl_opts(
    dest_dir: Path,
    *,
    cookies_file: Path | None = None,
    langs: tuple[str, ...] = ("en", "en-orig"),
) -> dict[str, object]:
    """Build the ``yt_dlp.YoutubeDL`` opts dict for a probe-and-fetch.

    Uses exact lang codes only. The regex ``en.*`` is a ``re.fullmatch`` filter
    that also matches translated variants (``en-de-DE``, ``en-ja``) and triggers
    429s. Do NOT loosen this without reading the upstream issue.

    ``retries`` is lowered from the yt-dlp default of 10 so an empty HLS
    response fails fast instead of printing a 4 second error burst, and
    ``socket_timeout`` bounds a hung connection. ``ignore_no_formats_error``
    keeps a format rotation from failing the probe. ``extractor_args`` tries the
    default web player first, then the ``ios`` and ``tv_embedded`` clients in
    order, since bot-checked egress IPs are often accepted by those.

    ``cookies_file`` is added as ``cookiefile`` only when provided so an
    unauthenticated probe sends no cookie header at all.
    """
    opts: dict[str, object] = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        # We want a DownloadError raised so the worker can classify it.
        "ignoreerrors": False,
        # Fail fast on an empty response: the default of 10 retries produces a
        # 4 second burst of "Did not get any data blocks" from the HLS
        # downloader. Three rides out a transient blip, and the worker's own
        # backoff handles anything worse.
        "retries": 3,
        # Fail fast on a hung connection instead of waiting on the OS default.
        # The 48 hour give-up is the upper bound; one probe must not run longer
        # than a minute.
        "socket_timeout": 30,
        # A page with no playable formats should not fail the probe. The worker
        # still marks the record NOT_READY and reschedules it, which keeps the
        # probe robust against format rotation.
        "ignore_no_formats_error": True,
        # Cluster egress IPs are commonly bot-checked on the default web client
        # but accepted on mobile or embedded clients. ``visionos`` is the
        # current preferred default; ``web_safari`` and ``mweb`` are common
        # fallbacks; ``tv`` and ``web_embedded`` are the last-resort embedded
        # clients. ``tv_embedded`` was removed in 2026.
        "extractor_args": {
            "youtube": {
                "player_client": [
                    "visionos",
                    "web_safari",
                    "tv",
                    "mweb",
                    "web_embedded",
                ]
            }
        },
    }
    if cookies_file is not None:
        opts["cookiefile"] = str(cookies_file)
    return opts


def classify_error(message: str) -> tuple[ProbeState, str]:
    """Map a yt-dlp error message to ``(ProbeState, short detail)``.

    Priority is 429, then unavailable, then missing PO token, then an unknown
    fallback. The fallback is NOT_READY rather than a hard failure: a message we
    have never seen is far more likely to be a transient extractor quirk than a
    permanently dead video. Callers that need to distinguish will see the raw
    message in the detail.
    """
    if _RE_429.search(message):
        return ProbeState.RATE_LIMITED, "rate_limited"
    if _RE_UNAVAILABLE.search(message):
        return ProbeState.UNAVAILABLE, "unavailable"
    if _RE_PO_TOKEN.search(message):
        return ProbeState.NOT_READY, "po_token"
    return ProbeState.NOT_READY, f"unknown:{message}"


def _video_id_from_url(url: str) -> str:
    """Return the YouTube video id from a watch or short URL.

    Handles ``watch?v=ID`` and the ``youtu.be/ID`` short form. Returns an empty
    string when neither shape yields an id, so the caller can skip the file scan
    instead of globbing for everything.
    """
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("v")
    if values:
        return values[0]
    path = parsed.path.rstrip("/")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _find_transcript(dest_dir: Path, video_id: str) -> Path | None:
    """Return the first fetched ``.vtt`` or ``.srt`` file for ``video_id``."""
    if not video_id:
        return None
    for pattern in (f"{video_id}*.vtt", f"{video_id}*.srt"):
        matches = sorted(dest_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def probe_and_fetch(
    url: str,
    *,
    opts: dict[str, object],
    ydl_class: Callable[..., YoutubeDLProtocol] | None = None,
    dest_dir: Path,
) -> ProbeResult:
    """Blocking probe. Call via ``asyncio.to_thread()`` from async code.

    Uses ``ydl_class`` if given, else ``yt_dlp.YoutubeDL``. One call fetches the
    player response (1 YouTube request), writes the subtitle file if available,
    and returns the info dict.

    Returns READY only when both a subtitle file for the video id exists on disk
    and ``info["requested_subtitles"]`` is non-empty. The file check alone is not
    enough: a stale file from an earlier attempt must not mark a new probe ready.
    Any yt-dlp error is classified into a retryable or permanent state.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Import inside the function so importing this module has no yt-dlp side
    # effects and tests can inject a fake YoutubeDL without the real library.
    import yt_dlp
    from yt_dlp.utils import YoutubeDLError

    factory: Callable[..., YoutubeDLProtocol]
    if ydl_class is None:
        # yt-dlp ships no type stubs, so its class is cast to the protocol we
        # rely on rather than taught to the type checker attribute by attribute.
        factory = cast(Callable[..., YoutubeDLProtocol], yt_dlp.YoutubeDL)
    else:
        factory = ydl_class

    info: dict[str, object] = {}
    try:
        with factory(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except YoutubeDLError as exc:
        # YoutubeDLError is the common base of DownloadError and ExtractorError,
        # so both the download layer and the extractor layer are covered.
        state, detail = classify_error(str(exc))
        return ProbeResult(state=state, transcript_path=None, detail=detail)

    video_id = _video_id_from_url(url)
    transcript = _find_transcript(dest_dir, video_id)
    if transcript is not None and info.get("requested_subtitles"):
        return ProbeResult(
            state=ProbeState.READY,
            transcript_path=str(transcript),
            detail=None,
        )
    return ProbeResult(
        state=ProbeState.NOT_READY,
        transcript_path=None,
        detail=None,
    )


def parse_vtt(text: str) -> list[str]:
    """Turn a YouTube auto-caption VTT into clean text lines.

    Header blocks (``WEBVTT``, ``NOTE``, ``Kind:``, ``Language:``) and cue
    timing lines are dropped, inline markup such as ``<c>`` and
    ``<00:00:01.000>`` is stripped, and runs of the same line are collapsed to
    one. Auto-captions repeat a line across successive cues for emphasis during
    scrolling, so without collapsing the transcript looks stuttered.
    """
    lines_out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        raw_lines = [line.rstrip("\r") for line in block.split("\n")]
        content_lines = [line for line in raw_lines if line.strip()]
        if not content_lines:
            continue
        first_line = content_lines[0].strip()
        if first_line.startswith(_HEADER_PREFIXES):
            continue
        if _RE_TIMESTAMP.match(first_line):
            content_lines = content_lines[1:]
        for raw in content_lines:
            cleaned = _RE_TAG.sub("", raw).strip()
            if not cleaned:
                continue
            if lines_out and lines_out[-1] == cleaned:
                continue
            lines_out.append(cleaned)
    return lines_out


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


@dataclass(frozen=True)
class Cue:
    """One timed subtitle cue: start time in seconds + clean text."""

    start: float
    text: str


# A cue timing line: "<start> --> <end>" with optional trailing cue settings
# such as "align:start position:0%". Both VTT and SRT share this shape, so one
# pattern serves both parsers. Only the start time is needed, but the end token
# is captured to keep the pattern explicit and readable.
_RE_CUE_TIMING = re.compile(r"^\s*(\S+)\s*-->\s*(\S+)(?:\s+.*)?$")

# A VTT or SRT timestamp with optional hours. VTT uses a period as the decimal
# separator and SRT uses a comma, so both are accepted here.
_RE_SUBTITLE_TS = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:[.,]\d+)?)$")


def _vtt_ts_to_seconds(ts: str) -> float:
    """Convert a VTT or SRT timestamp to total seconds.

    Accepts ``HH:MM:SS.mmm`` and ``HH:MM:SS,mmm`` with the hours part optional.
    The SRT comma form is accepted too so the SRT parser can reuse this helper.
    Raises ValueError on a malformed token so the caller can skip the cue.
    """
    match = _RE_SUBTITLE_TS.match(ts.strip())
    if match is None:
        raise ValueError(f"malformed subtitle timestamp: {ts!r}")
    hours = int(match.group(1)) if match.group(1) is not None else 0
    minutes = int(match.group(2))
    seconds = float(match.group(3).replace(",", "."))
    return hours * 3600 + minutes * 60 + seconds


def _parse_timed_cues(text: str) -> list[Cue]:
    """Walk VTT or SRT blocks into cues, preserving each cue's start time.

    Header, NOTE, STYLE, and REGION blocks are skipped. A cue is found by
    scanning a block for its timing line, which also skips an SRT numeric index
    line and a VTT cue identifier. Inline ``<...>`` markup is stripped and a
    line that repeats the previous line within the same cue is collapsed, which
    matches the rolling-caption stutter the plain text parser already handles.
    Malformed timestamps and blocks without a timing line are skipped silently.
    """
    cues: list[Cue] = []
    skip_prefixes = (*_HEADER_PREFIXES, "STYLE", "REGION")
    for block in re.split(r"\n\s*\n", text):
        raw_lines = [line.rstrip("\r") for line in block.split("\n")]
        content_lines = [line for line in raw_lines if line.strip()]
        if not content_lines:
            continue
        if content_lines[0].strip().startswith(skip_prefixes):
            continue

        timing_index: int | None = None
        for index, line in enumerate(content_lines):
            if "-->" in line:
                timing_index = index
                break
        if timing_index is None:
            continue

        timing = _RE_CUE_TIMING.match(content_lines[timing_index])
        if timing is None:
            continue
        try:
            start = _vtt_ts_to_seconds(timing.group(1))
        except ValueError:
            continue

        parts: list[str] = []
        for raw in content_lines[timing_index + 1 :]:
            cleaned = _RE_TAG.sub("", raw).strip()
            if not cleaned:
                continue
            if parts and parts[-1] == cleaned:
                continue
            parts.append(cleaned)
        if not parts:
            continue
        cues.append(Cue(start=start, text=" ".join(parts)))
    return cues


def parse_vtt_timed(text: str) -> list[Cue]:
    """Parse WebVTT text into timed cues, preserving cue start times.

    - Strips WEBVTT header, Kind:, Language:, NOTE blocks.
    - Strips inline timing tags (<00:00:01.000>), color tags (<c.color>, </c>),
      and other <...> markup.
    - Collapses repeated lines within a single cue.
    - Returns cues in source order; skips malformed blocks silently.
    """
    return _parse_timed_cues(text)


def parse_srt_timed(text: str) -> list[Cue]:
    """Parse SubRip text into timed cues, preserving cue start times.

    SRT format: number line, then "HH:MM:SS,mmm --> HH:MM:SS,mmm", then text,
    then blank line. SRT uses comma as decimal separator in timestamps (vs
    VTT's period); the shared timestamp helper accepts both. The numeric index
    line at the start of each block is skipped.
    """
    return _parse_timed_cues(text)


def _format_anchor(seconds: float) -> str:
    """Format seconds as a bracketed anchor, e.g. ``[1:05]`` or ``[1:01:01]``.

    This is the bracketed counterpart of ``format_timestamp`` in youtube.py,
    which returns ``0:00`` without brackets. Kept separate in this slice so the
    two modules do not depend on each other.
    """
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes}:{secs:02d}]"


def render_transcript_for_llm(
    cues: list[Cue],
    *,
    block_seconds: int = 25,
) -> str:
    """Render a timed transcript for LLM consumption.

    Merges adjacent cues into ~block_seconds-wide chunks, each prefixed with a
    [m:ss] anchor the model can quote in its output. This gives the LLM the
    timestamp text it needs to produce accurate, linkable timestamps.

    Example output:
        [0:00] Hello and welcome to the show. Today we're talking about Postgres.
        [0:25] Let's dive into the first topic...
        [12:34] This is the key insight about indexing.
    """
    if not cues:
        return ""

    lines: list[str] = []
    block_start = cues[0].start
    block_texts: list[str] = []
    for cue in cues:
        # Close the block once this cue would push its span past the window.
        if block_texts and cue.start - block_start > block_seconds:
            lines.append(f"{_format_anchor(block_start)} {' '.join(block_texts)}")
            block_start = cue.start
            block_texts = []
        block_texts.append(cue.text)
    # cues is non-empty here, so the loop above always leaves a final block.
    lines.append(f"{_format_anchor(block_start)} {' '.join(block_texts)}")
    return "\n".join(lines)
