"""Audio-only download and compression for the audio pipeline.

This module owns the two blocking steps that turn a YouTube URL into a small
mono Opus file the speech-to-text backend can consume:

1. ``download_audio`` asks yt-dlp for the smallest audio-only stream. It does
   not use yt-dlp's ``-x``/``--extract-audio`` option, because that runs an
   extra lossy encode before our own ffmpeg pass and would stack two
   generations of compression.
2. ``compress_audio`` runs one ffmpeg pass that downmixes to mono 16 kHz Opus,
   which is the shape the transcription service wants.

Both functions are blocking. The worker calls them through
``asyncio.to_thread(...)`` so the event loop keeps serving requests while
yt-dlp waits on the network or ffmpeg grinds on the CPU.

The download result uses the ``ProbeState`` enum this module owns so the worker
state machine does not need a second vocabulary for the same outcomes (READY,
NOT_READY, UNAVAILABLE, RATE_LIMITED).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast
from urllib.parse import parse_qs, urlparse

__all__ = [
    "ProbeState",
    "DownloadResult",
    "build_audio_ydl_opts",
    "download_audio",
    "CompressError",
    "compress_audio",
]

# The extensions yt-dlp can produce for an audio-only selection, in the order
# the scan prefers them. The extractor's reported ``info["ext"]`` is not
# reliable for the on-disk name, so the file is located by globbing instead.
_AUDIO_EXTS = ("webm", "m4a", "mp4", "opus", "mkv")

# These are the exact yt-dlp error fragments we have seen live, in match
# priority order. A 429 is transient, an unavailable video is permanent, and a
# missing PO token means the request was rejected before the media was even
# considered, so it is retryable rather than a hard failure.
_RE_429 = re.compile(r"HTTP Error 429|Too Many Requests|RequestBlocked")
_RE_UNAVAILABLE = re.compile(
    r"This video is unavailable|Private video|"
    r"Sign in to confirm you're not a bot|terminated|removed"
)
_RE_PO_TOKEN = re.compile(r"PO Token was not provided|PO token was not provided")


class YoutubeDLProtocol(Protocol):
    """Subset of yt_dlp.YoutubeDL the download uses. Lets tests inject fakes."""

    def __enter__(self) -> YoutubeDLProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...

    def extract_info(self, url: str, download: bool = True) -> dict[str, object]: ...


class ProbeState(StrEnum):
    """Outcome of one download, from the worker's point of view.

    READY, NOT_READY, and RATE_LIMITED are all retryable in their own way, while
    UNAVAILABLE is permanent. The worker maps these onto queue terminal states.
    """

    READY = "READY"  # media file written
    NOT_READY = "NOT_READY"  # no media; either pending or never (worker decides)
    UNAVAILABLE = "UNAVAILABLE"  # video private/removed/bot-checked; permanent
    RATE_LIMITED = "RATE_LIMITED"  # HTTP 429; transient


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


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """The state of one audio download plus an optional path and detail."""

    state: ProbeState
    path: str | None
    detail: str | None


def build_audio_ydl_opts(
    dest_dir: Path,
    *,
    cookies_file: Path | None = None,
) -> dict[str, object]:
    """Build the ``yt_dlp.YoutubeDL`` opts dict for an audio-only download.

    No ``-x``/``--extract-audio``: yt-dlp would decode the chosen stream and
    re-encode it, and ``compress_audio`` already does one ffmpeg pass. Two
    lossy encodes in a row waste quality for nothing.

    The format chain prefers an audio-only webm, then an audio-only m4a, then
    any audio-only stream, and ``format_sort`` breaks ties toward the smallest
    bitrate and size. The player client ladder starts with ``visionos``;
    ``tv_embedded`` was removed in 2026.

    ``cookies_file`` is added as ``cookiefile`` only when provided, so an
    unauthenticated download sends no cookie header at all.
    """
    opts: dict[str, object] = {
        "format": "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio",
        "format_sort": ["+abr", "+size"],
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        # We want a DownloadError raised so the worker can classify it.
        "ignoreerrors": False,
        "retries": 3,
        "socket_timeout": 30,
        # A page with no playable formats should not fail the download; the
        # worker marks the record NOT_READY and reschedules it.
        "ignore_no_formats_error": True,
        # Cluster egress IPs are commonly bot-checked on the default web client
        # but accepted on mobile or embedded clients: visionos first, then
        # web_safari, tv, mweb, web_embedded.
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


def _audio_id_from_url(url: str) -> str:
    """Return the YouTube video id from a watch or short URL.

    Handles ``watch?v=ID`` and the ``youtu.be/ID`` short form. Returns an empty
    string when neither shape yields an id, so the caller can skip the file
    scan instead of globbing for everything.
    """
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("v")
    if values:
        return values[0]
    path = parsed.path.rstrip("/")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _find_audio_file(dest_dir: Path, video_id: str) -> Path | None:
    """Return the first downloaded audio file for ``video_id``, or None."""
    if not video_id:
        return None
    for ext in _AUDIO_EXTS:
        matches = sorted(dest_dir.glob(f"{video_id}*.{ext}"))
        if matches:
            return matches[0]
    return None


def download_audio(
    url: str,
    *,
    opts: dict[str, object],
    ydl_class: Callable[..., YoutubeDLProtocol] | None = None,
    dest_dir: Path,
) -> DownloadResult:
    """Blocking audio-only download. Call via ``asyncio.to_thread()``.

    Uses ``ydl_class`` if given, else ``yt_dlp.YoutubeDL``. ``extract_info``
    runs with ``download=True`` so yt-dlp fetches the chosen audio stream and
    writes it under the outtmpl directory.

    Any yt-dlp error is classified through ``classify_error``
    into a retryable or permanent state. When the call returns without an error
    but no file landed on disk, the result is NOT_READY so the worker retries.
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

    try:
        with factory(opts) as ydl:
            ydl.extract_info(url, download=True)
    except YoutubeDLError as exc:
        # YoutubeDLError is the common base of DownloadError and ExtractorError,
        # so both the download layer and the extractor layer are covered.
        state, detail = classify_error(str(exc))
        return DownloadResult(state=state, path=None, detail=detail)

    video_id = _audio_id_from_url(url)
    audio = _find_audio_file(dest_dir, video_id)
    if audio is not None:
        return DownloadResult(state=ProbeState.READY, path=str(audio), detail=None)
    return DownloadResult(state=ProbeState.NOT_READY, path=None, detail=None)


class CompressError(RuntimeError):
    """Raised when the ffmpeg compress step cannot produce its output."""


def compress_audio(
    src: Path,
    dst: Path,
    *,
    bitrate: str = "32k",
    timeout_s: float = 900.0,
) -> Path:
    """Compress ``src`` to a mono 16 kHz Opus webm at ``dst``.

    One ffmpeg pass downmixes to mono, resamples to 16 kHz, and encodes Opus in
    the ``voip`` application profile, which is tuned for speech and keeps the
    file small. The argv is always a list and ``shell`` is never used, so no
    path or bitrate value can be interpreted by a shell.

    Raises ``CompressError`` when ffmpeg is missing, when it times out, or when
    it exits nonzero; the last 500 characters of stderr are included so the
    failure is debuggable without a full log.
    """
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-application",
        "voip",
        "-f",
        "webm",
        str(dst),
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CompressError("ffmpeg not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CompressError(f"ffmpeg timed out after {timeout_s}s") from exc
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-500:].strip()
        raise CompressError(f"ffmpeg exit {completed.returncode}: {stderr_tail}")
    return dst


# TODO(chunking): videos longer than roughly 100 minutes should be split before
# compression so a single ffmpeg pass cannot exceed the worker's timeout and so
# the transcription service receives chunks it can process in parallel. The
# ffmpeg segment muxer is the obvious tool, but it only writes a segment when
# the *next* keyframe arrives, so each segment boundary drifts and the final
# segment can be much shorter or longer than the target. Fixing that means
# forcing keyframes at exact intervals (or splitting on silence with
# silencedetect) and then remuxing, which is more than this step needs today.

