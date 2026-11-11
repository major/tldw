"""Prepare bounded, losslessly segmented audio for transcription."""

import asyncio
import json
import math
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

CHUNK_SECONDS = 1200
DURATION_TOLERANCE_SECONDS = 1
MAX_UPLOAD_BYTES = 24_000_000
MAX_CHUNKS = 128
PREPARATION_REASONS = {
    "preparation_failed": "Audio preparation failed.",
    "invalid_duration": "Audio duration is unavailable or invalid.",
    "preparation_limit": "Audio exceeds local preparation limits.",
    "incomplete_audio": "Prepared chunks do not cover the complete audio.",
    "invalid_preparation": "Audio preparation validation failed.",
}


class AudioPreparationError(RuntimeError):
    """Audio could not be safely prepared for transcription."""

    def __init__(self, reason: str = "preparation_failed", *, permanent: bool = False):
        """Store a safe reason and whether retrying can help."""
        self.reason = reason if reason in PREPARATION_REASONS else "preparation_failed"
        self.permanent = permanent
        super().__init__(PREPARATION_REASONS[self.reason])


async def _run(*args: str) -> bytes:
    """Run a media tool without exposing its errors or leaving it running."""
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await process.communicate()
        except asyncio.CancelledError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        if process.returncode != 0:
            raise AudioPreparationError()
        return stdout
    except OSError as exc:
        raise AudioPreparationError() from exc


async def _probe(path: Path) -> float:
    """Read the duration reported by ffprobe without exposing its output."""
    try:
        output = await _run(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        )
        duration = float(json.loads(output)["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
        return duration
    except (ValueError, TypeError, KeyError) as exc:
        raise AudioPreparationError("invalid_duration", permanent=True) from exc


async def _validate(
    path: Path, max_bytes: int, max_duration: float
) -> tuple[int, float]:
    """Validate actual file size and a finite positive measured duration."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AudioPreparationError() from exc
    if size <= 0:
        raise AudioPreparationError("invalid_preparation", permanent=True)
    if size > max_bytes:
        raise AudioPreparationError("preparation_limit", permanent=True)
    duration = await _probe(path)
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise AudioPreparationError("invalid_duration", permanent=True)
    if duration > max_duration:
        raise AudioPreparationError("preparation_limit", permanent=True)
    return size, duration


@asynccontextmanager
async def audio_chunks(audio_path: Path) -> AsyncGenerator[list[Path]]:
    """Yield validated source or temporary stream-copy segments, then clean up."""
    size, duration = await _validate(
        audio_path,
        MAX_CHUNKS * MAX_UPLOAD_BYTES,
        MAX_CHUNKS * CHUNK_SECONDS,
    )
    if (
        size <= MAX_UPLOAD_BYTES
        and duration <= CHUNK_SECONDS + DURATION_TOLERANCE_SECONDS
    ):
        yield [audio_path]
        return
    segment_seconds = min(
        CHUNK_SECONDS, max(1, math.floor(duration * MAX_UPLOAD_BYTES / (size * 2)))
    )
    if math.ceil(duration / segment_seconds) > MAX_CHUNKS:
        raise AudioPreparationError("preparation_limit", permanent=True)
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "segment-%04d.webm"
            await _run(
                "ffmpeg",
                "-nostdin",
                "-i",
                str(audio_path),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                str(segment_seconds),
                "-reset_timestamps",
                "1",
                str(output),
            )
            chunks = sorted(Path(directory).glob("segment-*.webm"))
            if not chunks:
                raise AudioPreparationError("invalid_preparation", permanent=True)
            if len(chunks) > MAX_CHUNKS:
                raise AudioPreparationError("preparation_limit", permanent=True)
            total_duration = 0.0
            for chunk in chunks:
                _, chunk_duration = await _validate(
                    chunk, MAX_UPLOAD_BYTES, CHUNK_SECONDS + DURATION_TOLERANCE_SECONDS
                )
                total_duration += chunk_duration
            if abs(total_duration - duration) > len(chunks) * DURATION_TOLERANCE_SECONDS:
                raise AudioPreparationError("incomplete_audio", permanent=True)
            yield chunks
    except OSError as exc:
        raise AudioPreparationError() from exc
