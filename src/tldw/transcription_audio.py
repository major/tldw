"""Prepare bounded, losslessly segmented audio for transcription."""

import asyncio
import json
import math
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncGenerator

CHUNK_SECONDS = 1200
DURATION_TOLERANCE_SECONDS = 1
MAX_UPLOAD_BYTES = 24_000_000
MAX_CHUNKS = 128


class AudioPreparationError(RuntimeError):
    """Audio could not be safely prepared for transcription."""


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
            raise AudioPreparationError("Audio preparation failed.")
        return stdout
    except OSError as exc:
        raise AudioPreparationError("Audio preparation failed.") from exc


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
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise AudioPreparationError("Audio preparation failed.") from exc


async def _validate(
    path: Path, max_bytes: int, max_duration: float
) -> tuple[int, float]:
    """Validate actual file size and a finite positive measured duration."""
    try:
        size = path.stat().st_size
        if not 0 < size <= max_bytes:
            raise ValueError
        duration = await _probe(path)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ValueError
        if not 0 < duration <= max_duration:
            raise ValueError
        return size, duration
    except (OSError, ValueError, TypeError, AudioPreparationError) as exc:
        raise AudioPreparationError("Audio preparation failed.") from exc


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
                str(CHUNK_SECONDS),
                "-reset_timestamps",
                "1",
                str(output),
            )
            chunks = sorted(Path(directory).glob("segment-*.webm"))
            if not chunks or len(chunks) > MAX_CHUNKS:
                raise AudioPreparationError("Audio preparation failed.")
            for chunk in chunks:
                await _validate(
                    chunk, MAX_UPLOAD_BYTES, CHUNK_SECONDS + DURATION_TOLERANCE_SECONDS
                )
            yield chunks
    except (OSError, AudioPreparationError) as exc:
        raise AudioPreparationError("Audio preparation failed.") from exc
