"""Offline tests for bounded transcription audio preparation."""

import asyncio
from pathlib import Path

import pytest

import tldw.transcription_audio as audio


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Create a nonempty source file."""
    path = tmp_path / "source.webm"
    path.write_bytes(b"audio")
    return path


async def test_short_audio_is_passed_through(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small short file is yielded unchanged without running ffmpeg."""

    async def probe(_path: Path) -> float:
        """Return a duration within tolerance."""
        return audio.CHUNK_SECONDS + audio.DURATION_TOLERANCE_SECONDS

    async def run(*_args: str) -> bytes:
        """Reject an unnecessary ffmpeg invocation."""
        pytest.fail("unexpected ffmpeg")

    monkeypatch.setattr(audio, "_probe", probe)
    monkeypatch.setattr(audio, "_run", run)
    async with audio.audio_chunks(source) as chunks:
        assert chunks == [source]
    assert source.exists()


@pytest.mark.parametrize(
    "duration", [0, -1, float("nan"), float("inf"), 128 * 1200 + 1]
)
async def test_invalid_source_duration_is_rejected(
    source: Path, monkeypatch: pytest.MonkeyPatch, duration: float
) -> None:
    """Invalid or unbounded source durations never reach the caller."""

    async def probe(_path: Path) -> float:
        """Return the invalid source duration."""
        return duration

    monkeypatch.setattr(audio, "_probe", probe)
    with pytest.raises(audio.AudioPreparationError):
        async with audio.audio_chunks(source):
            pytest.fail("invalid source yielded")
    assert source.exists()


@pytest.mark.parametrize("missing", [False, True])
async def test_empty_or_missing_source_is_rejected(source: Path, missing: bool) -> None:
    """A source must exist and contain bytes."""
    source.unlink() if missing else source.write_bytes(b"")
    with pytest.raises(audio.AudioPreparationError):
        async with audio.audio_chunks(source):
            pytest.fail("invalid source yielded")


@pytest.mark.parametrize("invalid", [False, True])
async def test_segments_are_sorted_validated_and_removed(
    source: Path, monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    """Validate every segment before yield and clean the temporary directory."""
    generated: list[Path] = []
    commands: list[tuple[str, ...]] = []

    async def probe(path: Path) -> float:
        """Make the source long and optionally reject the last segment."""
        if path == source:
            return 1300
        return 1202 if invalid and path.name == "segment-0001.webm" else 600

    async def run(*args: str) -> bytes:
        """Create segments in reverse order without running ffmpeg."""
        commands.append(args)
        pattern = Path(args[-1])
        for name in ("segment-0001.webm", "segment-0000.webm"):
            path = pattern.parent / name
            path.write_bytes(b"chunk")
            generated.append(path)
        return b""

    monkeypatch.setattr(audio, "_probe", probe)
    monkeypatch.setattr(audio, "_run", run)
    if invalid:
        with pytest.raises(audio.AudioPreparationError):
            async with audio.audio_chunks(source):
                pytest.fail("invalid segments yielded")
    else:
        async with audio.audio_chunks(source) as chunks:
            assert chunks == sorted(generated)
            assert all(path.exists() for path in chunks)
    assert (
        source.exists() and generated and not any(path.exists() for path in generated)
    )
    assert commands and all(
        value in commands[0]
        for value in (
            "-nostdin",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            "segment",
            "1200",
            "-reset_timestamps",
        )
    )


async def test_subprocess_cancellation_kills_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation reaps the media subprocess before propagating."""
    started = asyncio.Event()

    class Process:
        """A process that waits until cancelled."""

        killed = False
        waited = False

        async def communicate(self):
            """Wait for cancellation after announcing startup."""
            started.set()
            await asyncio.Event().wait()

        def kill(self):
            """Record termination."""
            self.killed = True

        async def wait(self):
            """Record reaping."""
            self.waited = True

    process = Process()

    async def spawn(*args, **kwargs):
        """Check safe subprocess streams and return the fake."""
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(audio._run("ffmpeg"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed and process.waited


@pytest.mark.parametrize("mode", ["none", "empty", "oversized", "too_many", "failure"])
async def test_bad_segments_never_escape(
    source: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Missing, invalid, excessive, or failed output is rejected safely."""
    generated: list[Path] = []

    async def probe(path: Path) -> float:
        """Force splitting and give chunks a valid duration."""
        return 1300 if path == source else 100

    async def run(*args: str) -> bytes:
        """Simulate bad ffmpeg output or a failed subprocess."""
        if mode == "failure":
            raise audio.AudioPreparationError("secret stderr")
        directory = Path(args[-1]).parent
        count = 129 if mode == "too_many" else (0 if mode == "none" else 1)
        for i in range(count):
            path = directory / f"segment-{i:04d}.webm"
            path.write_bytes(b"" if mode == "empty" else b"x")
            if mode == "oversized":
                with path.open("r+b") as stream:
                    stream.truncate(audio.MAX_UPLOAD_BYTES + 1)
            generated.append(path)
        return b""

    monkeypatch.setattr(audio, "_probe", probe)
    monkeypatch.setattr(audio, "_run", run)
    with pytest.raises(audio.AudioPreparationError) as caught:
        async with audio.audio_chunks(source):
            pytest.fail("invalid output yielded")
    assert "secret stderr" not in str(caught.value)
    assert source.exists() and not any(path.exists() for path in generated)


@pytest.mark.parametrize(
    "output",
    [
        b"bad",
        b"{}",
        b'{"format":{"duration":"N/A"}}',
        b'{"format":{"duration":"nan"}}',
        b'{"format":{"duration":null}}',
    ],
)
async def test_bad_probe_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, output: bytes
) -> None:
    """Malformed or nonfinite probe results never pass validation."""

    # Arrange
    async def run(*_args: str) -> bytes:
        """Return invalid ffprobe output."""
        return output

    monkeypatch.setattr(audio, "_run", run)
    # Act / Assert
    with pytest.raises(audio.AudioPreparationError):
        await audio._probe(Path("source.webm"))


async def test_source_size_cap_rejects_before_probe(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sparse oversized source is rejected before media tools run."""
    # Arrange
    with source.open("r+b") as stream:
        stream.truncate(audio.MAX_CHUNKS * audio.MAX_UPLOAD_BYTES + 1)

    async def probe(_path: Path) -> float:
        """Reject unexpected probe calls."""
        pytest.fail("unexpected probe")

    monkeypatch.setattr(audio, "_probe", probe)
    # Act / Assert
    with pytest.raises(audio.AudioPreparationError):
        async with audio.audio_chunks(source):
            pytest.fail("oversized source yielded")


async def test_failed_process_cleans_partial_segments(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonzero ffmpeg exit removes partial output and keeps the source."""
    # Arrange
    generated: list[Path] = []

    async def probe(_path: Path) -> float:
        """Force segmentation."""
        return 1300

    class Process:
        """Return a failed process result after writing a segment."""

        returncode = 1

        async def communicate(self) -> tuple[bytes, None]:
            """Return the failed subprocess output."""
            return b"secret", None

    async def spawn(*args: str, **_kwargs: object) -> Process:
        """Write a partial output before the subprocess fails."""
        path = Path(args[-1]).parent / "segment-0000.webm"
        path.write_bytes(b"partial")
        generated.append(path)
        return Process()

    monkeypatch.setattr(audio, "_probe", probe)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    # Act / Assert
    with pytest.raises(audio.AudioPreparationError, match="Audio preparation failed"):
        async with audio.audio_chunks(source):
            pytest.fail("failed output yielded")
    assert source.exists() and generated and not generated[0].exists()
