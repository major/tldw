"""Tests for the ffmpeg audio compression step.

``compress_audio`` shells out to ffmpeg. Every test monkeypatches
``tldw.audio.subprocess.run`` so no test ever invokes the real binary. The fake
records the argv and kwargs it was called with, which lets the tests assert the
exact command line and that ``shell=True`` is never used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tldw.audio import CompressError, compress_audio

# The exact ffmpeg command line compress_audio must build. Kept as a builder so
# the bitrate can be parametrized without duplicating the rest of the argv.
def _expected_argv(src: Path, dst: Path, bitrate: str) -> list[str]:
    """Return the argv compress_audio is expected to pass to subprocess.run."""
    return [
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


class _RecordingRun:
    """Fake ``subprocess.run`` that records its call and returns/raises canned."""

    def __init__(
        self,
        *,
        result: subprocess.CompletedProcess[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        self.calls.append((argv, kwargs))
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._result
        return subprocess.CompletedProcess(argv, 0, "", "")


def _install(monkeypatch: pytest.MonkeyPatch, run: _RecordingRun) -> None:
    """Point tldw.audio.subprocess.run at the recording fake."""
    monkeypatch.setattr("tldw.audio.subprocess.run", run)


# ---------------------------------------------------------------------------
# argv and call shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bitrate", ["32k", "48k", "24k"])
def test_compress_audio_builds_correct_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bitrate: str
) -> None:
    """The ffmpeg argv is built exactly, with the requested bitrate."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun()
    _install(monkeypatch, run)

    # Act
    compress_audio(src, dst, bitrate=bitrate)

    # Assert
    argv, _ = run.calls[0]
    assert argv == _expected_argv(src, dst, bitrate)


def test_compress_audio_passes_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's timeout reaches subprocess.run unchanged."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun()
    _install(monkeypatch, run)

    # Act
    compress_audio(src, dst, timeout_s=123.0)

    # Assert
    _, kwargs = run.calls[0]
    assert kwargs["timeout"] == 123.0


def test_compress_audio_never_uses_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The argv list is passed directly, never through a shell."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun()
    _install(monkeypatch, run)

    # Act
    compress_audio(src, dst)

    # Assert
    argv, kwargs = run.calls[0]
    assert isinstance(argv, list)
    assert kwargs.get("shell") is not True


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------


def test_compress_audio_returns_dst_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero exit returns the destination path."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun(result=subprocess.CompletedProcess([], 0, "", ""))
    _install(monkeypatch, run)

    # Act
    result = compress_audio(src, dst)

    # Assert
    assert result == dst


def test_compress_audio_creates_parent_dir_of_dst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing parent directory for dst is created before ffmpeg runs."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "nested" / "deep" / "out.webm"
    run = _RecordingRun()
    _install(monkeypatch, run)

    # Act
    compress_audio(src, dst)

    # Assert
    assert dst.parent.is_dir()


def test_compress_audio_raises_compress_error_on_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ffmpeg binary becomes a CompressError with a clear message."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun(error=FileNotFoundError("ffmpeg"))
    _install(monkeypatch, run)

    # Act / Assert
    with pytest.raises(CompressError) as excinfo:
        compress_audio(src, dst)
    assert "ffmpeg not found" in str(excinfo.value)


def test_compress_audio_raises_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonzero exit becomes a CompressError carrying the stderr tail."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun(result=subprocess.CompletedProcess([], 1, "", "boom"))
    _install(monkeypatch, run)

    # Act / Assert
    with pytest.raises(CompressError) as excinfo:
        compress_audio(src, dst)
    message = str(excinfo.value)
    assert "ffmpeg exit 1" in message
    assert "boom" in message


def test_compress_audio_raises_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout becomes a CompressError naming the timeout value."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    run = _RecordingRun(error=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=900))
    _install(monkeypatch, run)

    # Act / Assert
    with pytest.raises(CompressError) as excinfo:
        compress_audio(src, dst, timeout_s=900)
    assert "timed out after 900s" in str(excinfo.value)


def test_compress_audio_truncates_long_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the last 500 stderr characters are kept in the error message."""
    # Arrange
    src = tmp_path / "in.webm"
    dst = tmp_path / "out.webm"
    long_stderr = "x" * 5000
    run = _RecordingRun(result=subprocess.CompletedProcess([], 1, "", long_stderr))
    _install(monkeypatch, run)

    # Act / Assert
    with pytest.raises(CompressError) as excinfo:
        compress_audio(src, dst)
    # The message is the prefix plus at most the 500-char stderr tail.
    assert len(str(excinfo.value)) <= 520
