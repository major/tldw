"""Tests for the audio-only yt-dlp download.

``build_audio_ydl_opts`` is a pure function and is tested directly.
``download_audio`` is tested with an injected ``FakeYoutubeDL`` so no test
touches the network, the real yt-dlp library, or a live YouTube request. The
fake can return a canned info dict, raise a canned error, and drop a
pre-written media file into the output directory so the READY/NOT_READY file
scan is exercised end to end.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

from tldw.audio import (
    DownloadResult,
    build_audio_ydl_opts,
    download_audio,
)
from tldw.transcript import ProbeState


class FakeYoutubeDL:
    """Test double for yt_dlp.YoutubeDL. Configurable per test.

    Mirrors the fake in ``tests/test_transcript.py``. ``write_files`` drops
    files into the outtmpl directory so the download scan finds them exactly
    where yt-dlp would have written them.
    """

    def __init__(
        self,
        opts: dict[str, object],
        *,
        info: dict[str, object] | None = None,
        error: Exception | None = None,
        write_files: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.opts = opts
        self._info = info
        self._error = error
        self._write_files = write_files

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def extract_info(self, url: str, download: bool = True) -> dict[str, object]:
        # Write any pre-configured files to the outtmpl directory so the
        # download sees them exactly where yt-dlp would have put them.
        if self._write_files:
            dest = Path(str(self.opts["outtmpl"])).parent
            dest.mkdir(parents=True, exist_ok=True)
            for name, content in self._write_files:
                (dest / name).write_text(content, encoding="utf-8")
        if self._error is not None:
            raise self._error
        return self._info or {}


def _fake(
    *,
    info: dict[str, object] | None = None,
    error: Exception | None = None,
    write_files: tuple[tuple[str, str], ...] = (),
) -> partial[FakeYoutubeDL]:
    """Build a ydl_class the download can call with just the opts dict."""
    return partial(FakeYoutubeDL, info=info, error=error, write_files=write_files)


_VIDEO_ID = "abc123"
_SHORT_URL = f"https://youtu.be/{_VIDEO_ID}"


# ---------------------------------------------------------------------------
# build_audio_ydl_opts
# ---------------------------------------------------------------------------


def test_build_audio_ydl_opts_default_format_chain(tmp_path: Path) -> None:
    """The format chain prefers webm/m4a audio-only, smallest first."""
    # Arrange
    # tmp_path stands in for the download destination.

    # Act
    opts = build_audio_ydl_opts(tmp_path)

    # Assert
    assert opts["format"] == "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio"
    assert opts["format_sort"] == ["+abr", "+size"]


def test_build_audio_ydl_opts_uses_2026_player_clients(tmp_path: Path) -> None:
    """The client ladder matches the VTT fix; tv_embedded is gone in 2026."""
    # Arrange
    # Default opts.

    # Act
    opts = build_audio_ydl_opts(tmp_path)

    # Assert
    extractor_args = opts["extractor_args"]
    assert isinstance(extractor_args, dict)
    youtube = extractor_args["youtube"]
    assert isinstance(youtube, dict)
    assert youtube["player_client"] == [
        "visionos",
        "web_safari",
        "tv",
        "mweb",
        "web_embedded",
    ]
    assert "tv_embedded" not in youtube["player_client"]


def test_build_audio_ydl_opts_without_cookies_omits_cookiefile(
    tmp_path: Path,
) -> None:
    """No cookies file means no cookiefile key at all."""
    # Arrange
    # cookies_file defaults to None.

    # Act
    opts = build_audio_ydl_opts(tmp_path)

    # Assert
    assert "cookiefile" not in opts


def test_build_audio_ydl_opts_with_cookies_passes_cookiefile(
    tmp_path: Path,
) -> None:
    """A cookies file is passed through as a string path."""
    # Arrange
    cookies = tmp_path / "cookies.txt"

    # Act
    opts = build_audio_ydl_opts(tmp_path, cookies_file=cookies)

    # Assert
    assert opts["cookiefile"] == str(cookies)


# ---------------------------------------------------------------------------
# download_audio
# ---------------------------------------------------------------------------


def test_download_audio_returns_ready_when_file_written(tmp_path: Path) -> None:
    """A landed media file makes the download READY."""
    # Arrange
    dest = tmp_path / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(
        info={"id": _VIDEO_ID, "ext": "webm"},
        write_files=((f"{_VIDEO_ID}.webm", "binary-ish"),),
    )

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert isinstance(result, DownloadResult)
    assert result.state == ProbeState.READY
    assert result.path is not None
    assert result.path.endswith(f"{_VIDEO_ID}.webm")


def test_download_audio_returns_not_ready_when_no_file(tmp_path: Path) -> None:
    """An info dict without a landed file is retryable NOT_READY."""
    # Arrange
    dest = tmp_path / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(info={"id": _VIDEO_ID, "ext": "webm"})

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.NOT_READY
    assert result.path is None


@pytest.mark.parametrize(
    "message",
    [
        "HTTP Error 429: too many requests",
        "HTTP Error 429: Too Many Requests",
        "Too Many Requests",
    ],
)
def test_download_audio_classifies_429_as_rate_limited(
    tmp_path: Path, message: str
) -> None:
    """A 429 download error becomes the transient RATE_LIMITED state."""
    # Arrange
    dest = tmp_path / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(error=DownloadError(message))

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.RATE_LIMITED
    assert result.path is None


def test_download_audio_classifies_unavailable(tmp_path: Path) -> None:
    """A permanent download error becomes UNAVAILABLE."""
    # Arrange
    dest = tmp_path / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(error=DownloadError("Private video"))

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.UNAVAILABLE
    assert result.path is None


def test_download_audio_classifies_unknown_as_not_ready(tmp_path: Path) -> None:
    """An unseen download error is retryable and keeps the raw detail."""
    # Arrange
    dest = tmp_path / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(error=DownloadError("mystery extractor glitch"))

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.NOT_READY
    assert result.path is None
    assert result.detail is not None
    assert result.detail.startswith("unknown:")


def test_download_audio_creates_dest_dir(tmp_path: Path) -> None:
    """A missing destination directory is created before the download runs."""
    # Arrange
    dest = tmp_path / "does" / "not" / "exist"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(info={})

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert dest.is_dir()
    assert result.state == ProbeState.NOT_READY


def test_download_audio_uses_destination_dir_for_glob(tmp_path: Path) -> None:
    """The landed file is found under the real outtmpl directory."""
    # Arrange
    dest = tmp_path / "nested" / "audio"
    opts = build_audio_ydl_opts(dest)
    ydl = _fake(
        info={"id": _VIDEO_ID, "ext": "m4a"},
        write_files=((f"{_VIDEO_ID}.m4a", "binary-ish"),),
    )

    # Act
    result = download_audio(_SHORT_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.READY
    assert result.path is not None
    assert Path(result.path).parent == dest
    assert Path(result.path).is_file()
