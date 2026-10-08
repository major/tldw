"""Tests for the tldw yt-dlp transcript probe.

The pure helpers (``classify_error``, ``parse_vtt``, ``first_lines``, and
``build_ydl_opts``) are tested directly. ``probe_and_fetch`` is tested with an
injected ``FakeYoutubeDL`` so no test touches the network, the real yt-dlp
library, or a live YouTube request. The fake can return a canned info dict,
raise a canned error, and drop pre-written files into the output directory so
the READY/NOT_READY file checks are exercised end to end.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

from tldw.transcript import (
    ProbeState,
    build_ydl_opts,
    classify_error,
    first_lines,
    parse_vtt,
    probe_and_fetch,
)


class FakeYoutubeDL:
    """Test double for yt_dlp.YoutubeDL. Configurable per test."""

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
        # Write any pre-configured files to the outtmpl directory so the probe
        # sees them exactly where yt-dlp would have put them.
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
    """Build a ydl_class the probe can call with just the opts dict."""
    return partial(FakeYoutubeDL, info=info, error=error, write_files=write_files)


# ---------------------------------------------------------------------------
# parse_vtt
# ---------------------------------------------------------------------------


def test_parse_vtt_strips_header_blocks() -> None:
    """WEBVTT, NOTE, Kind, and Language blocks never reach the output."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n"
        "\n"
        "NOTE this is a note\n"
        "\n"
        "00:00:01.000 --> 00:00:04.500\n"
        "Hello world\n"
    )

    # Act
    result = parse_vtt(vtt)

    # Assert
    assert result == ["Hello world"]
    assert not any("WEBVTT" in line for line in result)
    assert not any("Kind:" in line for line in result)
    assert not any("Language:" in line for line in result)
    assert not any("NOTE" in line for line in result)


def test_parse_vtt_drops_timestamp_only_lines() -> None:
    """A cue with no text is dropped, and no timing line survives."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:04.500 align:start position:0%\n"
        "\n"
        "00:00:05.000 --> 00:00:08.000\n"
        "Actual line\n"
    )

    # Act
    result = parse_vtt(vtt)

    # Assert
    assert result == ["Actual line"]
    assert not any("-->" in line for line in result)


def test_parse_vtt_strips_inline_tags() -> None:
    """Colour and timing tags are markup and are removed."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:04.500\n"
        "<c.colorCCCCCC>Hello</c>\n"
    )

    # Act
    result = parse_vtt(vtt)

    # Assert
    assert result == ["Hello"]


def test_parse_vtt_collapses_rolling_duplicates() -> None:
    """The same line across successive cues is kept only once."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Repeated\n"
        "\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "Repeated\n"
        "\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "Repeated\n"
    )

    # Act
    result = parse_vtt(vtt)

    # Assert
    assert result == ["Repeated"]


def test_parse_vtt_keeps_non_consecutive_duplicates() -> None:
    """A repeated line separated by other text is not collapsed."""
    # Arrange
    vtt = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Same\n"
        "\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "Different\n"
        "\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "Same\n"
    )

    # Act
    result = parse_vtt(vtt)

    # Assert
    assert result == ["Same", "Different", "Same"]


# ---------------------------------------------------------------------------
# first_lines
# ---------------------------------------------------------------------------


def test_first_lines_returns_first_n_non_empty_lines() -> None:
    """Empty and whitespace-only lines are skipped before slicing."""
    # Arrange
    text = "\n".join(["", "   ", *[f"line {i}" for i in range(1, 16)]])

    # Act
    result = first_lines(text, 10)

    # Assert
    assert result == [f"line {i}" for i in range(1, 11)]


def test_first_lines_strips_each_line() -> None:
    """Surrounding whitespace is removed from every returned line."""
    # Arrange
    text = "  alpha  \n\tbeta\t\n"

    # Act
    result = first_lines(text, 10)

    # Assert
    assert result == ["alpha", "beta"]


def test_first_lines_rejects_negative_n() -> None:
    """A negative count is a caller bug, not an empty result."""
    # Arrange
    text = "alpha\nbeta\n"

    # Act / Assert
    with pytest.raises(ValueError):
        first_lines(text, -1)


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "HTTP Error 429: Too Many Requests",
        "Too Many Requests",
        "RequestBlocked: the request was blocked",
    ],
)
def test_classify_error_429_variants(message: str) -> None:
    """Every 429 spelling maps to the transient RATE_LIMITED state."""
    # Arrange
    # Each parametrized message is one live yt-dlp 429 spelling.

    # Act
    state, _ = classify_error(message)

    # Assert
    assert state == ProbeState.RATE_LIMITED


@pytest.mark.parametrize(
    "message",
    [
        "This video is unavailable",
        "Private video",
        "Sign in to confirm you're not a bot",
        "Video removed by uploader",
    ],
)
def test_classify_error_unavailable_variants(message: str) -> None:
    """Permanent errors map to UNAVAILABLE."""
    # Arrange
    # Each parametrized message is one live yt-dlp permanent-error spelling.

    # Act
    state, _ = classify_error(message)

    # Assert
    assert state == ProbeState.UNAVAILABLE


def test_classify_error_po_token() -> None:
    """A missing PO token is retryable and carries its own detail."""
    # Arrange
    message = "PO Token was not provided"

    # Act
    state, detail = classify_error(message)

    # Assert
    assert state == ProbeState.NOT_READY
    assert detail == "po_token"


def test_classify_error_unknown_defaults_to_not_ready() -> None:
    """An unseen message is retryable and keeps the raw text in the detail."""
    # Arrange
    message = "Some new error we have never seen"

    # Act
    state, detail = classify_error(message)

    # Assert
    assert state == ProbeState.NOT_READY
    assert detail.startswith("unknown:")


# ---------------------------------------------------------------------------
# build_ydl_opts
# ---------------------------------------------------------------------------


def test_build_ydl_opts_uses_exact_lang_codes(tmp_path: Path) -> None:
    """The default langs stay exact; the en.* pattern caused 429 storms."""
    # Arrange
    # tmp_path stands in for the subtitle destination.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["subtitleslangs"] == ["en", "en-orig"]


def test_build_ydl_opts_without_cookies_omits_cookiefile(tmp_path: Path) -> None:
    """No cookies file means no cookiefile key at all."""
    # Arrange
    # cookies_file defaults to None.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert "cookiefile" not in opts


def test_build_ydl_opts_with_cookies_passes_cookiefile(tmp_path: Path) -> None:
    """A cookies file is passed through as a string path."""
    # Arrange
    cookies = tmp_path / "cookies.txt"

    # Act
    opts = build_ydl_opts(tmp_path, cookies_file=cookies)

    # Assert
    assert opts["cookiefile"] == str(cookies)


def test_build_ydl_opts_sets_skip_download_and_subs(tmp_path: Path) -> None:
    """The probe only wants subtitles, never the media file."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["skip_download"] is True
    assert opts["writesubtitles"] is True
    assert opts["writeautomaticsub"] is True


def test_build_ydl_opts_outtmpl_uses_dest_dir(tmp_path: Path) -> None:
    """The output template lives inside the destination directory."""
    # Arrange
    # tmp_path is the destination.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert str(opts["outtmpl"]).startswith(str(tmp_path))


def test_build_ydl_opts_accepts_custom_langs(tmp_path: Path) -> None:
    """A custom lang tuple replaces the default list."""
    # Arrange
    langs = ("de",)

    # Act
    opts = build_ydl_opts(tmp_path, langs=langs)

    # Assert
    assert opts["subtitleslangs"] == ["de"]


def test_build_ydl_opts_sets_retries_to_three(tmp_path: Path) -> None:
    """Retries are lowered from the yt-dlp default of 10 to fail fast."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["retries"] == 3


def test_build_ydl_opts_sets_socket_timeout(tmp_path: Path) -> None:
    """A socket timeout bounds a hung connection so a probe fails fast."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["socket_timeout"] == 30


def test_build_ydl_opts_ignores_no_formats_error(tmp_path: Path) -> None:
    """A page with no playable formats does not fail the probe."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["ignore_no_formats_error"] is True


def test_build_ydl_opts_uses_fallback_player_clients(tmp_path: Path) -> None:
    """The visionos web client runs first, then 2026 fallback clients."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["extractor_args"] == {
        "youtube": {
            "player_client": [
                "visionos",
                "web_safari",
                "tv",
                "mweb",
                "web_embedded",
            ]
        }
    }


def test_build_ydl_opts_new_opts_apply_with_cookies(tmp_path: Path) -> None:
    """The cookies block does not shadow the four new resilience options."""
    # Arrange
    cookies = tmp_path / "cookies.txt"

    # Act
    opts = build_ydl_opts(tmp_path, cookies_file=cookies)

    # Assert
    assert opts["retries"] == 3
    assert opts["socket_timeout"] == 30
    assert opts["ignore_no_formats_error"] is True
    assert opts["extractor_args"] == {
        "youtube": {
            "player_client": [
                "visionos",
                "web_safari",
                "tv",
                "mweb",
                "web_embedded",
            ]
        }
    }
    assert opts["cookiefile"] == str(cookies)


def test_build_ydl_opts_new_opts_apply_with_langs(tmp_path: Path) -> None:
    """A custom lang tuple does not drop the four new resilience options."""
    # Arrange
    langs = ("en",)

    # Act
    opts = build_ydl_opts(tmp_path, langs=langs)

    # Assert
    assert opts["retries"] == 3
    assert opts["socket_timeout"] == 30
    assert opts["ignore_no_formats_error"] is True
    assert opts["extractor_args"] == {
        "youtube": {
            "player_client": [
                "visionos",
                "web_safari",
                "tv",
                "mweb",
                "web_embedded",
            ]
        }
    }


def test_build_ydl_opts_preserves_existing_keys(tmp_path: Path) -> None:
    """The pre-existing probe options are still present after the change."""
    # Arrange
    # Default opts.

    # Act
    opts = build_ydl_opts(tmp_path)

    # Assert
    assert opts["skip_download"] is True
    assert opts["writesubtitles"] is True
    assert opts["writeautomaticsub"] is True
    assert opts["subtitleslangs"] == ["en", "en-orig"]
    assert str(opts["outtmpl"]).startswith(str(tmp_path))
    assert opts["quiet"] is True
    assert opts["noprogress"] is True
    assert opts["ignoreerrors"] is False


# ---------------------------------------------------------------------------
# probe_and_fetch
# ---------------------------------------------------------------------------

_VIDEO_ID = "dQw4w9WgXcQ"
_WATCH_URL = f"https://www.youtube.com/watch?v={_VIDEO_ID}"
_SHORT_URL = f"https://youtu.be/{_VIDEO_ID}"
_REQUESTED: dict[str, object] = {"requested_subtitles": {"en": {"url": "https://example/sub"}}}


@pytest.mark.parametrize("url", [_WATCH_URL, _SHORT_URL])
def test_probe_and_fetch_ready_when_file_and_requested_subs(
    tmp_path: Path, url: str
) -> None:
    """A fetched file plus requested_subtitles is READY for both URL forms."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(
        info=_REQUESTED,
        write_files=((f"{_VIDEO_ID}.en.vtt", "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n"),),
    )

    # Act
    result = probe_and_fetch(url, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.READY
    assert result.transcript_path is not None
    assert result.transcript_path.endswith(f"{_VIDEO_ID}.en.vtt")


def test_probe_and_fetch_not_ready_when_no_requested_subs_and_no_file(
    tmp_path: Path,
) -> None:
    """No subs and no file is the retryable NOT_READY state."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(info={})

    # Act
    result = probe_and_fetch(_WATCH_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.NOT_READY
    assert result.transcript_path is None


def test_probe_and_fetch_unavailable_on_unavailable_error(tmp_path: Path) -> None:
    """A permanent error becomes UNAVAILABLE with a readable detail."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(error=DownloadError("This video is unavailable"))

    # Act
    result = probe_and_fetch(_WATCH_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.UNAVAILABLE
    assert result.detail is not None
    assert "unavailable" in result.detail


def test_probe_and_fetch_rate_limited_on_429_error(tmp_path: Path) -> None:
    """A 429 becomes the transient RATE_LIMITED state."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(error=DownloadError("HTTP Error 429: Too Many Requests"))

    # Act
    result = probe_and_fetch(_WATCH_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.RATE_LIMITED


def test_probe_and_fetch_po_token_returns_not_ready(tmp_path: Path) -> None:
    """A missing PO token becomes retryable NOT_READY."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(error=DownloadError("PO Token was not provided"))

    # Act
    result = probe_and_fetch(_WATCH_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.NOT_READY
    assert result.detail == "po_token"


def test_probe_and_fetch_creates_dest_dir_if_missing(tmp_path: Path) -> None:
    """A missing destination directory is created before the probe runs."""
    # Arrange
    dest = tmp_path / "does" / "not" / "exist"
    opts = build_ydl_opts(dest)

    # Act
    result = probe_and_fetch(
        _WATCH_URL, opts=opts, ydl_class=_fake(info={}), dest_dir=dest
    )

    # Assert
    assert dest.is_dir()
    assert result.state == ProbeState.NOT_READY


def test_probe_and_fetch_treats_file_without_requested_subs_as_not_ready(
    tmp_path: Path,
) -> None:
    """A stray file with no requested_subtitles does not mark READY."""
    # Arrange
    dest = tmp_path / "subs"
    opts = build_ydl_opts(dest)
    ydl = _fake(
        info={},
        write_files=((f"{_VIDEO_ID}.en.vtt", "WEBVTT\n"),),
    )

    # Act
    result = probe_and_fetch(_WATCH_URL, opts=opts, ydl_class=ydl, dest_dir=dest)

    # Assert
    assert result.state == ProbeState.NOT_READY
    assert result.transcript_path is None
