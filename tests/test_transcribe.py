"""Tests for the tldw.transcribe OpenAI gpt-transcribe seam.

Every test monkeypatches the module-level ``AsyncOpenAI`` with a local fake, so
no request ever leaves the process (pytest-socket also blocks real TCP). The
fake records the constructor and ``create`` kwargs for inspection and returns a
canned result whose ``.text`` the production code reads.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

import tldw.transcribe as transcribe


class _FakeResult:
    """Minimal stand-in for the SDK Transcription object."""

    def __init__(self, text: str) -> None:
        """Store the transcript text the production code reads back."""
        self.text = text


class _FakeTranscriptions:
    """Fake ``client.audio.transcriptions`` namespace."""

    def __init__(self, parent: FakeAsyncOpenAI) -> None:
        """Remember the parent fake so calls can be recorded on it."""
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        file: Any,
        languages: list[str],
        response_format: str,
        **kwargs: Any,
    ) -> Any:
        """Record the call and return an awaitable yielding the canned result."""
        self._parent.create_kwargs = {
            "model": model,
            "file": file,
            "languages": languages,
            "response_format": response_format,
            **kwargs,
        }

        async def _coro() -> _FakeResult:
            if self._parent.delay:
                await asyncio.sleep(self._parent.delay)
            if self._parent.raises is not None:
                raise self._parent.raises
            return self._parent.result or _FakeResult("")

        return _coro()


class _FakeAudio:
    """Fake ``client.audio`` namespace exposing ``transcriptions``."""

    def __init__(self, parent: FakeAsyncOpenAI) -> None:
        """Remember the parent fake for the transcriptions namespace."""
        self._parent = parent

    @property
    def transcriptions(self) -> _FakeTranscriptions:
        """Return the fake transcriptions namespace."""
        return _FakeTranscriptions(self._parent)


class FakeAsyncOpenAI:
    """Test double mirroring the ``AsyncOpenAI`` surface transcribe uses."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        result: _FakeResult | None = None,
        raises: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        """Configure canned behavior and record constructor kwargs."""
        self.init_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
        }
        self.create_kwargs: dict[str, Any] = {}
        self.result = result
        self.raises = raises
        self.delay = delay

    async def __aenter__(self) -> FakeAsyncOpenAI:
        """Support ``async with`` the way the real client does."""
        return self

    async def __aexit__(self, *args: object) -> bool:
        """Leave the client context without suppressing exceptions."""
        return False

    @property
    def audio(self) -> _FakeAudio:
        """Return the fake audio namespace."""
        return _FakeAudio(self)


def _install_fake(monkeypatch: pytest.MonkeyPatch, **config: Any) -> FakeAsyncOpenAI:
    """Patch ``tldw.transcribe.AsyncOpenAI`` and return the configured fake.

    The production code calls ``AsyncOpenAI(api_key=..., base_url=...)``. The
    factory records those kwargs on the returned fake before handing it back.
    """
    fake = FakeAsyncOpenAI(**config)

    def _factory(**kwargs: Any) -> FakeAsyncOpenAI:
        fake.init_kwargs = dict(kwargs)
        return fake

    monkeypatch.setattr(transcribe, "AsyncOpenAI", _factory)
    return fake


def _audio_file(tmp_path: Path) -> Path:
    """Write a tiny placeholder audio file and return its path."""
    audio_path = tmp_path / "clip.webm"
    audio_path.write_bytes(b"\x1a\x45\xdf\xa3fake-webm")
    return audio_path


def _kwargs(audio_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build the default transcribe kwargs with optional overrides."""
    values: dict[str, Any] = {
        "model": "gpt-transcribe",
        "langs": ["en"],
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
        "timeout_s": transcribe.DEFAULT_TIMEOUT_SECONDS,
    }
    values.update(overrides)
    return values


def _status_error(status_code: int = 429) -> openai.APIStatusError:
    """Build an ``APIStatusError`` with a real httpx response for the status."""
    request = httpx2.Request("POST", "https://example.test/v1/audio/transcriptions")
    response = httpx2.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def test_transcribe_uses_gpt_transcribe_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model argument reaches the SDK unchanged."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert fake.create_kwargs["model"] == "gpt-transcribe"


def test_transcribe_uses_languages_plural_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plural ``languages`` kwarg is sent; singular ``language`` is not."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert fake.create_kwargs["languages"] == ["en"]
    assert "language" not in fake.create_kwargs


def test_transcribe_uses_json_response_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gpt-transcribe only accepts the json response format."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert fake.create_kwargs["response_format"] == "json"


def test_transcribe_opens_file_as_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK receives a handle opened in binary mode."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert fake.create_kwargs["file"].mode == "rb"


def test_transcribe_closes_file_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file handle is closed once the call returns."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert fake.create_kwargs["file"].closed is True


def test_transcribe_returns_text_from_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plain transcript text is returned to the caller."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    _install_fake(monkeypatch, result=_FakeResult("hello world"))

    # Act
    text = asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))

    # Assert
    assert text == "hello world"


def test_transcribe_uses_provided_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured gateway base URL reaches the client constructor."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(
        transcribe.transcribe(
            audio_path,
            **_kwargs(audio_path, base_url="https://my-gateway.example/v1"),
        )
    )

    # Assert
    assert fake.init_kwargs["base_url"] == "https://my-gateway.example/v1"


def test_transcribe_uses_provided_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured API key reaches the client constructor."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(
        transcribe.transcribe(audio_path, **_kwargs(audio_path, api_key="sk-..."))
    )

    # Assert
    assert fake.init_kwargs["api_key"] == "sk-..."


def test_transcribe_wraps_timeout_in_asyncio_wait_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow SDK call is cut off by the timeout."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    _install_fake(monkeypatch, result=_FakeResult("ok"), delay=0.05)

    # Act / Assert
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            transcribe.transcribe(audio_path, **_kwargs(audio_path, timeout_s=0.01))
        )


def test_transcribe_raises_transcribe_error_on_empty_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty API key fails fast with a clear TranscribeError."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act / Assert
    with pytest.raises(transcribe.TranscribeError, match="api_key is required"):
        asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path, api_key="")))


def test_transcribe_translates_api_status_error_to_transcribe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API status errors become TranscribeError with the status chained."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    original = _status_error(429)
    _install_fake(monkeypatch, raises=original)

    # Act / Assert
    with pytest.raises(transcribe.TranscribeError) as exc_info:
        asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path)))
    assert exc_info.value.__cause__ is original
    assert exc_info.value.status_code == 429


@pytest.mark.parametrize("langs", [["en"], ["en", "es"], []])
def test_transcribe_passes_languages_list_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, langs: list[str]
) -> None:
    """The caller's language list is forwarded with no normalization."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    asyncio.run(transcribe.transcribe(audio_path, **_kwargs(audio_path, langs=langs)))

    # Assert
    assert fake.create_kwargs["languages"] == langs
