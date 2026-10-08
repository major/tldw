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
    """Build the default transcribe kwargs with optional overrides.

    The language is hardcoded inside ``transcribe()`` - there is no
    ``langs`` kwarg to pass. Callers can override the optional
    ``prompt`` and ``keywords`` context fields.
    """
    values: dict[str, Any] = {
        "model": "gpt-transcribe",
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


def test_transcribe_always_pins_us_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``languages`` request field is hardcoded to ``["en"]``.

    The plural ``languages`` field is the right key for gpt-transcribe
    per OpenAI's docs (the singular ``language=`` is for older models).
    tldw's contract is US English always - no caller input can change
    the list.
    """
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


def test_transcribe_forwards_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty ``prompt`` reaches the SDK as the ``prompt`` kwarg."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))
    expected_prompt = (
        "YouTube video titled 'Why Do Police Ask THIS During a Traffic Stop?' "
        "from channel 'Hampton Law'. US English transcript."
    )

    # Act
    asyncio.run(
        transcribe.transcribe(
            audio_path,
            **_kwargs(audio_path, prompt=expected_prompt),
        )
    )

    # Assert
    assert fake.create_kwargs["prompt"] == expected_prompt


def test_transcribe_omits_empty_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty/None ``prompt`` is omitted from the SDK call.

    Per OpenAI's docs: "use these inputs only for context relevant to the
    audio; don't restate the transcription task." So we forward only
    meaningful context.
    """
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    for empty in ("", None):
        asyncio.run(
            transcribe.transcribe(
                audio_path,
                **_kwargs(audio_path, prompt=empty),
            )
        )

    # Assert
    assert "prompt" not in fake.create_kwargs


def test_transcribe_forwards_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty ``keywords`` list reaches the SDK as the ``keywords`` kwarg."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))
    keywords = ["officer", "medication", "traffic stop", "DUI"]

    # Act
    asyncio.run(
        transcribe.transcribe(
            audio_path,
            **_kwargs(audio_path, keywords=keywords),
        )
    )

    # Assert
    assert fake.create_kwargs["keywords"] == keywords


def test_transcribe_omits_empty_keywords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty/None ``keywords`` list is omitted from the SDK call."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, result=_FakeResult("ok"))

    # Act
    for empty in ([], None):
        asyncio.run(
            transcribe.transcribe(
                audio_path,
                **_kwargs(audio_path, keywords=empty),
            )
        )

    # Assert
    assert "keywords" not in fake.create_kwargs


def test_build_prompt_returns_none_with_no_metadata() -> None:
    """``build_prompt()`` returns ``None`` so the API call omits the field."""
    # Act
    result = transcribe.build_prompt()

    # Assert
    assert result is None


def test_build_prompt_explicitly_says_us_english() -> None:
    """The prompt anchors the ASR to US English.

    This is the literal guarantee that fulfills the "always US English,
    no matter what" contract.
    """
    # Act
    result = transcribe.build_prompt(
        title="Why Do Police Ask THIS During a Traffic Stop?",
        channel_name="Hampton Law",
    )

    # Assert
    assert result is not None
    assert "US English" in result


def test_build_prompt_quotes_title_and_channel() -> None:
    """``title`` and ``channel_name`` are quoted so the model sees them as
    labels, not as instructions."""
    # Act
    result = transcribe.build_prompt(
        title="Coast to Coast AM",
        channel_name="Hampton Law",
    )

    # Assert
    assert result is not None
    assert repr("Coast to Coast AM") in result
    assert repr("Hampton Law") in result


def test_build_prompt_with_only_title() -> None:
    """A title alone is enough to anchor the ASR; the channel is optional."""
    # Act
    result = transcribe.build_prompt(title="Just a title")

    # Assert
    assert result is not None
    assert "US English" in result


def test_build_prompt_with_only_channel() -> None:
    """A channel alone is enough to anchor the ASR; the title is optional."""
    # Act
    result = transcribe.build_prompt(channel_name="Hampton Law")

    # Assert
    assert result is not None
    assert "US English" in result
