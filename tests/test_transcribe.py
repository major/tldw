"""Tests for the tldw.transcribe OpenAI gpt-transcribe seam.

Every test monkeypatches the module-level ``AsyncOpenAI`` with a local fake, so
no request ever leaves the process (pytest-socket also blocks real TCP). The
fake records the constructor and ``create`` kwargs for inspection and returns a
canned result whose ``.text`` the production code reads.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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
        self._parent.calls.append(self._parent.create_kwargs)

        async def _coro() -> _FakeResult:
            """Return the next scripted result or raise its error."""
            if self._parent.delay:
                await asyncio.sleep(self._parent.delay)
            if self._parent.script:
                item = self._parent.script.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
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
        script: list[Any] | None = None,
    ) -> None:
        """Configure canned behavior and record constructor kwargs."""
        self.init_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
        }
        self.create_kwargs: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.script = list(script or [])
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
        """Record client options and return the configured fake."""
        fake.init_kwargs = dict(kwargs)
        return fake

    @asynccontextmanager
    async def _chunks(path: Path):
        """Keep placeholder audio away from real media tools."""
        yield [path]

    monkeypatch.setattr(transcribe, "AsyncOpenAI", _factory)
    monkeypatch.setattr(transcribe, "audio_chunks", _chunks, raising=False)
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
        asyncio.run(
            transcribe.transcribe(audio_path, **_kwargs(audio_path, api_key=""))
        )


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


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, "credit_balance_exhausted"), (400, "invalid_value")],
)
async def test_transcribe_preserves_provider_failure_as_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    """Provider failures retain their structured cause without producing text."""
    # Arrange
    audio_path = _audio_file(tmp_path)
    request = httpx2.Request("POST", "https://example.test/v1/audio/transcriptions")
    body = {"code": code, "message": "Provider failure"}
    original = openai.APIStatusError(
        "Provider failure",
        response=httpx2.Response(status, request=request),
        body=body,
    )
    _install_fake(monkeypatch, raises=original)

    # Act
    with pytest.raises(transcribe.TranscribeError) as exc_info:
        await transcribe.transcribe(audio_path, **_kwargs(audio_path))

    # Assert
    assert exc_info.value.status_code == status
    assert exc_info.value.error_code == (code if status == 429 else None)
    assert exc_info.value.__cause__ is original
    assert original.body == body
    assert audio_path.exists()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"type": "insufficient_quota"}}, "insufficient_quota"),
        ({"code": "rate_limit_exceeded"}, "rate_limit_exceeded"),
        ({"error": "secret"}, None),
        (None, None),
    ],
)
async def test_transcribe_classifies_safe_error_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: object, expected: str | None
) -> None:
    """Only known structured codes cross the transcription boundary."""
    audio_path = _audio_file(tmp_path)
    request = httpx2.Request("POST", "https://example.test")
    original = openai.APIStatusError(
        "secret", response=httpx2.Response(429, request=request), body=body
    )
    _install_fake(monkeypatch, raises=original)
    with pytest.raises(transcribe.TranscribeError) as exc_info:
        await transcribe.transcribe(audio_path, **_kwargs(audio_path))
    assert exc_info.value.error_code == expected


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


@pytest.mark.parametrize("fails", [False, True])
async def test_chunks_are_sequential_and_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """Join ordered chunks, or stop on failure without leaking handles."""
    # Arrange
    source = _audio_file(tmp_path)
    paths = [tmp_path / f"chunk-{i}.webm" for i in range(3)]
    for path in paths:
        path.write_bytes(b"chunk")
    script = [
        _FakeResult("first"),
        _status_error(400) if fails else _FakeResult("second"),
        _FakeResult("third"),
    ]
    fake = _install_fake(monkeypatch, script=script)
    cleaned = []

    @asynccontextmanager
    async def chunks(_path: Path):
        """Simulate temporary chunks and record cleanup."""
        try:
            yield paths
        finally:
            cleaned.append(True)
            for path in paths:
                path.unlink()

    monkeypatch.setattr(transcribe, "audio_chunks", chunks)
    # Act
    if fails:
        with pytest.raises(transcribe.TranscribeError):
            await transcribe.transcribe(
                source, **_kwargs(source, prompt="context", keywords=["word"])
            )
    else:
        result = await transcribe.transcribe(
            source, **_kwargs(source, prompt="context", keywords=["word"])
        )
        assert result == "first\n\nsecond\n\nthird"
    # Assert
    assert len(fake.calls) == (2 if fails else 3)
    assert [Path(call["file"].name) for call in fake.calls] == paths[: len(fake.calls)]
    assert all(
        call["file"].closed
        and call["prompt"] == "context"
        and call["keywords"] == ["word"]
        for call in fake.calls
    )
    assert fake.init_kwargs["max_retries"] == 0
    assert cleaned and source.exists() and not any(path.exists() for path in paths)


async def test_timeout_covers_preparation_and_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation and requests share one deadline."""
    # Arrange
    source = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch, delay=0.04, result=_FakeResult("late"))

    @asynccontextmanager
    async def chunks(path: Path):
        """Consume part of the total deadline before the request."""
        await asyncio.sleep(0.04)
        yield [path]

    monkeypatch.setattr(transcribe, "audio_chunks", chunks)
    # Act / Assert
    with pytest.raises(TimeoutError):
        await transcribe.transcribe(source, **_kwargs(source, timeout_s=0.06))
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("duration_limit", "Audio exceeds the provider duration limit."),
        ("size_limit", "Audio exceeds the provider upload size limit."),
        ("context_limit", "Audio exceeds the provider context limit."),
        ("invalid_audio", "Provider could not decode the audio."),
        ("unsupported_format", "Provider does not support the audio format."),
        ("secret", None),
    ],
)
def test_reason_allowlist(category: str, expected: str | None) -> None:
    """Only fixed, recognized reason categories are exposed."""
    assert transcribe.TranscribeError("secret", reason=category).reason == expected


async def test_permanent_preparation_failure_preserves_safe_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local rejection keeps its disposition without fabricating an HTTP status."""
    # Arrange
    from tldw.transcription_audio import AudioPreparationError

    source = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch)

    @asynccontextmanager
    async def chunks(_path: Path):
        """Reject invalid duration metadata before any provider request."""
        raise AudioPreparationError("invalid_duration", permanent=True)
        yield []

    monkeypatch.setattr(transcribe, "audio_chunks", chunks)

    # Act / Assert
    with pytest.raises(transcribe.TranscribeError) as caught:
        await transcribe.transcribe(source, **_kwargs(source))
    assert caught.value.permanent
    assert caught.value.status_code is None
    assert caught.value.reason == "Audio duration is unavailable or invalid."
    assert fake.calls == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {
                "error": {
                    "message": "Audio duration exceeds the maximum duration. secret @everyone https://secret.test\nsecret"
                }
            },
            "Audio exceeds the provider duration limit.",
        ),
        (
            {"message": "secret " + "x" * 2000 + " maximum audio duration exceeded"},
            None,
        ),
        ({"error": ["secret"]}, None),
        ({"message": "secret @everyone https://secret.test\nsecret"}, None),
    ],
)
async def test_provider_reason_is_safe_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: object, expected: str | None
) -> None:
    """Provider text is classified within a bound and never surfaced."""
    # Arrange
    source = _audio_file(tmp_path)
    request = httpx2.Request("POST", "https://example.test")
    original = openai.APIStatusError(
        "secret", response=httpx2.Response(400, request=request), body=body
    )
    _install_fake(monkeypatch, raises=original)
    # Act
    with pytest.raises(transcribe.TranscribeError) as caught:
        await transcribe.transcribe(source, **_kwargs(source))
    # Assert
    assert caught.value.reason == expected
    assert caught.value.__cause__ is original
    assert "secret" not in str(caught.value) and "@everyone" not in str(caught.value)


async def test_preparation_failure_makes_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation failures are sanitized and never start an API request."""
    # Arrange
    source = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch)

    @asynccontextmanager
    async def chunks(_path: Path):
        """Fail before yielding any chunk."""
        raise transcribe.AudioPreparationError("secret")
        yield []

    monkeypatch.setattr(transcribe, "audio_chunks", chunks)
    # Act / Assert
    with pytest.raises(
        transcribe.TranscribeError, match="Audio preparation failed"
    ) as caught:
        await transcribe.transcribe(source, **_kwargs(source))
    assert caught.value.status_code is None and not fake.calls and source.exists()


async def test_timeout_cancels_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The total deadline cancels preparation before any API request."""
    # Arrange
    source = _audio_file(tmp_path)
    fake = _install_fake(monkeypatch)
    cancelled = False

    @asynccontextmanager
    async def chunks(_path: Path):
        """Block until the deadline cancels preparation."""
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        finally:
            cancelled = True
        yield []

    monkeypatch.setattr(transcribe, "audio_chunks", chunks)
    # Act / Assert
    with pytest.raises(TimeoutError):
        await transcribe.transcribe(source, **_kwargs(source, timeout_s=0.01))
    assert cancelled and not fake.calls and source.exists()
