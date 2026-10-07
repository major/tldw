"""Tests for the tldw LLM takeaway seam.

The Agent is exercised with pydantic-ai's ``FunctionModel`` so the real schema
validation, retry loop, and timeout path run fully offline. No test touches the
network: the OpenCode Go client is constructed but never sends a request.

The banner-off env var is set before pydantic-ai is imported so the agent banner
does not pollute pytest output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import pytest
from pydantic import ValidationError

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel  # noqa: E402

from tldw.config import Settings  # noqa: E402
from tldw.llm import (  # noqa: E402
    OpencodeGoAnalyzer,
    Takeaway,
    TakeawayBullet,
    Takeaways,
    session_id_for,
)


def _settings(**overrides: Any) -> Settings:
    """Build Settings with an API key so the analyzer is enabled by default."""
    values: dict[str, Any] = {"opencode_api_key": "test-key"}
    values.update(overrides)
    return Settings(**values)


def _takeaway(index: int = 0, *, bullet_count: int = 1) -> Takeaway:
    """Build one valid Takeaway with the given number of bullets."""
    return Takeaway(
        title=f"Takeaway {index}",
        summary=f"Summary {index}",
        bullets=[
            TakeawayBullet(text=f"Bullet {index}-{n}", timestamp_seconds=index + n)
            for n in range(bullet_count)
        ],
    )


def _valid_payload(*, bullets_per_item: int = 1) -> dict[str, object]:
    """Build a JSON-ready payload with exactly three valid takeaways."""
    return {
        "items": [
            {
                "title": f"Takeaway {i}",
                "summary": f"Summary {i}",
                "bullets": [
                    {"text": f"Bullet {i}-{n}", "timestamp_seconds": i + n}
                    for n in range(bullets_per_item)
                ],
            }
            for i in range(3)
        ]
    }


def _function_model(fn: FunctionDef) -> FunctionModel:
    """Wrap a response function in a FunctionModel for offline Agent runs."""
    return FunctionModel(fn)


def _user_prompt(messages: list[ModelMessage]) -> str:
    """Return the user prompt text from a captured FunctionModel message list."""
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = part.content
                assert isinstance(content, str)
                return content
    raise AssertionError("no UserPromptPart found in messages")


# ---------------------------------------------------------------------------
# session_id_for
# ---------------------------------------------------------------------------


def test_session_id_for_deterministic() -> None:
    """The same video id yields the same session id across many calls."""
    assert len({session_id_for("abc123") for _ in range(100)}) == 1


def test_session_id_for_format() -> None:
    """The session id is ``ses_`` plus exactly 32 lowercase hex characters."""
    session_id = session_id_for("abc123")
    assert session_id.startswith("ses_")
    suffix = session_id[len("ses_") :]
    assert len(suffix) == 32
    assert all(char in "0123456789abcdef" for char in suffix)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [2, 4])
def test_takeaways_rejects_two_or_four_items(count: int) -> None:
    """Takeaways requires exactly three items."""
    items = [_takeaway(i) for i in range(count)]
    with pytest.raises(ValidationError):
        Takeaways(items=items)


def test_takeaway_requires_at_least_one_bullet() -> None:
    """A takeaway with no bullets fails validation."""
    with pytest.raises(ValidationError):
        Takeaway(title="t", summary="s", bullets=[])


def test_takeaway_bullet_timestamp_must_be_non_negative() -> None:
    """A negative bullet timestamp fails validation."""
    with pytest.raises(ValidationError):
        TakeawayBullet(text="b", timestamp_seconds=-1)


# ---------------------------------------------------------------------------
# OpencodeGoAnalyzer: Agent runs against FunctionModel
# ---------------------------------------------------------------------------


async def test_opencode_analyzer_returns_canned_takeaways() -> None:
    """A valid canned JSON response is parsed into Takeaways."""
    payload = _valid_payload()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpencodeGoAnalyzer(_settings(), model=_function_model(respond))
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert isinstance(result, Takeaways)
    assert len(result.items) == 3
    assert result.items[0].title == "Takeaway 0"
    assert result.items[0].bullets[0].timestamp_seconds == 0


async def test_opencode_analyzer_retries_on_invalid_first_response() -> None:
    """Invalid JSON on the first call is retried and the second call succeeds."""
    payload = _valid_payload()
    calls = {"count": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(parts=[TextPart(content="not json")])
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpencodeGoAnalyzer(_settings(), model=_function_model(respond))
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert calls["count"] == 2
    assert len(result.items) == 3


async def test_opencode_analyzer_times_out() -> None:
    """A slow model call is cut off by llm_timeout_seconds."""
    payload = _valid_payload()

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        await asyncio.sleep(0.5)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpencodeGoAnalyzer(
        _settings(llm_timeout_seconds=0.1), model=_function_model(respond)
    )
    with pytest.raises(TimeoutError):
        await analyzer.analyze("transcript", video_id="vid1", title="Title")


async def test_opencode_analyzer_truncates_oversized_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transcript over the cap is truncated before it reaches the model."""
    payload = _valid_payload()
    seen: dict[str, str] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["prompt"] = _user_prompt(messages)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpencodeGoAnalyzer(
        _settings(llm_max_input_chars=100), model=_function_model(respond)
    )
    with caplog.at_level(logging.WARNING):
        await analyzer.analyze("x" * 500, video_id="vid1", title="Title")

    assert len(seen["prompt"]) <= 100
    assert any("truncating" in record.message for record in caplog.records)


async def test_opencode_analyzer_caps_bullets() -> None:
    """More bullets than the config cap are sliced down per item."""
    payload = _valid_payload(bullets_per_item=4)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpencodeGoAnalyzer(
        _settings(takeaway_max_bullets=2), model=_function_model(respond)
    )
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert all(len(item.bullets) == 2 for item in result.items)


# ---------------------------------------------------------------------------
# OpencodeGoAnalyzer: client construction
# ---------------------------------------------------------------------------


async def test_opencode_analyzer_build_client_sets_required_headers() -> None:
    """The client carries the bearer token and the opencode session header."""
    analyzer = OpencodeGoAnalyzer(_settings())
    client = analyzer._build_client()
    try:
        headers = client.default_headers
        assert headers["Authorization"] == "Bearer test-key"
        session_header = headers["x-opencode-session"]
        assert isinstance(session_header, str)
        assert session_header.startswith("ses_")
    finally:
        await client.close()


def test_opencode_analyzer_requires_api_key() -> None:
    """Building the client without an API key raises ValueError."""
    analyzer = OpencodeGoAnalyzer(_settings(opencode_api_key=None))
    with pytest.raises(ValueError):
        analyzer._build_client()
