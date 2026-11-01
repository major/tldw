"""Tests for the tldw LLM takeaway seam.

The Agent is exercised with pydantic-ai's ``FunctionModel`` so the real schema
validation, retry loop, and timeout path run fully offline. No test touches the
network: the OpenAI client is constructed but never sends a request.

The banner-off env var is set before pydantic-ai is imported so the agent banner
does not pollute pytest output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from unittest.mock import AsyncMock

import pytest
from openai.resources.responses import AsyncResponses
from openai.types.responses import Response
from pydantic import ValidationError

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.messages import (  # noqa: E402
    FinishReason,
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel  # noqa: E402
from pydantic_ai.exceptions import IncompleteToolCall  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402

from helpers import make_settings, make_takeaways  # noqa: E402
from tldw.config import Settings  # noqa: E402
from tldw.llm import (  # noqa: E402
    BULLET_MAX_CHARS,
    MAX_WATCH_ITEMS,
    SYSTEM_PROMPT,
    Cue,
    OpenAIAnalyzer,
    Takeaway,
    TakeawayBullet,
    Takeaways,
    snap_timestamps,
)


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


def _valid_payload(
    *, bullets_per_item: int = 1, watch: tuple[str, ...] = ()
) -> dict[str, object]:
    """Build a JSON-ready payload with a header and exactly three takeaways."""
    return {
        "bottom_line": "Bottom line for the trader.",
        "watch": list(watch),
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
# Schema validation
# ---------------------------------------------------------------------------


def test_llm_reexports_takeaway_contract() -> None:
    """The old and new paths expose the same schema classes and constraints."""
    # Arrange
    import tldw.llm as llm
    import tldw.takeaways as takeaways

    classes = ("TakeawayBullet", "Takeaway", "Takeaways")
    constraints = (
        "TITLE_MAX_CHARS", "SUMMARY_MAX_CHARS", "BULLET_MAX_CHARS", "MAX_BULLETS",
        "BOTTOM_LINE_MAX_CHARS", "WATCH_ITEM_MAX_CHARS", "MAX_WATCH_ITEMS",
    )

    # Act and assert
    for name in classes:
        assert getattr(llm, name) is getattr(takeaways, name)
    for name in constraints:
        assert getattr(llm, name) == getattr(takeaways, name)


@pytest.mark.parametrize("count", [2, 4])
def test_takeaways_rejects_two_or_four_items(count: int) -> None:
    """Takeaways requires exactly three items."""
    items = [_takeaway(i) for i in range(count)]
    with pytest.raises(ValidationError):
        Takeaways(bottom_line="Bottom line", items=items)


def test_takeaway_requires_at_least_one_bullet() -> None:
    """A takeaway with no bullets fails validation."""
    with pytest.raises(ValidationError):
        Takeaway(title="t", summary="s", bullets=[])


def test_takeaway_bullet_timestamp_must_be_non_negative() -> None:
    """A negative bullet timestamp fails validation."""
    with pytest.raises(ValidationError):
        TakeawayBullet(text="b", timestamp_seconds=-1)


def test_takeaway_bullet_rejects_text_over_max_chars() -> None:
    """A bullet longer than BULLET_MAX_CHARS fails validation."""
    with pytest.raises(ValidationError):
        TakeawayBullet(text="x" * (BULLET_MAX_CHARS + 1))


def test_takeaways_rejects_more_than_max_watch_items() -> None:
    """A watch list longer than MAX_WATCH_ITEMS fails validation."""
    watch = [f"SPY {i}" for i in range(MAX_WATCH_ITEMS + 1)]
    with pytest.raises(ValidationError):
        Takeaways(
            bottom_line="Bottom line",
            watch=watch,
            items=[_takeaway(i) for i in range(3)],
        )


# ---------------------------------------------------------------------------
# OpenAIAnalyzer: Agent runs against FunctionModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("override", [None, "Custom prompt"])
async def test_analyzer_system_prompt(override: str | None) -> None:
    """Keep the trading prompt exact and forward the selected prompt to the agent."""
    # Arrange
    from pydantic_ai.messages import SystemPromptPart

    captured: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Capture system prompts and return a valid response."""
        captured.extend(
            part.content
            for msg in messages
            for part in msg.parts
            if isinstance(part, SystemPromptPart)
        )
        return ModelResponse(parts=[TextPart(content=json.dumps(_valid_payload()))])

    kwargs = {} if override is None else {"system_prompt": override}
    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key"), model=_function_model(respond), **kwargs
    )
    # Act
    await analyzer.analyze("transcript", video_id="v", title="title")
    # Assert
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == (
        "910c538d53ebaeb3f9ecb616443730e7ddc4643d723142fbed548d606712fda5"
    )
    assert captured == [override or SYSTEM_PROMPT]


async def test_openai_analyzer_returns_canned_takeaways() -> None:
    """A valid canned JSON response is parsed into Takeaways."""
    payload = _valid_payload()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(make_settings(openai_api_key="test-key"), model=_function_model(respond))
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert isinstance(result, Takeaways)
    assert len(result.items) == 3
    assert result.items[0].title == "Takeaway 0"
    assert result.items[0].bullets[0].timestamp_seconds == 0


async def test_openai_analyzer_retries_on_invalid_first_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid output is retried and each response logs its own token usage."""
    payload = _valid_payload()
    calls = {"count": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Return invalid output once, with distinct usage for each response."""
        calls["count"] += 1
        return ModelResponse(
            parts=[TextPart(content="not json" if calls["count"] == 1 else json.dumps(payload))],
            usage=RequestUsage(input_tokens=100, output_tokens=calls["count"] * 10),
        )

    analyzer = OpenAIAnalyzer(make_settings(openai_api_key="test-key"), model=_function_model(respond))
    with caplog.at_level(logging.INFO, logger="tldw.llm"):
        result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert calls["count"] == 2
    assert len(result.items) == 3
    records = [r for r in caplog.records if getattr(r, "event", None) == "llm_response"]
    assert [r.__dict__["attempt"] for r in records] == [1, 2]
    assert [r.__dict__["output_tokens"] for r in records] == [10, 20]


@pytest.mark.parametrize(
    "output_tokens,reasoning_tokens,finish_reason,provider_reason,budget,level,message",
    [
        (899, 0, "stop", "completed", 1000, logging.INFO, "LLM response"),
        (900, 100, "stop", "completed", 1000, logging.WARNING, "LLM output token usage is near the limit"),
        (1000, None, "stop", None, 1000, logging.WARNING, "LLM output token usage is near the limit"),
        (800, 700, "length", "max_output_tokens", 1000, logging.WARNING, "LLM output token limit reached"),
        (800, None, None, "max_output_tokens", 1000, logging.WARNING, "LLM output token limit reached"),
        (100, None, None, None, 1000, logging.INFO, "LLM response"),
        (100, None, "stop", None, 0, logging.INFO, "LLM response"),
        (100, None, "stop", None, -1, logging.INFO, "LLM response"),
    ],
)
async def test_openai_analyzer_logs_response_metadata(
    caplog: pytest.LogCaptureFixture,
    output_tokens: int,
    reasoning_tokens: int | None,
    finish_reason: FinishReason | None,
    provider_reason: str | None,
    budget: int,
    level: int,
    message: str,
) -> None:
    """Usage warnings distinguish high usage from explicit budget exhaustion."""
    # Arrange: include private provider metadata that must never reach logs.
    response = ModelResponse(
        parts=[TextPart(content=json.dumps(_valid_payload()))],
        model_name="reported-model",
        usage=RequestUsage(
            input_tokens=1234,
            output_tokens=output_tokens,
            details={} if reasoning_tokens is None else {"reasoning_tokens": reasoning_tokens},
        ),
        finish_reason=finish_reason,
        provider_details=None if provider_reason is None else {
            "finish_reason": provider_reason, "private": "DO_NOT_LOG",
        },
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Return a valid briefing with the selected response metadata."""
        return response

    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key", llm_max_output_tokens=budget),
        model=FunctionModel(respond, model_name="reported-model"),
    )

    # Act: reuse the analyzer to confirm response numbering resets per video.
    with caplog.at_level(logging.INFO, logger="tldw.llm"):
        result = await analyzer.analyze("transcript", video_id="vid1", title="Title")
        await analyzer.analyze("transcript", video_id="vid2", title="Title")

    # Assert: one structured record per response, with no response content.
    assert len(result.items) == 3
    records = [r for r in caplog.records if getattr(r, "event", None) == "llm_response"]
    assert len(records) == 2
    for record, video_id in zip(records, ["vid1", "vid2"], strict=True):
        assert record.levelno == level
        assert record.getMessage() == message
        assert {key: record.__dict__[key] for key in (
            "video_id", "model", "attempt", "input_tokens", "output_tokens",
            "reasoning_tokens", "max_output_tokens", "output_budget_fraction",
            "finish_reason", "provider_finish_reason",
        )} == {
            "video_id": video_id,
            "model": "reported-model",
            "attempt": 1,
            "input_tokens": 1234,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "max_output_tokens": budget,
            "output_budget_fraction": output_tokens / budget if budget > 0 else None,
            "finish_reason": finish_reason,
            "provider_finish_reason": provider_reason,
        }
        assert "DO_NOT_LOG" not in repr(record.__dict__)
        assert "Bottom line for the trader." not in repr(record.__dict__)


async def test_openai_analyzer_logs_truncated_response_before_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """The real Responses adapter logs exhausted usage before validation fails."""
    # Arrange: emulate an SDK response with truncated structured tool output.
    response = Response.model_validate({
        "id": "resp_test", "created_at": 1, "object": "response",
        "model": "reported-model", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "parallel_tool_calls": False, "tool_choice": "required", "tools": [],
        "output": [{
            "type": "function_call", "id": "fc_test", "call_id": "call_test",
            "name": "final_result", "arguments": '{"bottom_line":',
            "status": "incomplete",
        }],
        "usage": {
            "input_tokens": 1200, "output_tokens": 8192, "total_tokens": 9392,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 7000},
        },
    })
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(AsyncResponses, "create", create)
    analyzer = OpenAIAnalyzer(make_settings(
        openai_api_key="test-key", llm_max_output_tokens=8192,
    ))

    # Act: no network is used, but the SDK response goes through the real adapter.
    with caplog.at_level(logging.INFO, logger="tldw.llm"):
        with pytest.raises(IncompleteToolCall):
            await analyzer.analyze("transcript", video_id="vid1", title="Title")

    # Assert: the warning survives the failed run and preserves provider details.
    assert create.await_count == 3
    assert create.call_args.kwargs["max_output_tokens"] == 8192
    records = [r for r in caplog.records if getattr(r, "event", None) == "llm_response"]
    assert [r.__dict__["attempt"] for r in records] == [1, 2, 3]
    for record in records:
        assert record.levelno == logging.WARNING
        assert record.getMessage() == "LLM output token limit reached"
        assert record.__dict__["finish_reason"] == "length"
        assert record.__dict__["provider_finish_reason"] == "max_output_tokens"
        assert record.__dict__["output_tokens"] == 8192
        assert record.__dict__["reasoning_tokens"] == 7000


async def test_openai_analyzer_times_out() -> None:
    """A slow model call is cut off by llm_timeout_seconds."""
    payload = _valid_payload()

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        await asyncio.sleep(0.5)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key", llm_timeout_seconds=0.1), model=_function_model(respond)
    )
    with pytest.raises(TimeoutError):
        await analyzer.analyze("transcript", video_id="vid1", title="Title")


async def test_openai_analyzer_truncates_oversized_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transcript over the cap is truncated; the title header is not counted."""
    payload = _valid_payload()
    seen: dict[str, str] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["prompt"] = _user_prompt(messages)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key", llm_max_input_chars=100), model=_function_model(respond)
    )
    with caplog.at_level(logging.WARNING):
        await analyzer.analyze("x" * 500, video_id="vid1", title="Title")

    assert seen["prompt"] == "Video title: Title\n---\n" + "x" * 100
    assert any("truncating" in record.message for record in caplog.records)


async def test_openai_analyzer_sends_title_header_before_transcript() -> None:
    """The user message starts with the video title, a rule, then the transcript."""
    payload = _valid_payload()
    seen: dict[str, str] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["prompt"] = _user_prompt(messages)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(make_settings(openai_api_key="test-key"), model=_function_model(respond))
    await analyzer.analyze("[0:00] SPY is testing 500.", video_id="vid1", title="Market Open")

    assert seen["prompt"] == "Video title: Market Open\n---\n[0:00] SPY is testing 500."


async def test_openai_analyzer_caps_bullets() -> None:
    """More bullets than the config cap are sliced down per item."""
    payload = _valid_payload(bullets_per_item=4)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key", takeaway_max_bullets=2), model=_function_model(respond)
    )
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert all(len(item.bullets) == 2 for item in result.items)


async def test_openai_analyzer_cap_bullets_keeps_bottom_line_and_watch() -> None:
    """Capping bullets must not drop the header fields bottom_line and watch."""
    payload = _valid_payload(bullets_per_item=4, watch=("SPY 500 if reclaimed",))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(
        make_settings(openai_api_key="test-key", takeaway_max_bullets=2), model=_function_model(respond)
    )
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert result.bottom_line == "Bottom line for the trader."
    assert result.watch == ["SPY 500 if reclaimed"]
    assert all(len(item.bullets) == 2 for item in result.items)


# ---------------------------------------------------------------------------
# OpenAIAnalyzer: client construction
# ---------------------------------------------------------------------------


async def test_openai_analyzer_build_client_sets_bearer_token() -> None:
    """The client carries the API key for the bearer token header."""
    analyzer = OpenAIAnalyzer(make_settings(openai_api_key="test-key"))
    client = analyzer._build_client()
    try:
        # The OpenAI SDK exposes api_key on the client; the Authorization
        # header is built lazily from it on each request, so checking the
        # attribute is the most direct evidence that the credential made it
        # onto the client.
        assert client.api_key == "test-key"
    finally:
        await client.close()


def test_openai_analyzer_requires_api_key() -> None:
    """Building the client without an API key raises ValueError."""
    analyzer = OpenAIAnalyzer(make_settings(openai_api_key=None))
    with pytest.raises(ValueError):
        analyzer._build_client()


# ---------------------------------------------------------------------------
# snap_timestamps
# ---------------------------------------------------------------------------


def _all_bullet_timestamps(takeaways: Takeaways) -> list[int | None]:
    """Return every bullet timestamp across all items, in item then bullet order."""
    return [
        bullet.timestamp_seconds
        for item in takeaways.items
        for bullet in item.bullets
    ]


class TestSnapTimestamps:
    """Snapping LLM-returned timestamps to real cue starts."""

    def test_snap_to_exact_match(self) -> None:
        """A bullet whose timestamp exactly matches a cue start is unchanged."""
        takeaways = make_takeaways(bullet_times=[[60], [60], [60]])
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_to_nearest(self) -> None:
        """A bullet between two cues snaps to the nearer one."""
        takeaways = make_takeaways(bullet_times=[[75], [75], [75]])
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_below_first_cue(self) -> None:
        """A bullet before all cues snaps to the first cue."""
        takeaways = make_takeaways(bullet_times=[[10], [10], [10]])
        cues = [Cue(start=60.0, text="a"), Cue(start=120.0, text="b")]

        # The 50s drift is over the default 30s limit, so raise the limit to
        # isolate the "no cue before this timestamp" boundary behavior.
        result = snap_timestamps(takeaways, cues, max_drift_seconds=1000)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_above_last_cue(self) -> None:
        """A bullet after all cues snaps to the last cue."""
        takeaways = make_takeaways(bullet_times=[[999], [999], [999]])
        cues = [Cue(start=0.0, text="a"), Cue(start=60.0, text="b")]

        # The 939s drift is over the default 30s limit, so raise the limit to
        # isolate the "no cue after this timestamp" boundary behavior.
        result = snap_timestamps(takeaways, cues, max_drift_seconds=1000)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_kept_when_drift_exceeds_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bullet far from any cue is kept as-is and logs a WARNING."""
        takeaways = make_takeaways(bullet_times=[[999], [999], [999]])
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        with caplog.at_level(logging.WARNING):
            result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [999, 999, 999]
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_empty_cues_returns_unchanged(self) -> None:
        """Empty cues list returns the takeaways unchanged."""
        takeaways = make_takeaways(bullet_times=[[10], [20], [30]])

        result = snap_timestamps(takeaways, [])

        assert result is takeaways

    def test_multiple_takeaways_all_processed(self) -> None:
        """All takeaways and all bullets are processed; structure preserved."""
        takeaways = make_takeaways(
            bullet_count=2, bullet_times=[[10, 70], [15, 75], [20, 80]]
        )
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [0, 60, 0, 60, 0, 60]
        assert [item.title for item in result.items] == [
            "Takeaway 0",
            "Takeaway 1",
            "Takeaway 2",
        ]
        assert [item.summary for item in result.items] == [
            "Summary 0",
            "Summary 1",
            "Summary 2",
        ]
        assert len(result.items) == 3
        assert all(len(item.bullets) == 2 for item in result.items)


# ---------------------------------------------------------------------------
# Optional timestamps (audio backend)
# ---------------------------------------------------------------------------


def test_takeaway_bullet_timestamp_seconds_defaults_to_none() -> None:
    """A bullet without a timestamp is valid and defaults the field to None."""
    bullet = TakeawayBullet(text="x")

    assert bullet.timestamp_seconds is None


def test_takeaway_bullet_rejects_negative_timestamp_when_set() -> None:
    """The ge=0 constraint still applies when a timestamp is provided."""
    with pytest.raises(ValidationError):
        TakeawayBullet(text="x", timestamp_seconds=-1)


def test_snap_timestamps_preserves_none_values() -> None:
    """Bullets with no timestamp pass through snap_timestamps unchanged."""
    takeaways = make_takeaways(bullet_text="Bullet {i}")
    cues = [Cue(start=0.0, text="a"), Cue(start=60.0, text="b")]

    result = snap_timestamps(takeaways, cues)

    assert result == takeaways
    assert all(
        bullet.timestamp_seconds is None
        for item in result.items
        for bullet in item.bullets
    )


def test_system_prompt_mentions_optional_timestamps() -> None:
    """The prompt tells the model it may omit timestamp_seconds."""
    assert "timestamp_seconds" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "rule",
    [
        "bottom_line",
        "watch",
        "US English",
        "If X, then Y",
    ],
)
def test_system_prompt_contains_briefing_rule(rule: str) -> None:
    """The prompt asks for the briefing header, watch conditions, and US English."""
    assert rule in SYSTEM_PROMPT


def test_system_prompt_drops_ungrounded_fact_rule() -> None:
    """The old rule that blanked audio-only output is gone from the prompt."""
    assert "Omit anything you cannot ground" not in SYSTEM_PROMPT
