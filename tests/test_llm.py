"""Tests for the tldw LLM takeaway seam.

The Agent is exercised with pydantic-ai's ``FunctionModel`` so the real schema
validation, retry loop, and timeout path run fully offline. No test touches the
network: the OpenAI client is constructed but never sends a request.

The banner-off env var is set before pydantic-ai is imported so the agent banner
does not pollute pytest output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

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

from helpers import make_settings  # noqa: E402
from tldw.config import Settings  # noqa: E402
from tldw.llm import (  # noqa: E402
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
# OpenAIAnalyzer: Agent runs against FunctionModel
# ---------------------------------------------------------------------------


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


async def test_openai_analyzer_retries_on_invalid_first_response() -> None:
    """Invalid JSON on the first call is retried and the second call succeeds."""
    payload = _valid_payload()
    calls = {"count": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(parts=[TextPart(content="not json")])
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    analyzer = OpenAIAnalyzer(make_settings(openai_api_key="test-key"), model=_function_model(respond))
    result = await analyzer.analyze("transcript", video_id="vid1", title="Title")

    assert calls["count"] == 2
    assert len(result.items) == 3


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
    """A transcript over the cap is truncated before it reaches the model."""
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

    assert len(seen["prompt"]) <= 100
    assert any("truncating" in record.message for record in caplog.records)


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


def _takeaways_with_bullet_times(bullet_times: list[list[int]]) -> Takeaways:
    """Build valid Takeaways where each item's bullets use the given times.

    ``Takeaways`` enforces exactly three items, so ``bullet_times`` must hold
    three inner lists. Each inner list is one takeaway's bullet timestamps.
    """
    return Takeaways.model_validate(
        {
            "items": [
                {
                    "title": f"Takeaway {i}",
                    "summary": f"Summary {i}",
                    "bullets": [
                        {"text": f"Bullet {i}-{n}", "timestamp_seconds": seconds}
                        for n, seconds in enumerate(times)
                    ],
                }
                for i, times in enumerate(bullet_times)
            ]
        }
    )


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
        takeaways = _takeaways_with_bullet_times([[60], [60], [60]])
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_to_nearest(self) -> None:
        """A bullet between two cues snaps to the nearer one."""
        takeaways = _takeaways_with_bullet_times([[75], [75], [75]])
        cues = [
            Cue(start=0.0, text="a"),
            Cue(start=60.0, text="b"),
            Cue(start=120.0, text="c"),
        ]

        result = snap_timestamps(takeaways, cues)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_below_first_cue(self) -> None:
        """A bullet before all cues snaps to the first cue."""
        takeaways = _takeaways_with_bullet_times([[10], [10], [10]])
        cues = [Cue(start=60.0, text="a"), Cue(start=120.0, text="b")]

        # The 50s drift is over the default 30s limit, so raise the limit to
        # isolate the "no cue before this timestamp" boundary behavior.
        result = snap_timestamps(takeaways, cues, max_drift_seconds=1000)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_snap_above_last_cue(self) -> None:
        """A bullet after all cues snaps to the last cue."""
        takeaways = _takeaways_with_bullet_times([[999], [999], [999]])
        cues = [Cue(start=0.0, text="a"), Cue(start=60.0, text="b")]

        # The 939s drift is over the default 30s limit, so raise the limit to
        # isolate the "no cue after this timestamp" boundary behavior.
        result = snap_timestamps(takeaways, cues, max_drift_seconds=1000)

        assert _all_bullet_timestamps(result) == [60, 60, 60]

    def test_kept_when_drift_exceeds_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bullet far from any cue is kept as-is and logs a WARNING."""
        takeaways = _takeaways_with_bullet_times([[999], [999], [999]])
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
        takeaways = _takeaways_with_bullet_times([[10], [20], [30]])

        result = snap_timestamps(takeaways, [])

        assert result is takeaways

    def test_multiple_takeaways_all_processed(self) -> None:
        """All takeaways and all bullets are processed; structure preserved."""
        takeaways = _takeaways_with_bullet_times([[10, 70], [15, 75], [20, 80]])
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


def _takeaways_with_none_bullets() -> Takeaways:
    """Build valid Takeaways where every bullet has no timestamp."""
    return Takeaways.model_validate(
        {
            "items": [
                {
                    "title": f"Takeaway {i}",
                    "summary": f"Summary {i}",
                    "bullets": [{"text": f"Bullet {i}", "timestamp_seconds": None}],
                }
                for i in range(3)
            ]
        }
    )


def test_snap_timestamps_preserves_none_values() -> None:
    """Bullets with no timestamp pass through snap_timestamps unchanged."""
    takeaways = _takeaways_with_none_bullets()
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
