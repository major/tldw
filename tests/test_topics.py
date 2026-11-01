"""Tests for immutable topic configuration."""

import json
from pathlib import Path
from typing import cast, MutableMapping

import pytest


def test_load_topics_resolves_prompt_and_secret(tmp_path: Path) -> None:
    """Resolve a relative prompt and index its active source without leaking secrets."""
    from tldw.topics import load_topics

    # Arrange
    (tmp_path / "prompt.txt").write_text(" prompt\n")
    path = tmp_path / "topics.yml"
    path.write_text(
        "version: 1\ntopics:\n  default:\n    channels:\n      - channel_id: UCabcdefghijklmnopqrstuv\n    prompt: {file: prompt.txt}\n    discord: {webhook_env: WEBHOOK}\n"
    )
    # Act
    registry = load_topics(path, environ={"WEBHOOK": "secret-value"})
    # Assert
    assert registry.topics["default"].system_prompt == " prompt\n"
    assert registry.channel_ids == ("UCabcdefghijklmnopqrstuv",)
    assert "secret-value" not in repr(registry)
    with pytest.raises(TypeError):
        cast(MutableMapping, registry.topics)["other"] = registry.topics["default"]


@pytest.mark.parametrize(
    "text",
    [
        "version: 1\nversion: 1\ntopics: {}\n",
        "version: true\ntopics: {}\n",
        "version: 1\ntopics: {}\n",
        "version: 1\ntopics:\n  default: {enabled: false, channels: [], prompt: {builtin: unknown}, discord: {webhook_env: WEBHOOK}}\n",
    ],
)
def test_invalid_topics_are_rejected(tmp_path: Path, text: str) -> None:
    """Reject invalid structure and duplicate keys."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    path = tmp_path / "topics.yml"
    path.write_text(text)
    # Act and assert
    with pytest.raises(TopicConfigError):
        load_topics(path, environ={})


@pytest.fixture
def topic_document() -> dict:
    """Return a valid document that individual tests can modify."""
    return {
        "version": 1,
        "topics": {
            "default": {
                "channels": [{"channel_id": "UCabcdefghijklmnopqrstuv"}],
                "prompt": {"builtin": "trading"},
                "discord": {"webhook_env": "WEBHOOK"},
            }
        },
    }


@pytest.mark.parametrize("version", [True, 1.0])
def test_version_requires_integer(
    tmp_path: Path, topic_document: dict, version: object
) -> None:
    """Reject boolean and floating-point versions on otherwise valid documents."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    topic_document["version"] = version
    path = tmp_path / "topics.yml"
    path.write_text(json.dumps(topic_document))
    # Act and assert
    with pytest.raises(TopicConfigError, match="version"):
        load_topics(path, environ={"WEBHOOK": "secret"})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_default", "default topic is required"),
        ("invalid_slug", "invalid topic slug"),
        ("no_channels", "enabled topic requires channels"),
        ("both_prompts", "choose exactly one prompt source"),
        ("no_prompt", "choose exactly one prompt source"),
        ("unknown_field", "extra_forbidden"),
        ("invalid_channel", "channel_id"),
        ("duplicate_within", "duplicate active source"),
        ("duplicate_across", "duplicate active source"),
        ("missing_prompt", "cannot read prompt"),
        ("blank_prompt", "prompt is blank"),
        ("absolute_prompt", "expected relative path"),
    ],
)
def test_invalid_topic_details(
    tmp_path: Path, topic_document: dict, change: str, message: str
) -> None:
    """Report safe, useful errors for invalid topic documents."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    topic = topic_document["topics"]["default"]
    if change == "missing_default":
        topic_document["topics"] = {"other": topic}
    elif change == "invalid_slug":
        topic_document["topics"]["Bad slug"] = topic
    elif change == "no_channels":
        topic["channels"] = []
    elif change == "both_prompts":
        topic["prompt"]["file"] = "prompt.txt"
    elif change == "no_prompt":
        topic["prompt"] = {}
    elif change == "unknown_field":
        topic["sentinel-secret"] = "sentinel-secret"
    elif change == "invalid_channel":
        topic["channels"][0]["channel_id"] = "sentinel-secret"
    elif change == "duplicate_within":
        topic["channels"].append(topic["channels"][0])
    elif change == "duplicate_across":
        topic_document["topics"]["other"] = topic
    else:
        topic["prompt"] = {"file": "/absolute" if change == "absolute_prompt" else "prompt.txt"}
        if change == "blank_prompt":
            (tmp_path / "prompt.txt").write_text(" \n")
    path = tmp_path / "topics.yml"
    path.write_text(json.dumps(topic_document))
    # Act and assert
    with pytest.raises(TopicConfigError, match=message) as exc:
        load_topics(path, environ={"WEBHOOK": "sentinel-secret"})
    assert "sentinel-secret" not in str(exc.value)


@pytest.mark.parametrize("enabled", [True, False])
def test_missing_destination_depends_on_enabled(
    tmp_path: Path, topic_document: dict, enabled: bool
) -> None:
    """Only active topics require a resolved destination."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    topic_document["topics"]["default"]["enabled"] = enabled
    path = tmp_path / "topics.yml"
    path.write_text(json.dumps(topic_document))
    # Act and assert
    if enabled:
        with pytest.raises(TopicConfigError, match="destination"):
            load_topics(path, environ={})
    else:
        assert load_topics(path, environ={}).topics["default"].webhook_url is None


def test_prompt_preserves_crlf(tmp_path: Path, topic_document: dict) -> None:
    """Preserve prompt line endings exactly."""
    from tldw.topics import load_topics

    # Arrange
    (tmp_path / "prompt.txt").write_bytes(b"first\r\nsecond\r\n")
    topic_document["topics"]["default"]["prompt"] = {"file": "prompt.txt"}
    path = tmp_path / "topics.yml"
    path.write_text(json.dumps(topic_document))
    # Act and assert
    assert load_topics(path, environ={"WEBHOOK": "value"}).topics["default"].system_prompt == (
        "first\r\nsecond\r\n"
    )


def test_yaml_error_omits_source(tmp_path: Path) -> None:
    """Syntax errors contain locations rather than secret source excerpts."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    path = tmp_path / "topics.yml"
    path.write_text("secret: sentinel-secret\nbroken: [\n")
    # Act and assert
    with pytest.raises(TopicConfigError, match="line") as exc:
        load_topics(path)
    assert "sentinel-secret" not in str(exc.value)
