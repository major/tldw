"""Tests for immutable topic configuration."""

from pathlib import Path

import pytest


def test_load_topics_resolves_prompt_and_secret(tmp_path: Path) -> None:
    """Resolve a relative prompt and index its active source without leaking secrets."""
    from tldw.topics import load_topics

    # Arrange
    (tmp_path / "prompt.txt").write_text(" prompt\n")
    path = tmp_path / "topics.yml"
    path.write_text("version: 1\ntopics:\n  default:\n    channels:\n      - channel_id: UCabcdefghijklmnopqrstuv\n    prompt: {file: prompt.txt}\n    discord: {webhook_env: WEBHOOK}\n")
    # Act
    registry = load_topics(path, environ={"WEBHOOK": "secret-value"})
    # Assert
    assert registry.topics["default"].system_prompt == " prompt\n"
    assert registry.channel_ids == ("UCabcdefghijklmnopqrstuv",)
    assert "secret-value" not in repr(registry)
    with pytest.raises(TypeError):
        registry.topics["other"] = registry.topics["default"]


@pytest.mark.parametrize("text", [
    "version: 1\nversion: 1\ntopics: {}\n",
    "version: true\ntopics: {}\n",
    "version: 1\ntopics: {}\n",
    "version: 1\ntopics:\n  default: {enabled: false, channels: [], prompt: {builtin: unknown}, discord: {webhook_env: WEBHOOK}}\n",
])
def test_invalid_topics_are_rejected(tmp_path: Path, text: str) -> None:
    """Reject invalid structure and duplicate keys."""
    from tldw.topics import TopicConfigError, load_topics

    # Arrange
    path = tmp_path / "topics.yml"
    path.write_text(text)
    # Act and assert
    with pytest.raises(TopicConfigError):
        load_topics(path, environ={})
