"""Offline configuration preflight tests."""

import sqlite3
from pathlib import Path

import pytest


def test_missing_queue_is_not_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested missing database fails without creating it."""
    from tldw.validate_config import main

    # Arrange
    database = tmp_path / "missing.sqlite"
    monkeypatch.delenv("TLDW_TOPICS_FILE", raising=False)
    # Act and assert
    with pytest.raises(SystemExit) as exc:
        main(["--queue-file", str(database)])
    assert exc.value.code != 0
    assert not database.exists()


def test_legacy_queue_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading a legacy queue does not migrate or alter it."""
    from tldw.validate_config import main

    # Arrange
    database = tmp_path / "queue.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE videos (video_id TEXT, terminal_state TEXT)")
        db.execute("CREATE TABLE failure_alerts (video_id TEXT, sent_at REAL)")
        db.execute("INSERT INTO videos VALUES ('one', NULL)")
    before = database.read_bytes()
    monkeypatch.delenv("TLDW_TOPICS_FILE", raising=False)
    # Act
    main(["--queue-file", str(database)])
    # Assert
    assert database.read_bytes() == before


def test_compare_legacy_allows_csv_only_in_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Migration comparison permits CSV while normal YAML validation rejects it."""
    from tldw.validate_config import main

    # Arrange
    channel = "UCabcdefghijklmnopqrstuv"
    path = tmp_path / "topics.yml"
    path.write_text(f"version: 1\ntopics:\n  default:\n    channels: [{{channel_id: {channel}}}]\n    prompt: {{builtin: trading}}\n    discord: {{webhook_env: TLDW_DISCORD_WEBHOOK_URL}}\n")
    monkeypatch.setenv("TLDW_TOPICS_FILE", str(path))
    monkeypatch.setenv("TLDW_CHANNEL_IDS", channel)
    monkeypatch.setenv("TLDW_DISCORD_WEBHOOK_URL", "secret-value")
    # Act and assert
    with pytest.raises(SystemExit):
        main([])
    main(["--compare-legacy"])


def test_blocked_disabled_backlog_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A disabled topic with backlog and no destination fails without secrets."""
    from tldw.validate_config import main

    # Arrange
    path = tmp_path / "topics.yml"
    path.write_text("version: 1\ntopics:\n  default: {enabled: false, channels: [], prompt: {builtin: trading}, discord: {webhook_env: MISSING}}\n")
    database = tmp_path / "queue.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE videos (video_id TEXT, terminal_state TEXT)")
        db.execute("CREATE TABLE failure_alerts (video_id TEXT, sent_at REAL)")
        db.execute("INSERT INTO videos VALUES ('one', 'DONE')")
        db.execute("INSERT INTO failure_alerts VALUES ('one', NULL)")
    monkeypatch.setenv("TLDW_TOPICS_FILE", str(path))
    monkeypatch.delenv("TLDW_CHANNEL_IDS", raising=False)
    monkeypatch.delenv("MISSING", raising=False)
    monkeypatch.setenv("TLDW_DISCORD_WEBHOOK_URL", "secret-value")
    # Act and assert
    with pytest.raises(SystemExit):
        main(["--queue-file", str(database)])
    error = capsys.readouterr().err
    assert "blocked" in error
    assert "secret-value" not in error
