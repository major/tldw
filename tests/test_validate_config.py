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
