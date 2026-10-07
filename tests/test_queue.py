"""Tests for the tldw SQLite-backed transcript queue.

These tests drive a real QueueStore against a real SQLite file under tmp_path;
nothing is mocked. They pin the enqueue dedup rule, the ordering and filtering
that next_due performs, the attempt and terminal-state bookkeeping, the counts
summary, and the restart-durability path that is the whole reason the queue
lives in SQLite. A raw sqlite3 reader helper is used where a value is only
reachable through a terminal row that next_due intentionally skips.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from tldw.feed import VideoEntry
from tldw.queue import (
    DEFAULT_PACING_SECONDS,
    QueueStore,
    TerminalState,
    open_store,
)


def _entry(video_id: str = "dQw4w9WgXcQ", **overrides: object) -> VideoEntry:
    """Build a VideoEntry with sensible defaults for queue tests."""
    defaults: dict[str, object] = {
        "video_id": video_id,
        "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
        "title": f"Test {video_id}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel_name": "Test Channel",
        "published": None,
        "updated": None,
    }
    defaults.update(overrides)
    return VideoEntry(**defaults)  # type: ignore[arg-type]


def _read_row(path: Path, video_id: str) -> sqlite3.Row:
    """Read one videos row straight from the file for fields next_due hides."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def test_open_store_creates_db_file_when_missing(tmp_path: Path) -> None:
    """Opening a store at a nonexistent path creates the database file."""
    # Arrange
    path = tmp_path / "queue.sqlite3"

    # Act
    store = open_store(path)

    # Assert
    assert path.exists()
    store.close()


def test_open_store_creates_parent_directory(tmp_path: Path) -> None:
    """A path whose parent does not exist has the parent created for it."""
    # Arrange
    path = tmp_path / "nested" / "deeper" / "queue.sqlite3"

    # Act
    store = open_store(path)

    # Assert
    assert path.parent.is_dir()
    assert path.exists()
    store.close()


def test_open_store_sets_wal_journal_mode(tmp_path: Path) -> None:
    """The store enables WAL journaling for reader/writer concurrency."""
    # Arrange
    path = tmp_path / "queue.sqlite3"
    store = open_store(path)
    conn = sqlite3.connect(str(path))

    # Act
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    # Assert
    assert mode == "wal"
    store.close()


def test_enqueue_returns_true_then_false_for_duplicate(tmp_path: Path) -> None:
    """The same video enqueued twice inserts once."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    entry = _entry()

    # Act
    first = store.enqueue(entry, now=100.0)
    second = store.enqueue(entry, now=200.0)

    # Assert
    assert first is True
    assert second is False
    store.close()


def test_enqueue_persists_all_fields(tmp_path: Path) -> None:
    """A fresh record carries the entry fields and the initial queue values."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    entry = _entry("abc123", published="2026-01-01T00:00:00+00:00")

    # Act
    store.enqueue(entry, now=100.0)
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.video_id == "abc123"
    assert record.url == entry.url
    assert record.channel_id == entry.channel_id
    assert record.channel_name == entry.channel_name
    assert record.title == entry.title
    assert record.published == "2026-01-01T00:00:00+00:00"
    assert record.enqueued_at == 100.0
    assert record.attempts == 0
    assert record.last_attempt_at is None
    assert record.next_attempt_at == 100.0
    assert record.transcript_path is None
    assert record.terminal_state is None
    assert record.detail is None
    store.close()


def test_enqueue_with_default_now_uses_time_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When now is omitted the enqueue timestamps come from time.time()."""
    # Arrange
    fixed = 1234.5
    monkeypatch.setattr(time, "time", lambda: fixed)
    store = open_store(tmp_path / "queue.sqlite3")

    # Act
    store.enqueue(_entry())
    record = store.next_due()

    # Assert
    assert record is not None
    assert record.enqueued_at == fixed
    assert record.next_attempt_at == fixed
    store.close()


def test_next_due_returns_oldest_first(tmp_path: Path) -> None:
    """The record with the smallest next_attempt_at is returned first."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    for index, video_id in enumerate(["oldest", "middle", "newest"]):
        store.enqueue(_entry(video_id), now=100.0 + index)

    # Act
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.video_id == "oldest"
    store.close()


def test_next_due_skips_future_records(tmp_path: Path) -> None:
    """A record scheduled in the future is not returned yet."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    now = time.time()
    store.enqueue(_entry(), now=now + 999)

    # Act
    record = store.next_due(now=now)

    # Assert
    assert record is None
    store.close()


def test_next_due_skips_terminal_records(tmp_path: Path) -> None:
    """A record with a terminal state is never returned by next_due."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry("done"), now=100.0)
    store.enqueue(_entry("pending"), now=101.0)
    store.mark_terminal("done", TerminalState.DONE)

    # Act
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.video_id == "pending"
    store.close()


def test_next_due_returns_none_when_empty(tmp_path: Path) -> None:
    """An empty queue yields None."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")

    # Act
    record = store.next_due()

    # Assert
    assert record is None
    store.close()


def test_mark_attempt_bumps_attempts_and_sets_last_attempt_at(tmp_path: Path) -> None:
    """One mark_attempt increments the count and records the attempt time."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)

    # Act
    store.mark_attempt("dQw4w9WgXcQ", now=42.0)
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.attempts == 1
    assert record.last_attempt_at == 42.0
    store.close()


def test_mark_attempt_with_default_now_uses_time_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting now makes last_attempt_at come from time.time()."""
    # Arrange
    fixed = 777.0
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)
    monkeypatch.setattr(time, "time", lambda: fixed)

    # Act
    store.mark_attempt("dQw4w9WgXcQ")
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.last_attempt_at == fixed
    store.close()


def test_mark_attempt_updates_next_attempt_at_when_provided(tmp_path: Path) -> None:
    """A provided next_attempt_at replaces the stored schedule."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)

    # Act
    store.mark_attempt("dQw4w9WgXcQ", now=120.0, next_attempt_at=500.0)
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.next_attempt_at == 500.0
    store.close()


def test_mark_attempt_preserves_existing_next_attempt_at_when_not_provided(
    tmp_path: Path,
) -> None:
    """Omitting next_attempt_at leaves the existing schedule in place."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)

    # Act
    store.mark_attempt("dQw4w9WgXcQ", now=120.0, next_attempt_at=None)
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.next_attempt_at == 100.0
    store.close()


def test_mark_attempt_sets_transcript_path_when_provided(tmp_path: Path) -> None:
    """A provided transcript_path is stored on the record."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)

    # Act
    store.mark_attempt(
        "dQw4w9WgXcQ", now=120.0, transcript_path="/tmp/abc123.vtt"
    )
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.transcript_path == "/tmp/abc123.vtt"
    store.close()


def test_mark_attempt_preserves_transcript_path_when_not_provided(
    tmp_path: Path,
) -> None:
    """Omitting transcript_path leaves an existing path in place."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)
    store.mark_attempt("dQw4w9WgXcQ", now=110.0, transcript_path="/tmp/abc123.vtt")

    # Act
    store.mark_attempt("dQw4w9WgXcQ", now=120.0, transcript_path=None)
    record = store.next_due(now=1e12)

    # Assert
    assert record is not None
    assert record.transcript_path == "/tmp/abc123.vtt"
    store.close()


def test_mark_terminal_excludes_from_next_due(tmp_path: Path) -> None:
    """A terminal record is skipped by next_due."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry("gone"), now=100.0)

    # Act
    store.mark_terminal("gone", TerminalState.GIVE_UP_UNAVAILABLE)
    record = store.next_due(now=1e12)

    # Assert
    assert record is None
    store.close()


def test_mark_terminal_stores_detail(tmp_path: Path) -> None:
    """The detail passed to mark_terminal survives a reopen."""
    # Arrange
    path = tmp_path / "queue.sqlite3"
    store = open_store(path)
    store.enqueue(_entry(), now=100.0)

    # Act
    store.mark_terminal("dQw4w9WgXcQ", TerminalState.DONE, detail="posted to discord")
    store.close()
    reopened = open_store(path)
    row = _read_row(path, "dQw4w9WgXcQ")

    # Assert
    assert row["detail"] == "posted to discord"
    assert row["terminal_state"] == "DONE"
    reopened.close()


def test_counts_groups_by_state(tmp_path: Path) -> None:
    """Counts report pending and each terminal state separately."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    for index in range(5):
        store.enqueue(_entry(f"vid{index}"), now=100.0 + index)
    store.mark_terminal("vid0", TerminalState.DONE)
    store.mark_terminal("vid1", TerminalState.DONE)
    store.mark_terminal("vid2", TerminalState.GIVE_UP_NEVER)

    # Act
    counts = store.counts()

    # Assert
    assert counts == {"pending": 2, "DONE": 2, "GIVE_UP_NEVER": 1}
    store.close()


def test_counts_includes_pending_key_even_when_zero(tmp_path: Path) -> None:
    """An empty store still reports a zero pending count."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")

    # Act
    counts = store.counts()

    # Assert
    assert counts == {"pending": 0}
    store.close()


def test_counts_includes_unknown_terminal_state(tmp_path: Path) -> None:
    """A state written by a newer version is reported rather than dropped."""
    # Arrange
    path = tmp_path / "queue.sqlite3"
    store = open_store(path)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO videos "
            "(video_id, url, channel_id, channel_name, title, enqueued_at, "
            " attempts, next_attempt_at, terminal_state) "
            "VALUES ('future', 'u', 'c', 'n', 't', 1.0, 0, 1.0, 'FUTURE_STATE')"
        )
        conn.commit()
    finally:
        conn.close()

    # Act
    counts = store.counts()

    # Assert
    assert counts == {"pending": 0, "FUTURE_STATE": 1}
    store.close()


def test_store_survives_reopen(tmp_path: Path) -> None:
    """Queue contents and details persist across a close and reopen."""
    # Arrange
    path = tmp_path / "queue.sqlite3"
    store = open_store(path)
    store.enqueue(_entry("kept"), now=100.0)
    store.enqueue(_entry("done"), now=101.0)

    # Act
    store.mark_terminal("done", TerminalState.DONE, detail="finished")
    store.close()
    reopened = open_store(path)
    counts = reopened.counts()
    row = _read_row(path, "done")

    # Assert
    assert counts == {"pending": 1, "DONE": 1}
    assert row["detail"] == "finished"
    reopened.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Closing twice does not raise."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")

    # Act / Assert
    store.close()
    store.close()


def test_operations_after_close_raise_runtime_error(tmp_path: Path) -> None:
    """Every public operation rejects use after close."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(_entry(), now=100.0)
    store.close()

    # Act / Assert
    with pytest.raises(RuntimeError):
        store.enqueue(_entry("other"))
    with pytest.raises(RuntimeError):
        store.next_due()
    with pytest.raises(RuntimeError):
        store.mark_attempt("dQw4w9WgXcQ")
    with pytest.raises(RuntimeError):
        store.mark_terminal("dQw4w9WgXcQ", TerminalState.DONE)
    with pytest.raises(RuntimeError):
        store.counts()


def test_default_pacing_seconds_is_exported() -> None:
    """The worker's pacing constant has its single source of truth here."""
    # Arrange
    # The worker imports DEFAULT_PACING_SECONDS from this module.

    # Act
    value = DEFAULT_PACING_SECONDS

    # Assert
    assert value == 5.0
    assert isinstance(value, float)


def test_open_store_returns_queue_store(tmp_path: Path) -> None:
    """open_store is the documented factory for QueueStore."""
    # Arrange
    path = tmp_path / "queue.sqlite3"

    # Act
    store = open_store(path)

    # Assert
    assert isinstance(store, QueueStore)
    store.close()
