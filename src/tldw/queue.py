"""Durable SQLite-backed queue for the transcript pipeline.

When the hub notifies tldw of a new video, the transcript is usually not ready
yet: YouTube's automatic-speech-recognition pipeline can take anywhere from ten
minutes to an hour. Asking YouTube repeatedly in the meantime both wastes work
and gets the caller rate limited. This module owns the small persistent queue
that lets a notification be accepted immediately and drained later by a
background worker. Because the queue lives in SQLite, it survives a restart:
a video enqueued before a crash is still there after the process comes back.

Enqueue is idempotent. A video is keyed by ``video_id`` and inserted with
``INSERT OR IGNORE``, so the hub delivering the same notification twice, or a
renewal and a fresh notification racing, only ever produces one row. Records
start with a terminal state of NULL, meaning "pending"; once the worker resolves
a video it stamps a ``TerminalState`` so ``next_due`` stops returning it.

Threading rule: every method here runs on the event-loop thread only. The
worker calls store methods before and after ``asyncio.to_thread(probe...)``,
never from inside the thread. Each SQLite call is sub-millisecond, so no lock
is used; confining all access to one thread is what keeps that safe.

The database uses WAL journaling with ``synchronous=NORMAL``. WAL lets a reader
and the writer proceed without blocking each other and makes commits crash safe,
while NORMAL drops the extra fsync per commit that FULL would require. For a
queue that can be rebuilt from missed notifications this is the right trade
between durability and latency.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tldw.feed import VideoEntry

__all__ = [
    "TerminalState",
    "QueueRecord",
    "QueueStore",
    "open_store",
]

logger = logging.getLogger(__name__)

# The videos table is the whole queue. ``terminal_state`` is NULL while a video
# is still pending and holds a TerminalState value once resolved. The partial
# index covers only pending rows, which is exactly the set next_due scans, so
# it stays small even as completed history grows. ``rate_limit_streak`` counts
# consecutive 429 outcomes for the row; any other outcome resets it to 0.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    channel_id      TEXT NOT NULL,
    channel_name    TEXT NOT NULL,
    title           TEXT NOT NULL,
    published       TEXT,
    enqueued_at     REAL NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_attempt_at REAL,
    next_attempt_at REAL NOT NULL,
    transcript_path TEXT,
    terminal_state  TEXT,
    detail          TEXT,
    rate_limit_streak INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_due
    ON videos(next_attempt_at)
    WHERE terminal_state IS NULL;
"""

# Queue files created before rate_limit_streak existed need the column added.
# The ALTER is attempted on every open; a "duplicate column name" error just
# means the column is already present, which is the state we want.
_MIGRATIONS = (
    "ALTER TABLE videos ADD COLUMN rate_limit_streak INTEGER NOT NULL DEFAULT 0",
)


class TerminalState(StrEnum):
    """Why a queued video is no longer pending.

    DONE means the transcript was fetched and delivered. The GIVE_UP_* values
    record the three ways the worker stops trying: the 48 hour cutoff with no
    captions ever seen, a video that is gone or bot-checked, and a 429 budget
    that has been exhausted.
    """

    DONE = "DONE"
    GIVE_UP_NEVER = "GIVE_UP_NEVER"
    GIVE_UP_UNAVAILABLE = "GIVE_UP_UNAVAILABLE"
    GIVE_UP_RATE_LIMITED_DEAD = "GIVE_UP_RATE_LIMITED_DEAD"


@dataclass(frozen=True, slots=True)
class QueueRecord:
    """One row of the videos table as an immutable record."""

    video_id: str
    url: str
    channel_id: str
    channel_name: str
    title: str
    published: str | None
    enqueued_at: float
    attempts: int
    last_attempt_at: float | None
    next_attempt_at: float
    transcript_path: str | None
    terminal_state: TerminalState | None
    detail: str | None
    rate_limit_streak: int


class QueueStore:
    """SQLite-backed queue with WAL durability.

    Threading rule: all calls happen on the event-loop thread only. The worker
    calls store methods before/after asyncio.to_thread(probe...), never inside.
    SQLite calls here are sub-millisecond; no lock is needed.
    """

    def __init__(self, path: Path) -> None:
        """Open or create the queue at ``path`` and ensure the schema exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._closed = False
        # check_same_thread=False documents that access is single-threaded by
        # convention (see the class docstring), not that concurrent use is safe.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # Row access by column name keeps the row -> dataclass mapping readable
        # and avoids depending on column order.
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        for statement in _MIGRATIONS:
            try:
                self._conn.execute(statement)
            except sqlite3.OperationalError:
                # The column already exists on a database created with the
                # current schema; nothing to migrate.
                pass
        self._conn.commit()

    def _ensure_open(self) -> None:
        """Raise RuntimeError when the store has already been closed."""
        if self._closed:
            raise RuntimeError("queue store is closed")

    def _row_to_record(self, row: sqlite3.Row) -> QueueRecord:
        """Build a QueueRecord from a videos row, coercing the terminal state."""
        state_value = row["terminal_state"]
        terminal_state: TerminalState | None = None
        if state_value is not None:
            try:
                terminal_state = TerminalState(state_value)
            except ValueError:
                # A value written by a newer version should not crash an older
                # worker; treat it as pending and carry on.
                logger.warning(
                    "unknown terminal_state %r for video %s; treating as pending",
                    state_value,
                    row["video_id"],
                )
        return QueueRecord(
            video_id=row["video_id"],
            url=row["url"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            title=row["title"],
            published=row["published"],
            enqueued_at=row["enqueued_at"],
            attempts=row["attempts"],
            last_attempt_at=row["last_attempt_at"],
            next_attempt_at=row["next_attempt_at"],
            transcript_path=row["transcript_path"],
            terminal_state=terminal_state,
            detail=row["detail"],
            # Defensive: a row read before the column migration ran has no
            # rate_limit_streak, so treat a missing column as a zero streak.
            rate_limit_streak=(
                row["rate_limit_streak"]
                if "rate_limit_streak" in row.keys()
                else 0
            ),
        )

    def enqueue(self, entry: VideoEntry, *, now: float | None = None) -> bool:
        """Insert ``entry`` by video_id; return True on insert, False if seen.

        The first probe is due immediately: ``enqueued_at`` and
        ``next_attempt_at`` are both set to ``now``. The worker applies its own
        pacing between probes, so the queue does not need to delay the first
        lookup. A duplicate video is ignored rather than updated, which keeps
        the original enqueue time and attempt count intact.
        """
        self._ensure_open()
        if now is None:
            now = time.time()
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO videos "
            "(video_id, url, channel_id, channel_name, title, published, "
            " enqueued_at, attempts, next_attempt_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                entry.video_id,
                entry.url,
                entry.channel_id,
                entry.channel_name,
                entry.title,
                entry.published,
                now,
                now,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def next_due(self, now: float | None = None) -> QueueRecord | None:
        """Return the oldest pending record due at or before ``now``, else None."""
        self._ensure_open()
        if now is None:
            now = time.time()
        row = self._conn.execute(
            "SELECT * FROM videos "
            "WHERE terminal_state IS NULL AND next_attempt_at <= ? "
            "ORDER BY next_attempt_at LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def mark_attempt(
        self,
        video_id: str,
        *,
        now: float | None = None,
        next_attempt_at: float | None = None,
        transcript_path: str | None = None,
    ) -> None:
        """Record one probe and optionally reschedule or store the transcript.

        ``attempts`` is incremented and ``last_attempt_at`` is set to ``now`` on
        every call. ``next_attempt_at`` and ``transcript_path`` are updated only
        when a value is provided, so a caller that only wants to count the
        attempt leaves the existing schedule and path untouched.
        """
        self._ensure_open()
        if now is None:
            now = time.time()
        assignments = ["attempts = attempts + 1", "last_attempt_at = ?"]
        params: list[object] = [now]
        if next_attempt_at is not None:
            assignments.append("next_attempt_at = ?")
            params.append(next_attempt_at)
        if transcript_path is not None:
            assignments.append("transcript_path = ?")
            params.append(transcript_path)
        params.append(video_id)
        self._conn.execute(
            f"UPDATE videos SET {', '.join(assignments)} WHERE video_id = ?",
            params,
        )
        self._conn.commit()

    def reschedule(
        self,
        video_id: str,
        *,
        next_attempt_at: float | None = None,
        transcript_path: str | None = None,
        rate_limit_streak: int | None = None,
        now: float | None = None,
    ) -> None:
        """Update schedule, transcript path, or rate-limit streak in place.

        Every argument left as None is left unchanged on the row. This method
        never touches ``attempts`` or ``last_attempt_at``; those belong to
        ``mark_attempt`` and must be incremented exactly once per probe. Use
        ``reschedule`` in the worker's outcome branches so a single probe does
        not count as two attempts.

        ``now`` is accepted for symmetry with ``mark_attempt`` but is unused:
        this method only ever changes the columns named in the arguments.
        """
        self._ensure_open()
        assignments: list[str] = []
        params: list[object] = []
        if next_attempt_at is not None:
            assignments.append("next_attempt_at = ?")
            params.append(next_attempt_at)
        if transcript_path is not None:
            assignments.append("transcript_path = ?")
            params.append(transcript_path)
        if rate_limit_streak is not None:
            assignments.append("rate_limit_streak = ?")
            params.append(rate_limit_streak)
        if not assignments:
            # Nothing to change; skip the UPDATE and the commit entirely.
            return
        params.append(video_id)
        self._conn.execute(
            f"UPDATE videos SET {', '.join(assignments)} WHERE video_id = ?",
            params,
        )
        self._conn.commit()

    def mark_terminal(
        self,
        video_id: str,
        state: TerminalState,
        *,
        detail: str | None = None,
    ) -> None:
        """Stamp ``state`` on a record so next_due stops returning it.

        This never clears an existing terminal state: once a video is resolved
        it stays resolved. Calling twice with the same state is a no-op on the
        second call because the WHERE clause no longer matches.
        """
        self._ensure_open()
        self._conn.execute(
            "UPDATE videos SET terminal_state = ?, detail = ? "
            "WHERE video_id = ? AND terminal_state IS NULL",
            (state.value, detail, video_id),
        )
        self._conn.commit()

    def counts(self) -> dict[str, int]:
        """Return row counts grouped by state, always including ``pending``.

        Keys are ordered with ``pending`` first, then terminal states in their
        declaration order. Any value that is not a known TerminalState is
        appended afterward so an unknown state is visible rather than dropped.
        """
        self._ensure_open()
        rows = self._conn.execute(
            "SELECT terminal_state, COUNT(*) AS n FROM videos GROUP BY terminal_state"
        ).fetchall()
        pending = 0
        grouped: dict[str, int] = {}
        for row in rows:
            state = row["terminal_state"]
            if state is None:
                pending = row["n"]
            else:
                grouped[state] = row["n"]
        result: dict[str, int] = {"pending": pending}
        for state in TerminalState:
            if state.value in grouped:
                result[state.value] = grouped[state.value]
        for state, count in grouped.items():
            if state not in result:
                result[state] = count
        return result

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._conn.close()


def open_store(path: Path) -> QueueStore:
    """Open or create a QueueStore at ``path``."""
    return QueueStore(path)
