"""Characterize durable topic routing in the queue."""

from pathlib import Path

from helpers import make_video_entry
from tldw.queue import TerminalState, open_store


def test_duplicate_keeps_original_topic_and_filters_due(tmp_path: Path) -> None:
    """Duplicate notifications cannot change a saved route."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    entry = make_video_entry("video")

    # Act
    assert store.enqueue(entry, now=1, topic_id="first")
    assert not store.enqueue(entry, now=2, topic_id="second")

    # Assert
    assert store.next_due(now=3, topic_ids=[]) is None
    assert store.next_due(now=3, topic_ids=["second"]) is None
    record = store.next_due(now=3, topic_ids=["first"])
    assert record is not None and record.topic_id == "first"
    store.close()


def test_terminal_alert_keeps_topic_and_is_filterable(tmp_path: Path) -> None:
    """An alert remains routable after its video becomes terminal."""
    # Arrange
    store = open_store(tmp_path / "queue.sqlite3")
    store.enqueue(make_video_entry("video"), now=1, topic_id="first")
    store.enqueue_failure_alert("video", "download", "failed", False, now=2)
    store.mark_terminal("video", TerminalState.GIVE_UP_NEVER)

    # Act
    alert = store.next_failure_alert(topic_ids=["first"])

    # Assert
    assert store.next_failure_alert(topic_ids=[]) is None
    assert store.next_failure_alert(topic_ids=["second"]) is None
    assert alert is not None and alert.topic_id == "first"
    store.close()
