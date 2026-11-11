"""Offline coverage for durable operational alerts."""

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx2
import openai
import pytest

from helpers import make_capturing_transport, make_settings, make_video_entry
from tldw import alerts, worker
from tldw.audio import DownloadResult, ProbeState
from tldw.config import Settings
from tldw.queue import QueueStore, TerminalState
from tldw.transcribe import TranscribeError
from test_worker_audio import (
    FakeAnalyze,
    FakeCompress,
    FakeDownload,
    FakeTranscribe,
    CountingSend,
    CountingSendEmbeds,
)

USER = "559804419261071363"


def test_outbox_keeps_first_failure_across_reopen(tmp_path: Path) -> None:
    """Only the first failure per stage is stored, including after restart."""
    # Arrange
    path = tmp_path / "queue.db"
    store = QueueStore(path)
    store.enqueue(make_video_entry("video1"), now=1)
    store.enqueue_failure_alert("video1", "download", "first", True, now=2)
    store.enqueue_failure_alert("video1", "download", "second", False, now=3)
    store.close()
    store = QueueStore(path)
    alert = store.next_failure_alert()
    assert alert is not None
    assert (alert.reason, alert.retrying) == ("first", True)
    store.mark_failure_alert_sent(alert.video_id, alert.stage, now=4)
    # Act
    store.enqueue_failure_alert("video1", "download", "third", True, now=5)
    store.close()
    store = QueueStore(path)
    # Assert
    assert store.next_failure_alert() is None
    store.close()


@pytest.mark.parametrize(
    ("state", "detail", "expected"),
    [
        (TerminalState.DONE, None, "Video summary delivered successfully."),
        (
            TerminalState.DONE,
            "llm_fallback",
            "Delivered using a plain transcript fallback.",
        ),
        (TerminalState.GIVE_UP_AUDIO, None, "Stopped; no automatic retry."),
        (
            TerminalState.SKIPPED_LIVE,
            None,
            "Skipped because live events are excluded; no automatic retry.",
        ),
        (None, None, "Video remains queued for processing."),
    ],
)
async def test_delayed_alert_reports_current_outcome(
    tmp_path: Path,
    queue_store: QueueStore,
    state: TerminalState | None,
    detail: str | None,
    expected: str,
) -> None:
    """Delayed alerts distinguish the failure time from the current outcome."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_alert_user_id=USER,
        discord_webhook_url="https://discord.test/hook",
    )
    queue_store.enqueue(make_video_entry("video1"), now=1)
    queue_store.enqueue_failure_alert("video1", "analysis", "Failed.", True, now=2)
    if state is not None:
        queue_store.mark_terminal("video1", state, detail=detail)
    requests: list[httpx2.Request] = []
    transport = make_capturing_transport(requests, [httpx2.Response(204)])
    # Act
    async with httpx2.AsyncClient(transport=transport) as client:
        assert await alerts.flush_failure_alert(settings, queue_store, client)
    # Assert
    content = json.loads(requests[0].content)["content"]
    assert "1970-01-01 00:00:02 UTC" in content
    assert expected in content
    assert "Automatic retry scheduled" not in content
    assert "Requires attention" not in content
    assert len(content) <= 2000


async def test_cancelled_alert_delivery_remains_pending(
    tmp_path: Path, queue_store: QueueStore
) -> None:
    """Cancellation during HTTP delivery leaves the alert pending."""
    # Arrange
    settings = make_settings(
        tmp_path,
        discord_alert_user_id=USER,
        discord_webhook_url="https://discord.test/hook",
    )
    queue_store.enqueue(make_video_entry("video1"), now=1)
    queue_store.enqueue_failure_alert("video1", "download", "Failed.", True, now=2)

    async def cancel(_request: httpx2.Request) -> httpx2.Response:
        """Cancel the in-flight request immediately."""
        raise asyncio.CancelledError

    # Act
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(cancel)) as client:
        with pytest.raises(asyncio.CancelledError):
            await alerts.flush_failure_alert(settings, queue_store, client)
    # Assert
    assert queue_store.next_failure_alert() is not None


async def test_worker_alert_failure_cools_down(
    tmp_path: Path, queue_store: QueueStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed notification is not retried during the 60-second cooldown."""
    # Arrange
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    clock = [100.0]
    attempts: list[float] = []
    sleeps = 0

    async def fail(*args: object, **kwargs: object) -> bool:
        """Record a failed alert attempt."""
        attempts.append(clock[0])
        return False

    async def advance(_delay: float) -> None:
        """Advance without waiting and stop after two iterations."""
        nonlocal sleeps
        sleeps += 1
        clock[0] += 59
        if sleeps == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(worker, "flush_failure_alert", fail)
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    # Act
    async with httpx2.AsyncClient() as client:
        with pytest.raises(asyncio.CancelledError):
            await worker._run_loop(
                queue_store,
                client,
                settings,
                sleep=advance,
                download_audio=FakeDownload([]),
                compress_audio=FakeCompress(),
                transcribe=FakeTranscribe(),
                analyze=FakeAnalyze(),
                send=CountingSend(),
                send_embeds=CountingSendEmbeds(),
            )
    # Assert
    assert attempts == [100.0]


def test_existing_database_gains_empty_outbox(tmp_path: Path) -> None:
    """Opening a pre-outbox database adds the table without backfilling."""
    path = tmp_path / "queue.db"
    store = QueueStore(path)
    store.enqueue(make_video_entry("old"), now=1)
    store.mark_terminal("old", TerminalState.GIVE_UP_AUDIO)
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE failure_alerts")
    store = QueueStore(path)
    assert store.next_failure_alert() is None
    store.close()


@pytest.mark.parametrize("value", ["123", "x" * 18, "1" * 21])
def test_alert_user_id_rejects_invalid_values(value: str) -> None:
    """Reject invalid Discord snowflakes at configuration time."""
    with pytest.raises(ValueError):
        Settings(discord_alert_user_id=value)


@pytest.mark.parametrize(
    ("status", "code", "expected", "retrying"),
    [
        (429, "insufficient_quota", "check billing", True),
        (400, None, "HTTP 400", False),
    ],
)
async def test_transcription_failure_is_queued(
    tmp_path: Path,
    queue_store: QueueStore,
    status: int,
    code: str | None,
    expected: str,
    retrying: bool,
) -> None:
    """Quota and permanent errors produce safe alerts with the right disposition."""
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    queue_store.enqueue(make_video_entry("video1"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None
    await worker._transcribe_stage(
        tmp_path / "audio",
        record,
        settings,
        queue_store,
        transcribe=FakeTranscribe(
            raises=TranscribeError("secret", status_code=status, error_code=code)
        ),
        now=1,
    )
    alert = queue_store.next_failure_alert()
    assert alert is not None
    assert expected in alert.reason and "secret" not in alert.reason
    assert alert.retrying is retrying


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("duration_limit", "Audio exceeds the provider duration limit."),
        ("secret-category", None),
    ],
)
async def test_transcription_reason_is_safe_in_alert_detail_and_log(
    tmp_path: Path,
    queue_store: QueueStore,
    caplog: pytest.LogCaptureFixture,
    category: str,
    expected: str | None,
) -> None:
    """Persist and log only the allowlisted reason, never exception text."""
    # Arrange
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    queue_store.enqueue(make_video_entry("video1"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None
    error = TranscribeError(
        "secret @everyone https://secret.test", status_code=400, reason=category
    )
    # Act
    await worker._transcribe_stage(
        tmp_path / "audio",
        record,
        settings,
        queue_store,
        transcribe=FakeTranscribe(raises=error),
        now=1,
    )
    # Assert
    alert = queue_store.next_failure_alert()
    assert alert is not None
    detail = queue_store._conn.execute(
        "SELECT detail FROM videos WHERE video_id = ?", ("video1",)
    ).fetchone()[0]
    for text in (alert.reason, detail, caplog.text):
        assert "secret" not in text and "@everyone" not in text
        if expected:
            assert expected in text
    assert "400" in caplog.text


async def test_fallback_api_error_never_persists_provider_text(
    tmp_path: Path, queue_store: QueueStore
) -> None:
    """The direct SDK error fallback stores a fixed terminal detail."""
    # Arrange
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    queue_store.enqueue(make_video_entry("video1"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None
    error = openai.APIError(
        "secret", request=httpx2.Request("POST", "https://example.test"), body=None
    )
    # Act
    await worker._transcribe_stage(
        tmp_path / "audio",
        record,
        settings,
        queue_store,
        transcribe=FakeTranscribe(raises=error),
        now=1,
    )
    # Assert
    detail = queue_store._conn.execute(
        "SELECT detail FROM videos WHERE video_id = ?", ("video1",)
    ).fetchone()[0]
    assert detail == "Transcription API failed."


async def test_quota_alert_takes_precedence_over_safe_reason(
    tmp_path: Path, queue_store: QueueStore
) -> None:
    """Billing guidance wins over audio-specific guidance for quota errors."""
    # Arrange
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    queue_store.enqueue(make_video_entry("video1"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None
    error = TranscribeError(
        "secret",
        status_code=429,
        error_code="insufficient_quota",
        reason="duration_limit",
    )
    # Act
    await worker._transcribe_stage(
        tmp_path / "audio",
        record,
        settings,
        queue_store,
        transcribe=FakeTranscribe(raises=error),
        now=1,
    )
    # Assert
    alert = queue_store.next_failure_alert()
    assert alert is not None
    assert "check billing" in alert.reason and "duration" not in alert.reason


async def test_failed_delivery_then_idle_worker_flush(
    tmp_path: Path, queue_store: QueueStore
) -> None:
    """A failed HTTP send stays pending and the idle worker sends it later."""
    settings = make_settings(
        tmp_path,
        discord_alert_user_id=USER,
        discord_webhook_url="https://discord.test/hook",
    )
    queue_store.enqueue(make_video_entry("video1", title="@everyone"), now=1)
    queue_store.mark_terminal("video1", TerminalState.GIVE_UP_AUDIO)
    queue_store.enqueue_failure_alert(
        "video1", "transcription", "Failed.", False, now=2
    )
    requests: list[httpx2.Request] = []
    transport = make_capturing_transport(
        requests, [httpx2.Response(500), httpx2.Response(204)]
    )

    async def stop(_delay: float) -> None:
        """Stop after the idle iteration without waiting."""
        raise asyncio.CancelledError

    async with httpx2.AsyncClient(transport=transport) as client:
        assert not await alerts.flush_failure_alert(
            settings, queue_store, client, now=3
        )
        assert queue_store.next_failure_alert() is not None
        with pytest.raises(asyncio.CancelledError):
            await worker._run_loop(
                queue_store,
                client,
                settings,
                sleep=stop,
                download_audio=FakeDownload([]),
                compress_audio=FakeCompress(),
                transcribe=FakeTranscribe(),
                analyze=FakeAnalyze(),
                send=CountingSend(),
                send_embeds=CountingSendEmbeds(),
            )

    assert queue_store.next_failure_alert() is None
    payload = json.loads(requests[-1].content)
    assert payload["content"].startswith(f"<@{USER}>")
    assert payload["allowed_mentions"] == {"parse": [], "users": [USER]}
    assert str(requests[-1].url).endswith("?wait=true")


async def test_not_ready_and_old_terminal_are_silent(
    tmp_path: Path, queue_store: QueueStore
) -> None:
    """Normal waiting and historical terminal records do not create alerts."""
    settings = make_settings(tmp_path, discord_alert_user_id=USER, audio_dir=tmp_path)
    queue_store.enqueue(make_video_entry("old"), now=1)
    queue_store.mark_terminal("old", TerminalState.GIVE_UP_AUDIO)
    queue_store.enqueue(make_video_entry("new"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None
    await worker._download_stage(
        record,
        settings,
        queue_store,
        download_audio=FakeDownload(
            [DownloadResult(state=ProbeState.NOT_READY, path=None, detail=None)]
        ),
        now=1,
    )
    assert queue_store.next_failure_alert() is None


async def test_outbox_failure_does_not_change_video_outcome(
    tmp_path: Path, queue_store: QueueStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken outbox cannot prevent a permanent failure from resolving."""
    settings = make_settings(tmp_path, discord_alert_user_id=USER)
    queue_store.enqueue(make_video_entry("video1"), now=1)
    record = queue_store.next_due(now=1)
    assert record is not None

    def broken(*args: object, **kwargs: object) -> None:
        """Simulate an outbox write failure."""
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(queue_store, "enqueue_failure_alert", broken)
    await worker._transcribe_stage(
        tmp_path / "audio",
        record,
        settings,
        queue_store,
        transcribe=FakeTranscribe(raises=TranscribeError("bad", status_code=400)),
        now=1,
    )
    assert queue_store.next_due(now=999999) is None
    assert queue_store.counts()[TerminalState.GIVE_UP_AUDIO] == 1
