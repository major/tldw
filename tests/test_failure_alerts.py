"""Offline coverage for durable operational alerts."""

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx2
import pytest
from fastapi import FastAPI

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
    assert store.next_failure_alert() is None
    store.close()


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
    app = FastAPI()
    app.state.queue = queue_store

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
                app,
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
