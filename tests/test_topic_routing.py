"""Characterize durable topic routing in the queue."""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from helpers import make_settings, make_takeaways, make_video_entry
from tldw import alerts, worker
from tldw.app import create_app
from tldw.queue import TerminalState, open_store
from tldw.topics import load_topics


@pytest.fixture
def topic_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create two independent topics and their destinations."""
    monkeypatch.setenv("WEBHOOK_ONE", "https://example.test/one")
    monkeypatch.setenv("WEBHOOK_TWO", "https://example.test/two")
    (tmp_path / "one.txt").write_text("first prompt")
    (tmp_path / "two.txt").write_text("second prompt")
    path = tmp_path / "topics.yml"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "topics": {
                    "default": {
                        "channels": [{"channel_id": "UCabcdefghijklmnopqrstuv"}],
                        "prompt": {"file": "one.txt"},
                        "discord": {"webhook_env": "WEBHOOK_ONE"},
                    },
                    "second": {
                        "channels": [{"channel_id": "UCABCDEFGHIJKLMNOPQRSTUV"}],
                        "prompt": {"file": "two.txt"},
                        "discord": {"webhook_env": "WEBHOOK_TWO"},
                    },
                },
            }
        )
    )
    return path


def atom(video: str, channel: str) -> bytes:
    """Create one small Atom delivery."""
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:yt="http://www.youtube.com/xml/schemas/2015"><entry>'
        f"<yt:videoId>{video}</yt:videoId><yt:channelId>{channel}</yt:channelId>"
        f'<title>Title</title><link rel="alternate" href="https://youtube.com/watch?v={video}"/>'
        "<author><name>Feed name</name></author></entry></feed>"
    ).encode()


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


def test_migration_preserves_old_rows(tmp_path: Path) -> None:
    """A genuinely pre-topic database retains artifacts and terminal state."""
    # Arrange
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE videos (video_id TEXT PRIMARY KEY, url TEXT NOT NULL, "
            "channel_id TEXT NOT NULL, channel_name TEXT NOT NULL, title TEXT NOT NULL, "
            "published TEXT, enqueued_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
            "last_attempt_at REAL, next_attempt_at REAL NOT NULL, transcript_path TEXT, "
            "audio_path TEXT, terminal_state TEXT, detail TEXT, "
            "rate_limit_streak INTEGER NOT NULL DEFAULT 0)"
        )
        db.executemany(
            "INSERT INTO videos VALUES (?, 'url', 'channel', 'name', 'title', NULL, "
            "10, 3, 20, 30, 'transcript', 'audio', ?, 'detail', 2)",
            [("pending", None), ("terminal", "GIVE_UP_AUDIO")],
        )
        before = db.execute("SELECT * FROM videos ORDER BY video_id").fetchall()

    # Act
    store = open_store(path)
    store.close()
    store = open_store(path)
    store.close()

    # Assert
    with sqlite3.connect(path) as db:
        after = db.execute("SELECT * FROM videos ORDER BY video_id").fetchall()
    assert [row[:-1] for row in after] == before
    assert [row[-1] for row in after] == ["default", "default"]


@pytest.mark.parametrize("conflict", [False, True])
def test_invalid_config_has_no_startup_side_effects(
    tmp_path: Path, conflict: bool
) -> None:
    """Invalid YAML or a CSV conflict fails before queue and subscription work."""
    # Arrange
    path = tmp_path / "topics.yml"
    path.write_text("invalid")
    settings = make_settings(
        tmp_path,
        topics_file=path,
        callback_url="https://callback.test",
        channel_ids_env_override="UCabcdefghijklmnopqrstuv" if conflict else None,
    )
    calls = []
    transport = httpx2.MockTransport(
        lambda request: calls.append(request) or httpx2.Response(204)
    )

    # Act and assert
    with pytest.raises(ValueError):
        with TestClient(create_app(settings, transport=transport)):
            pass
    assert not settings.queue_file.exists()
    assert not settings.transcript_dir.exists()
    assert not calls


@pytest.mark.parametrize(
    ("override", "conflict"),
    [
        (None, False),
        ("", False),
        (",", False),
        (" , ", False),
        (", UCabcdefghijklmnopqrstuv ,", True),
    ],
)
def test_csv_override_conflict_uses_nonblank_entries(
    tmp_path: Path, topic_file: Path, override: str | None, conflict: bool
) -> None:
    """Only actual CSV entries conflict with an otherwise valid topic document."""
    # Arrange
    settings = make_settings(
        tmp_path,
        topics_file=topic_file,
        channel_ids_env_override=override,
        callback_url=None,
    )
    app = create_app(
        settings, transport=httpx2.MockTransport(lambda _: httpx2.Response(204))
    )

    # Act and assert
    if conflict:
        with pytest.raises(ValueError, match="cannot be combined"):
            with TestClient(app):
                pass
        assert not settings.queue_file.exists()
    else:
        with TestClient(app):
            assert app.state.topics is not None


def test_callback_uses_immutable_topic_snapshot(
    tmp_path: Path, topic_file: Path
) -> None:
    """Callbacks keep original routing even after the YAML file changes."""
    # Arrange
    settings = make_settings(tmp_path, topics_file=topic_file, discord_webhook_url=None)
    app = create_app(
        settings, transport=httpx2.MockTransport(lambda _: httpx2.Response(204))
    )
    first = "UCabcdefghijklmnopqrstuv"
    second = "UCABCDEFGHIJKLMNOPQRSTUV"

    # Act
    with TestClient(app) as client:
        topic_file.write_text("invalid")
        assert (
            client.post("/pubsub/callback", content=atom("one", first)).status_code
            == 200
        )
        assert (
            client.post("/pubsub/callback", content=atom("two", second)).status_code
            == 200
        )
        assert (
            client.post(
                "/pubsub/callback", content=atom("unknown", "UCxxxxxxxxxxxxxxxxxxxxxx")
            ).status_code
            == 200
        )
        assert (
            app.state.queue.next_due(now=1e12, topic_ids=["default"]).video_id == "one"
        )
        assert (
            app.state.queue.next_due(now=1e12, topic_ids=["second"]).video_id == "two"
        )
        assert app.state.queue.counts()["pending"] == 2


async def test_worker_binds_saved_topic_prompt_and_webhook(
    tmp_path: Path, topic_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One serial worker uses each saved route and its own analyzer."""
    # Arrange
    registry = load_topics(topic_file)
    settings = make_settings(
        tmp_path, topics_file=topic_file, discord_webhook_url=None, openai_api_key="key"
    )
    store = open_store(tmp_path / "queue.db")
    transcript = tmp_path / "cached.txt"
    transcript.write_text("transcript")
    for video, topic in [("one", "default"), ("two", "second")]:
        store.enqueue(make_video_entry(video), topic_id=topic)
        store.reschedule(video, transcript_path=str(transcript))
    store.enqueue(make_video_entry("orphan"), topic_id="missing")
    seen = []

    class Analyzer:
        """Record the prompt attached to each production analyzer."""

        def __init__(self, _settings: object, *, system_prompt: str) -> None:
            self.prompt = system_prompt

        async def analyze(self, _text: str, *, video_id: str, title: str):
            """Record the video and return a valid result."""
            seen.append((video_id, self.prompt))
            return make_takeaways()

    sends = []
    iterations = 0

    async def send_embeds(_client, url, _embeds):
        """Record the destination for each message."""
        sends.append(url)

    async def stop(_delay: float) -> None:
        """Stop after two iterations, even if delivery is broken."""
        nonlocal iterations
        iterations += 1
        if iterations == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(worker, "OpenAIAnalyzer", Analyzer)
    # Act
    async with httpx2.AsyncClient() as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings,
                store,
                client,
                registry=registry,
                send_embeds=send_embeds,
                sleep=stop,
            )

    # Assert
    assert seen == [("one", "first prompt"), ("two", "second prompt")]
    assert sends == ["https://example.test/one", "https://example.test/two"]
    orphan = store.next_due()
    assert orphan is not None and orphan.video_id == "orphan"
    store.close()


async def test_callback_restart_reassignment_preserves_saved_route(
    tmp_path: Path, topic_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved routes survive restart while new intake uses reassigned sources."""
    # Arrange
    source = "UCabcdefghijklmnopqrstuv"
    other = "UCABCDEFGHIJKLMNOPQRSTUV"
    settings = make_settings(
        tmp_path,
        topics_file=topic_file,
        callback_url=None,
        discord_webhook_url=None,
        audio_download_delay_seconds=0,
    )
    transport = httpx2.MockTransport(lambda _: httpx2.Response(204))
    app = create_app(settings, transport=transport)
    with TestClient(app) as client:
        assert (
            client.post("/pubsub/callback", content=atom("old", source)).status_code
            == 200
        )
        old = app.state.queue.next_due(topic_ids=["default"])
        assert old is not None and old.video_id == "old"

    document = json.loads(topic_file.read_text())
    document["topics"]["default"]["channels"] = [{"channel_id": other}]
    document["topics"]["second"]["channels"] = [{"channel_id": source}]
    topic_file.write_text(json.dumps(document))
    (tmp_path / "one.txt").write_text("updated first prompt")
    (tmp_path / "two.txt").write_text("updated second prompt")
    monkeypatch.setenv("WEBHOOK_ONE", "https://example.test/updated-one")
    monkeypatch.setenv("WEBHOOK_TWO", "https://example.test/updated-two")
    app = create_app(settings, transport=transport)
    transcript = tmp_path / "cached.txt"
    transcript.write_text("cached transcript")
    with TestClient(app) as client:
        assert (
            client.post("/pubsub/callback", content=atom("new", source)).status_code
            == 200
        )
        new = app.state.queue.next_due(topic_ids=["second"])
        assert new is not None and new.video_id == "new"
        app.state.queue.reschedule("old", transcript_path=str(transcript))
        app.state.queue.reschedule("new", transcript_path=str(transcript))

    registry = load_topics(topic_file)
    store = open_store(settings.queue_file)
    analyzed = []
    requests = []
    iterations = 0

    class Analyzer:
        """Record the configured prompt for each saved route."""

        def __init__(self, _settings: object, *, system_prompt: str) -> None:
            self.prompt = system_prompt

        async def analyze(self, _text: str, *, video_id: str, title: str):
            """Return a valid result and record the chosen prompt."""
            analyzed.append((video_id, self.prompt))
            return make_takeaways()

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Capture each Discord request without a network connection."""
        requests.append((str(request.url), json.loads(request.content)))
        return httpx2.Response(204)

    async def stop(_delay: float) -> None:
        """Bound the worker even if routing or delivery breaks."""
        nonlocal iterations
        iterations += 1
        if iterations == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(worker, "OpenAIAnalyzer", Analyzer)
    worker_settings = settings.model_copy(update={"openai_api_key": "key"})

    # Act
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                worker_settings, store, client, registry=registry, sleep=stop
            )

    # Assert
    assert analyzed == [
        ("old", "updated first prompt"),
        ("new", "updated second prompt"),
    ]
    assert [url.split("?")[0] for url, _ in requests] == [
        "https://example.test/updated-one",
        "https://example.test/updated-two",
    ]
    assert "old" in requests[0][1]["embeds"][0]["url"]
    assert "new" in requests[1][1]["embeds"][0]["url"]
    assert store.counts()["DONE"] == 2
    store.close()


@pytest.mark.parametrize("failure", ["status", "transport"])
async def test_delivery_logs_do_not_expose_webhook(
    tmp_path: Path, topic_file: Path, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """HTTP status and transport failures never log webhook credentials."""
    # Arrange
    secret = "sentinel-webhook-token"
    url = f"https://example.test/{secret}"
    settings = make_settings(
        tmp_path, discord_webhook_url=url, discord_alert_user_id="1" * 18
    )
    store = open_store(tmp_path / "queue.db")
    store.enqueue(make_video_entry("one"))
    store.enqueue_failure_alert("one", "stage", "failed", True)

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Simulate a Discord failure without network access."""
        if failure == "transport":
            raise httpx2.ConnectError(f"failed {request.url}", request=request)
        return httpx2.Response(403)

    # Act
    caplog.set_level(logging.INFO)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        assert not await alerts.flush_failure_alert(settings, store, client)

    # Assert
    assert secret not in caplog.text
    store.close()


@pytest.mark.parametrize("failure", ["status", "transport"])
async def test_nested_delivery_failures_do_not_log_exception_chain(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Persistence errors during webhook handling cannot expose the webhook URL."""
    # Arrange
    secret = "sentinel-chained-webhook-token"
    settings = make_settings(
        tmp_path,
        openai_api_key="key",
        discord_alert_user_id="1" * 18,
        discord_webhook_url=f"https://example.test/{secret}",
    )
    store = open_store(tmp_path / "queue.db")
    store.enqueue(make_video_entry("video"))
    transcript = tmp_path / "cached.txt"
    transcript.write_text("")
    store.reschedule("video", transcript_path=str(transcript))

    def fail_store(*_args: object, **_kwargs: object) -> None:
        """Fail both alert persistence and the delivery reschedule."""
        raise sqlite3.OperationalError("database unavailable")

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Return a status failure or a URL-bearing transport exception."""
        if failure == "transport":
            raise httpx2.ConnectError(f"failed {request.url}", request=request)
        return httpx2.Response(403)

    async def analyze(_text: str, _video: str, _title: str):
        """Provide the required analyzer without contacting an API."""
        return make_takeaways()

    async def stop(_delay: float) -> None:
        """Stop after one iteration regardless of its outcome."""
        raise asyncio.CancelledError

    monkeypatch.setattr(store, "enqueue_failure_alert", fail_store)
    monkeypatch.setattr(store, "reschedule", fail_store)
    caplog.set_level(logging.INFO)

    # Act
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings, store, client, analyze=analyze, sleep=stop
            )

    # Assert
    assert "could not persist failure alert" in caplog.text
    assert "processing raised" in caplog.text
    assert secret not in caplog.text
    store.close()


async def test_disabled_topic_drains_and_bad_alert_does_not_starve_other(
    tmp_path: Path, topic_file: Path
) -> None:
    """Disabled routes drain, while one bad alert destination cools independently."""
    # Arrange
    doc = json.loads(topic_file.read_text())
    doc["topics"]["default"]["enabled"] = False
    topic_file.write_text(json.dumps(doc))
    registry = load_topics(topic_file)
    settings = make_settings(
        tmp_path,
        discord_webhook_url=None,
        openai_api_key="key",
        discord_alert_user_id="1" * 18,
    )
    store = open_store(tmp_path / "queue.db")
    for video, route in [("bad", "default"), ("good", "second")]:
        store.enqueue(make_video_entry(video), topic_id=route)
        store.enqueue_failure_alert(video, "stage", "failure", False)
        store.mark_terminal(video, TerminalState.GIVE_UP_NEVER)
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Fail only the first topic's destination."""
        requests.append(str(request.url))
        return httpx2.Response(403 if "/one?" in str(request.url) else 204)

    # Act
    cooldowns: dict[str, float] = {}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        assert not await alerts.flush_failure_alert(
            settings, store, client, registry=registry, cooldowns=cooldowns
        )
        assert await alerts.flush_failure_alert(
            settings, store, client, registry=registry, cooldowns=cooldowns
        )

    # Assert
    assert [url.split("?")[0] for url in requests] == [
        "https://example.test/one",
        "https://example.test/two",
    ]
    assert store.next_failure_alert(topic_ids=["second"]) is None
    assert store.next_failure_alert(topic_ids=["default"]) is not None
    store.close()


def test_disabled_only_snapshot_drops_every_callback(
    tmp_path: Path, topic_file: Path
) -> None:
    """An empty active index drops both disabled and unknown sources."""
    # Arrange
    doc = json.loads(topic_file.read_text())
    for topic in doc["topics"].values():
        topic["enabled"] = False
    topic_file.write_text(json.dumps(doc))
    app = create_app(
        make_settings(tmp_path, topics_file=topic_file),
        transport=httpx2.MockTransport(lambda _: httpx2.Response(204)),
    )

    # Act and assert
    with TestClient(app) as client:
        assert (
            client.post(
                "/pubsub/callback", content=atom("disabled", "UCabcdefghijklmnopqrstuv")
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/pubsub/callback", content=atom("unknown", "UCxxxxxxxxxxxxxxxxxxxxxx")
            ).status_code
            == 200
        )
        assert app.state.queue.counts()["pending"] == 0


async def test_legacy_does_not_drain_nondefault_and_repair_keeps_expiry(
    tmp_path: Path, topic_file: Path
) -> None:
    """Unsetting YAML cannot reroute a job; restoring it keeps its original age."""
    # Arrange
    settings = make_settings(
        tmp_path,
        openai_api_key="key",
        discord_webhook_url="https://example.test/legacy",
    )
    store = open_store(tmp_path / "queue.db")
    store.enqueue(
        make_video_entry("old"),
        now=time.time() - settings.giveup_seconds - 1,
        topic_id="second",
    )
    sleeps = 0

    async def stop(_delay: float) -> None:
        """Stop the worker after one iteration."""
        nonlocal sleeps
        sleeps += 1
        raise asyncio.CancelledError

    async def analyze(_text: str, _video: str, _title: str):
        """Return a result without contacting the model."""
        return make_takeaways()

    # Act
    async with httpx2.AsyncClient() as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings, store, client, sleep=stop, analyze=analyze
            )
        record = store.next_due()
        assert record is not None and record.attempts == 0
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings,
                store,
                client,
                sleep=stop,
                analyze=analyze,
                registry=load_topics(topic_file),
            )

    # Assert
    assert sleeps == 2
    assert store.next_due() is None
    assert store.counts()["GIVE_UP_NEVER"] == 1
    store.close()


async def test_only_disabled_route_drains_plain_fallback_and_terminal_alert(
    tmp_path: Path, topic_file: Path
) -> None:
    """Retained disabled routes deliver cached fallback and terminal alerts."""
    # Arrange
    doc = json.loads(topic_file.read_text())
    doc["topics"] = {"default": doc["topics"]["default"]}
    doc["topics"]["default"]["enabled"] = False
    topic_file.write_text(json.dumps(doc))
    registry = load_topics(topic_file)
    settings = make_settings(
        tmp_path,
        discord_webhook_url=None,
        openai_api_key="key",
        discord_alert_user_id="1" * 18,
    )
    store = open_store(tmp_path / "queue.db")
    transcript = tmp_path / "cached.txt"
    transcript.write_text("some transcript")
    store.enqueue(make_video_entry("cached"), topic_id="default")
    store.reschedule("cached", transcript_path=str(transcript))
    store.enqueue(make_video_entry("terminal"), topic_id="default")
    store.mark_terminal("terminal", TerminalState.GIVE_UP_NEVER)
    store.enqueue_failure_alert("terminal", "download", "failed", False)
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Capture Discord calls without network access."""
        requests.append((str(request.url), json.loads(request.content)))
        return httpx2.Response(204)

    async def fail_analysis(_text: str, _video: str, _title: str):
        """Force the existing plain digest fallback."""
        raise TimeoutError("model unavailable")

    async def stop(_delay: float) -> None:
        """Stop after the first iteration."""
        raise asyncio.CancelledError

    # Act
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings,
                store,
                client,
                registry=registry,
                analyze=fail_analysis,
                sleep=stop,
            )

    # Assert
    assert len(requests) == 2
    assert all(url.startswith("https://example.test/one?") for url, _ in requests)
    assert "content" in requests[0][1] and "content" in requests[1][1]
    assert store.counts()["DONE"] == 1
    assert (
        store.next_failure_alert(topic_ids=["default"]) is not None
    )  # analysis alert remains
    store.close()


async def test_failed_summary_does_not_block_other_topic(
    tmp_path: Path, topic_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad webhook reschedules only its video and never logs its token."""
    # Arrange
    doc = json.loads(topic_file.read_text())
    doc["topics"]["default"]["discord"]["webhook_env"] = "SECRET_WEBHOOK"
    topic_file.write_text(json.dumps(doc))
    registry = load_topics(
        topic_file,
        environ={
            "SECRET_WEBHOOK": "https://example.test/sentinel-webhook-token",
            "WEBHOOK_TWO": "https://example.test/two",
        },
    )
    settings = make_settings(tmp_path, openai_api_key="key", discord_webhook_url=None)
    store = open_store(tmp_path / "queue.db")
    transcript = tmp_path / "cached.txt"
    transcript.write_text("")
    for video, topic in [("bad", "default"), ("good", "second")]:
        store.enqueue(make_video_entry(video), topic_id=topic)
        store.reschedule(video, transcript_path=str(transcript))
    requests = []
    iterations = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Reject the first destination and accept the second."""
        requests.append(str(request.url))
        return httpx2.Response(403 if "sentinel" in str(request.url) else 204)

    async def analyze(_text: str, _video: str, _title: str):
        """Provide the required injected analyzer."""
        return make_takeaways()

    async def stop(_delay: float) -> None:
        """Stop after two iterations, even if delivery is broken."""
        nonlocal iterations
        iterations += 1
        if iterations == 2:
            raise asyncio.CancelledError

    # Act
    caplog.set_level(logging.INFO)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(asyncio.CancelledError):
            await worker.transcript_loop(
                settings, store, client, registry=registry, analyze=analyze, sleep=stop
            )

    # Assert
    assert len(requests) == 2
    assert store.counts() == {"pending": 1, "DONE": 1}
    assert "sentinel-webhook-token" not in caplog.text
    store.close()
