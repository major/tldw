"""Shared pytest fixtures for the tldw test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw.queue import QueueStore, open_store

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# A tiny two-entry Atom feed that mirrors the real YouTube shape. Hand-rolled so
# tests do not depend on the large captured payload when checking multi-entry
# behavior.
_MULTI_ENTRY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/">
  <title>Fixture Channel</title>
  <entry>
    <yt:videoId>v_Fixture1AAAAA</yt:videoId>
    <yt:channelId>UC_FIXTURE_AAAAAAAAAAAAAAA</yt:channelId>
    <title>Fixture Video One</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=v_Fixture1AAAAA"/>
    <author>
      <name>Fixture Channel</name>
    </author>
  </entry>
  <entry>
    <yt:videoId>v_Fixture2AAAAA</yt:videoId>
    <yt:channelId>UC_FIXTURE_AAAAAAAAAAAAAAA</yt:channelId>
    <title>Fixture Video Two</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=v_Fixture2AAAAA"/>
    <author>
      <name>Fixture Channel</name>
    </author>
  </entry>
</feed>
"""


@pytest.fixture
def real_atom_payload() -> bytes:
    """Return the captured Google for Developers YouTube Atom feed bytes."""
    return (FIXTURES_DIR / "yt_atom_sample.xml").read_bytes()


@pytest.fixture
def multi_entry_payload() -> bytes:
    """Return a small Atom feed containing two distinct video entries."""
    return _MULTI_ENTRY_XML.encode("utf-8")


@pytest.fixture
def empty_atom_payload() -> bytes:
    """Return a minimal valid Atom feed with no entries (test_feed shape)."""
    return b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"


@pytest.fixture
def empty_atom_envelope() -> bytes:
    """Return a minimal Atom envelope in the form the FastAPI app posts (test_app shape)."""
    return (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
    )


@pytest.fixture
def malformed_atom_payload() -> bytes:
    """Return a malformed Atom document (unclosed tag) shared by both test files."""
    return b"<feed><entry>"


@pytest.fixture
def missing_link_atom_payload() -> bytes:
    """Return an Atom entry with no <link> element to exercise the link-missing path."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:yt="http://www.youtube.com/xml/schemas/2015">'
        "<entry>"
        "<yt:videoId>v_NoLinkAAAAAA</yt:videoId>"
        "<title>No Link Here</title>"
        "<author><name>Fixture Channel</name></author>"
        "</entry>"
        "</feed>"
    ).encode("utf-8")


@pytest.fixture
def media_title_atom_payload() -> bytes:
    """Return an Atom entry whose title comes from the media: namespace."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:media="http://search.yahoo.com/mrss/">'
        "<entry>"
        "<title></title>"
        "<media:title>From Media Title</media:title>"
        '<link rel="alternate" href="https://www.youtube.com/watch?v=v_MediaAAAAAA"/>'
        "<author><name>Fixture Channel</name></author>"
        "</entry>"
        "</feed>"
    ).encode("utf-8")


@pytest.fixture
def queue_store(tmp_path: Path) -> Iterator[QueueStore]:
    """Open a real QueueStore backed by a temporary database.

    Used by the worker and worker_takeaways tests. Replaces two byte-for-byte
    identical ``store`` fixtures that previously lived in those files. Tests
    that need a non-default path or a closed-and-reopened store should still
    call ``open_store(...)`` directly with ``tmp_path / "..."``.
    """
    store = open_store(tmp_path / "queue.sqlite3")
    try:
        yield store
    finally:
        store.close()


# Environment variables the Settings model reads. Cleared before every test so a
# developer's shell or a CI job cannot leak a value into the assertions. Single
# source of truth -- test_config.py used to hold its own copy.
#
# Kept in sync with the TLDW_-prefixed fields in src/tldw/config.py. The live
# smoke test in test_worker_takeaways reads TLDW_OPENAI_API_KEY straight from
# os.environ, so it is intentionally not cleared here.
TLDW_ENV_VARS = (
    "TLDW_CALLBACK_URL",
    "TLDW_CHANNELS_FILE",
    "TLDW_CHANNEL_IDS",
    "TLDW_HUB_SECRET",
    "TLDW_DISCORD_WEBHOOK_URL",
    "TLDW_QUEUE_FILE",
    "TLDW_TRANSCRIPT_DIR",
    "TLDW_TRANSCRIPT_LINES",
    "TLDW_POLL_BASE_SECONDS",
    "TLDW_POLL_CAP_SECONDS",
    "TLDW_GIVEUP_SECONDS",
    "TLDW_YTDLP_COOKIES_FILE",
    "TLDW_OPENAI_BASE_URL",
    "TLDW_OPENAI_MODEL",
    "TLDW_LLM_TIMEOUT_SECONDS",
    "TLDW_LLM_MAX_OUTPUT_TOKENS",
    "TLDW_LLM_MAX_INPUT_CHARS",
    "TLDW_TAKEAWAY_MAX_BULLETS",
    "TLDW_AUDIO_DOWNLOAD_DELAY_SECONDS",
    "TLDW_AUDIO_DIR",
    "TLDW_AUDIO_FORMAT",
    "TLDW_AUDIO_BITRATE",
    "TLDW_FFMPEG_TIMEOUT_SECONDS",
    "TLDW_TRANSCRIBE_MODEL",
    "TLDW_TRANSCRIBE_KEYWORDS",
    "TLDW_TRANSCRIBE_TIMEOUT_SECONDS",
    "TLDW_INCLUDE_SHORTS",
)


@pytest.fixture(autouse=True)
def clean_tldw_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every TLDW_-prefixed env var before each test.

    Runs for every test in the suite (autouse) so a stray environment
    variable in a developer's shell or in CI cannot leak into Settings
    construction. Tests that want to set a TLDW_ env var can still do so
    with ``monkeypatch.setenv`` -- this fixture only clears at the start.
    """
    for name in TLDW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
