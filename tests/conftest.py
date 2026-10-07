"""Shared pytest fixtures for the tldw test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

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
