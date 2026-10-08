"""Shared factory helpers for the tldw test suite.

These are plain functions (not fixtures) so each test file imports only what
it needs. They live in a top-level module rather than ``conftest.py`` so the
boundary between auto-injected fixtures and explicit helpers stays clear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw.config import Settings
from tldw.feed import VideoEntry


def make_video_entry(
    video_id: str = "dQw4w9WgXcQ", **overrides: object
) -> VideoEntry:
    """Build a VideoEntry with the standard test channel and title pattern.

    The defaults match the values the queue and worker tests have used
    historically, so swapping a local ``_entry`` / ``_video_entry`` helper for
    this call site changes nothing observable.
    """
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
