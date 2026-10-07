"""Console rendering helpers for YouTube video notifications.

This module turns a parsed VideoEntry into a single, human-readable line so the
program can print notifications without leaking formatting concerns into the
feed parser or the networking code. It performs no I/O and adds no newlines;
the caller decides when to print.
"""

from __future__ import annotations

import re

from tldw.feed import VideoEntry

# Collapse any run of whitespace (including embedded newlines and carriage
# returns) into a single space so the rendered line always stays on one line.
_WHITESPACE_RUN = re.compile(r"\s+")


def format_video_line(entry: VideoEntry) -> str:
    """Render one VideoEntry as a single console-friendly line.

    The title is sanitized first: embedded newlines and carriage returns become
    spaces, runs of whitespace collapse to one space, and the surrounding
    whitespace is trimmed. The url and channel_name are used verbatim. The
    returned string has no trailing newline.
    """
    title = _WHITESPACE_RUN.sub(" ", entry.title).strip()
    return f"[YouTube] {entry.channel_name}: {title} ({entry.url})"
