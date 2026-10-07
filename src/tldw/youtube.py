"""Helpers for building YouTube deep links and readable timestamps.

Both helpers are pure and depend only on the standard library. They live in
their own module because two upcoming modules need them: the LLM module that
turns transcript text into timestamped takeaways, and the Discord module that
renders those takeaways into embeds. Keeping them here means neither module
has to import the other.
"""

from __future__ import annotations


def youtube_deep_link(video_id: str, seconds: int | float) -> str:
    """Return a youtu.be deep link that starts the video at ``seconds``.

    Examples:
        youtube_deep_link("dQw4w9WgXcQ", 0)   -> "https://youtu.be/dQw4w9WgXcQ?t=0"
        youtube_deep_link("dQw4w9WgXcQ", 754) -> "https://youtu.be/dQw4w9WgXcQ?t=754"

    Fractional seconds are truncated because YouTube only accepts whole
    seconds. The ``t=0`` case is kept as-is so every link shares one shape.
    """
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def format_timestamp(seconds: int | float) -> str:
    """Return a human-readable timestamp such as ``12:34`` or ``1:01:01``.

    Examples:
        format_timestamp(0)    -> "0:00"
        format_timestamp(754)  -> "12:34"
        format_timestamp(3661) -> "1:01:01"

    Minutes and seconds are always zero-padded to two digits. The hours field
    appears only when the value reaches one hour or more, and it is never
    padded.
    """
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
