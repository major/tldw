"""Parsing for YouTube Atom pubsub notification feeds.

The module turns the XML that YouTube sends through PubSubHubbub into plain
VideoEntry objects so the rest of the program never touches namespaces or
ElementTree directly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

# Single source of truth for the three namespaces a YouTube feed uses. Keeping
# them in one dict avoids scattering namespace string literals across the code.
NS: dict[str, str] = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


@dataclass(frozen=True, slots=True)
class VideoEntry:
    """A single video notification from a YouTube Atom feed."""

    video_id: str
    channel_id: str
    title: str
    url: str
    channel_name: str
    published: str | None
    updated: str | None


def _text_or_none(element: ET.Element | None) -> str | None:
    """Return stripped element text, or None when absent or blank."""
    if element is None or element.text is None:
        return None
    stripped = element.text.strip()
    return stripped or None


def _entry_url(entry: ET.Element) -> str:
    """Return the entry's watch URL, preferring the alternate link."""
    atom = NS["atom"]
    links = entry.findall(f"{{{atom}}}link")
    if not links:
        raise ValueError("entry is missing an Atom <link> element")
    alternate = next((link for link in links if link.get("rel") == "alternate"), None)
    chosen = alternate if alternate is not None else links[0]
    href = chosen.get("href")
    if href is None:
        raise ValueError("entry <link> is missing its href attribute")
    return href


def _parse_entry(entry: ET.Element) -> VideoEntry:
    """Build one VideoEntry from a single Atom <entry> element."""
    atom = NS["atom"]
    yt = NS["yt"]
    media = NS["media"]

    atom_title = _text_or_none(entry.find(f"{{{atom}}}title"))
    # media:title lives inside media:group, so search descendants.
    media_title = _text_or_none(entry.find(f".//{{{media}}}title"))

    return VideoEntry(
        video_id=_text_or_none(entry.find(f"{{{yt}}}videoId")) or "",
        channel_id=_text_or_none(entry.find(f"{{{yt}}}channelId")) or "",
        title=atom_title or media_title or "",
        url=_entry_url(entry),
        channel_name=_text_or_none(entry.find(f"{{{atom}}}author/{{{atom}}}name")) or "",
        published=_text_or_none(entry.find(f"{{{atom}}}published")),
        updated=_text_or_none(entry.find(f"{{{atom}}}updated")),
    )


def parse_atom(body: bytes | str) -> list[VideoEntry]:
    """Parse a YouTube Atom feed body into a list of VideoEntry objects.

    Accepts raw bytes or an already decoded string. Malformed XML raises the
    underlying xml.etree.ElementTree.ParseError. A feed with no entries yields
    an empty list.
    """
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    root = ET.fromstring(body)
    return [_parse_entry(entry) for entry in root.findall(f"{{{NS['atom']}}}entry")]
