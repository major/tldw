"""Configuration for the tldw PubSubHubbub subscriber.

Settings are read from environment variables in the TLDW_ namespace. The
channel list has two sources: the TLDW_CHANNEL_IDS environment variable and the
JSON file named by TLDW_CHANNELS_FILE. The environment variable wins when it
holds at least one non-blank id; otherwise the file is used. When neither
source yields ids the resolver returns an empty list and logs a warning so the
service can still start and an operator can fix the input.

Every channel id is validated against the YouTube channel id shape: "UC"
followed by 22 characters from [A-Za-z0-9_-]. Invalid ids raise ValueError
listing all of the offenders at once so the operator can fix them in one pass.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "resolve_channel_ids"]

logger = logging.getLogger(__name__)

# YouTube channel ids are "UC" followed by 22 characters from [A-Za-z0-9_-].
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


class Settings(BaseSettings):
    """Runtime settings loaded from the TLDW_ environment namespace.

    Environment names:
        TLDW_CALLBACK_URL: URL the hub calls to deliver notifications.
        TLDW_CHANNELS_FILE: path to the JSON channel id list.
        TLDW_CHANNEL_IDS: comma-separated channel ids that override the file.
        TLDW_HUB_SECRET: HMAC secret for signed hub deliveries.
    """

    model_config = SettingsConfigDict(
        env_prefix="TLDW_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    callback_url: str | None = None
    # pydantic-settings does not apply env_prefix to a field alias, so the two
    # aliased fields carry the full TLDW_ names to match the documented env
    # vars. populate_by_name lets callers still pass the Python field names.
    channel_ids_file: Path = Field(
        default=Path("channels.json"), alias="TLDW_CHANNELS_FILE"
    )
    channel_ids_env_override: str | None = Field(
        default=None, alias="TLDW_CHANNEL_IDS"
    )
    hub_secret: str | None = None


def _validate_channel_ids(channel_ids: list[str]) -> list[str]:
    """Return the channel ids unchanged, or raise ValueError for invalid ones.

    The error lists every invalid id at once so a single run reveals all of the
    fixes needed instead of failing one id at a time.
    """
    invalid = [cid for cid in channel_ids if not _CHANNEL_ID_RE.match(cid)]
    if invalid:
        raise ValueError("invalid YouTube channel id(s): " + ", ".join(invalid))
    return channel_ids


def resolve_channel_ids(settings: Settings) -> list[str]:
    """Resolve and validate the channel ids from the env override or the file.

    Precedence: when TLDW_CHANNEL_IDS holds at least one non-blank id it is
    used and the file is ignored. Whitespace around each comma-separated id is
    trimmed and blank entries are dropped. Otherwise the JSON file named by
    TLDW_CHANNELS_FILE is read: it must be an object with a "channel_ids" list
    of strings, and other keys are ignored. When the file is missing a warning
    naming the path is logged and an empty list is returned.

    Raises:
        ValueError: when any resolved id does not match the YouTube channel id
            shape. The message lists every invalid id.
    """
    env_value = settings.channel_ids_env_override
    if env_value is not None:
        candidates = [part.strip() for part in env_value.split(",")]
        candidates = [part for part in candidates if part]
        if candidates:
            return _validate_channel_ids(candidates)

    if settings.channel_ids_file.exists():
        raw = json.loads(settings.channel_ids_file.read_text(encoding="utf-8"))
        file_ids = raw.get("channel_ids", [])
        return _validate_channel_ids(list(file_ids))

    logger.warning(
        "channel id file %s does not exist and TLDW_CHANNEL_IDS is not set",
        settings.channel_ids_file,
    )
    return []
