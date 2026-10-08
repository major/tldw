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

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "resolve_channel_ids"]

logger = logging.getLogger(__name__)

# YouTube channel ids are "UC" followed by 22 characters from [A-Za-z0-9_-].
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

# OpenAI's /v1/audio/transcriptions endpoint accepts this exact set of input
# containers. Rejecting unknown values at startup prevents a typo (e.g. "ogg" or
# "opus") from turning into a 400 on the first video. Sourced from the OpenAI
# Speech-to-Text guide; the union covers both the docs and the SDK docstring.
_OPENAI_AUDIO_FORMATS: frozenset[str] = frozenset(
    {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}
)


class Settings(BaseSettings):
    """Runtime settings loaded from the TLDW_ environment namespace.

    Environment names:
        TLDW_CALLBACK_URL: URL the hub calls to deliver notifications.
        TLDW_CHANNELS_FILE: path to the JSON channel id list.
        TLDW_CHANNEL_IDS: comma-separated channel ids that override the file.
        TLDW_HUB_SECRET: HMAC secret for signed hub deliveries.
        TLDW_DISCORD_WEBHOOK_URL: webhook URL for transcript-to-Discord delivery.
        TLDW_QUEUE_FILE: path to the SQLite queue database.
        TLDW_TRANSCRIPT_DIR: directory where the audio pipeline caches its `.txt` transcripts.
        TLDW_TRANSCRIPT_LINES: how many transcript lines to send per message.
        TLDW_POLL_BASE_SECONDS: first retry delay for the worker, in seconds.
        TLDW_POLL_CAP_SECONDS: maximum retry delay for the worker, in seconds.
        TLDW_GIVEUP_SECONDS: stop retrying a video after this many seconds.
        TLDW_YTDLP_COOKIES_FILE: optional Netscape-format cookies file.
        TLDW_OPENAI_API_KEY: API key for the OpenAI LLM; unset disables takeaways.
        TLDW_OPENAI_BASE_URL: base URL for the OpenAI-compatible endpoint.
        TLDW_OPENAI_MODEL: model name to request from the endpoint.
        TLDW_LLM_TIMEOUT_SECONDS: per-call timeout for a takeaway request.
        TLDW_LLM_MAX_OUTPUT_TOKENS: max tokens the takeaway model may generate.
        TLDW_LLM_MAX_INPUT_CHARS: hard cap on transcript characters sent to the LLM.
        TLDW_TAKEAWAY_MAX_BULLETS: max bullets kept per takeaway item.
        TLDW_AUDIO_DOWNLOAD_DELAY_SECONDS: delay before the first audio download.
        TLDW_AUDIO_DIR: directory for raw audio downloads and compressed artifacts.
        TLDW_AUDIO_FORMAT: output container for ffmpeg; must be in OpenAI's accepted set.
        TLDW_AUDIO_BITRATE: target bitrate for ffmpeg Opus encoding.
        TLDW_FFMPEG_TIMEOUT_SECONDS: per-call ffmpeg timeout.
        TLDW_TRANSCRIBE_MODEL: OpenAI speech-to-text model name.
        TLDW_TRANSCRIBE_LANGS: JSON list of ISO-639-1 language hints for transcription.
        TLDW_TRANSCRIBE_TIMEOUT_SECONDS: per-call transcription timeout.
        TLDW_INCLUDE_SHORTS: when true, deliver YouTube Shorts too. Default false.
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

    # Discord webhook delivery.
    discord_webhook_url: str | None = None

    # Queue storage.
    queue_file: Path = Path("queue.sqlite3")

    # Transcript download directory.
    transcript_dir: Path = Path("transcripts")

    # How many lines of the transcript to send to Discord.
    transcript_lines: int = 10

    # Worker pacing.
    poll_base_seconds: float = 600.0  # 10 minutes
    poll_cap_seconds: float = 3600.0  # 1 hour
    # ge=0 rejects a negative window at startup instead of silently never
    # giving up. Pydantic raises a clear ValidationError naming the field.
    giveup_seconds: float = Field(default=172800.0, ge=0)  # 48 hours

    # Optional yt-dlp cookies (Netscape format) for reliability.
    ytdlp_cookies_file: Path | None = None

    # LLM video takeaways (OpenAI). An unset API key turns the whole feature
    # off; there is no separate enable flag. The base URL defaults to the
    # public OpenAI endpoint and the model defaults to ``gpt-6.1-sol``.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-6.1-sol"
    llm_timeout_seconds: float = 180.0
    llm_max_output_tokens: int = 2048
    # Hard cap on transcript characters sent to the LLM. Longer transcripts are
    # truncated with a warning rather than rejected, so one long video cannot
    # fail the whole run.
    llm_max_input_chars: int = 300_000
    # Per-takeaway bullet cap, applied by the analyzer after validation.
    takeaway_max_bullets: int = 5

    # Audio pipeline settings. The 5 minute default debounces notifications and
    # gives YouTube's ASR pipeline time to finish producing the video.
    audio_download_delay_seconds: float = Field(default=300.0, ge=0)
    audio_dir: Path = Path("audio")
    audio_format: str = "webm"
    audio_bitrate: str = "32k"
    ffmpeg_timeout_seconds: float = Field(default=900.0, gt=0)

    # OpenAI speech-to-text. ``gpt-transcribe`` is the current model; the
    # ``gpt-4o-transcribe`` family is deprecated and shuts down 2027-02-26.
    transcribe_model: str = "gpt-transcribe"
    transcribe_langs: list[str] = Field(default_factory=lambda: ["en"])
    transcribe_timeout_seconds: float = Field(default=600.0, gt=0)

    # Shorts filter. When false (the default), notify drops every entry whose
    # URL points at a YouTube Short so the digest only covers full-length
    # videos. Set TLDW_INCLUDE_SHORTS=true to keep them.
    include_shorts: bool = False

    @field_validator("audio_format")
    @classmethod
    def _validate_audio_format(cls, value: str) -> str:
        """Reject containers that OpenAI's transcription endpoint will not accept.

        Failing fast at startup is much better than failing the first video with
        a mysterious 400 from the API. The accepted set is documented at
        developers.openai.com/api/docs/guides/speech-to-text.
        """
        if value not in _OPENAI_AUDIO_FORMATS:
            choices = ", ".join(sorted(_OPENAI_AUDIO_FORMATS))
            raise ValueError(
                f"audio_format {value!r} is not supported by OpenAI; "
                f"choose one of: {choices}"
            )
        return value

    @field_validator("discord_webhook_url", mode="before")
    @classmethod
    def _normalize_webhook_url(cls, value: str | None) -> str | None:
        """Treat a blank webhook URL the same as an unset one.

        The shipped compose and k8s manifests set TLDW_DISCORD_WEBHOOK_URL to an
        empty string. Without this, an empty string is a valid ``str``, so the
        ``is None`` guards elsewhere would let the worker start and burn the
        YouTube request budget against a relative URL. mode="before" runs on the
        raw env value, so both env reads and direct construction normalize.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _normalize_openai_api_key(cls, value: str | None) -> str | None:
        """Treat a blank OpenAI API key the same as an unset one.

        ``.env.example`` ships ``TLDW_OPENAI_API_KEY=`` so a copied file has
        an empty string. An empty string is a valid ``str`` but is falsy, and
        the feature switch is "no key". Normalizing to ``None`` keeps that
        contract honest for both env reads and direct construction.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("include_shorts", mode="before")
    @classmethod
    def _normalize_include_shorts(cls, value: object) -> object:
        """Treat a blank or whitespace TLDW_INCLUDE_SHORTS as the default false.

        Pydantic rejects an empty string with a ValidationError, which would
        break operators who leave the line commented out or trailing in
        ``.env``. An empty value matches the documented default, so it is
        normalized to ``False`` before the bool parser runs.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
        return value


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
