"""Load and resolve topic configuration without side effects beyond file reads."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tldw.config import _CHANNEL_ID_RE


class TopicConfigError(ValueError):
    """A safe, operator-facing topic configuration error."""


class _Loader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Construct a mapping without silently overwriting keys."""
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError:
                raise TopicConfigError(
                    f"line {key_node.start_mark.line + 1}: invalid mapping key"
                ) from None
            if duplicate:
                raise TopicConfigError(
                    f"line {key_node.start_mark.line + 1}: duplicate mapping key"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class _Strict(BaseModel):
    """Base for strict configuration records."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _Source(_Strict):
    """A configured source and optional label."""

    channel_id: str = Field(pattern=_CHANNEL_ID_RE.pattern)
    name: str | None = None


class _Prompt(_Strict):
    """Exactly one prompt source."""

    builtin: Literal["trading"] | None = None
    file: str | None = None

    @model_validator(mode="after")
    def one_source(self) -> _Prompt:
        """Require exactly one non-null prompt source."""
        if (self.builtin is None) == (self.file is None):
            raise ValueError("choose exactly one prompt source")
        return self


class _Discord(_Strict):
    """Name the environment variable containing the destination."""

    webhook_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Topic(_Strict):
    """The unresolved topic definition."""

    enabled: bool = True
    channels: list[_Source]
    prompt: _Prompt
    discord: _Discord

    @model_validator(mode="after")
    def active_channels(self) -> _Topic:
        """Require sources for an enabled topic."""
        if self.enabled and not self.channels:
            raise ValueError("enabled topic requires channels")
        return self


class _Config(_Strict):
    """Versioned topic document."""

    version: Literal[1]
    topics: dict[str, _Topic]

    @model_validator(mode="after")
    def valid_topics(self) -> _Config:
        """Require a default topic and stable slug keys."""
        if "default" not in self.topics:
            raise ValueError("default topic is required")
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]*", key) for key in self.topics):
            raise ValueError("invalid topic slug")
        return self


@dataclass(frozen=True, slots=True)
class Source:
    """A source channel and its optional display label."""

    channel_id: str
    name: str | None


@dataclass(frozen=True, slots=True)
class Topic:
    """A resolved topic with a redacted destination."""

    topic_id: str
    enabled: bool
    channels: tuple[Source, ...]
    system_prompt: str
    webhook_env: str
    webhook_url: str | None = field(repr=False)
    _builtin: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class TopicRegistry:
    """Immutable topic and active source indexes."""

    topics: Mapping[str, Topic]
    by_channel: Mapping[str, Topic]

    @property
    def channel_ids(self) -> tuple[str, ...]:
        """Return active source IDs in document order."""
        return tuple(self.by_channel)


def load_topics(path: Path, *, environ: Mapping[str, str] | None = None) -> TopicRegistry:
    """Read and validate a topic document, prompts, and active destinations."""
    env = os.environ if environ is None else environ
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
        config = _Config.model_validate(raw)
        # Literal equality considers True equal to 1, so check the type explicitly.
        if type(raw.get("version")) is not int:
            raise TopicConfigError("version: expected integer 1")
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "document"
        raise TopicConfigError(f"{path}: invalid YAML at {location}") from None
    except ValidationError as exc:
        fields = {
            "version",
            "topics",
            "enabled",
            "channels",
            "channel_id",
            "name",
            "prompt",
            "builtin",
            "file",
            "discord",
            "webhook_env",
        }
        safe_messages = {
            "default topic is required",
            "invalid topic slug",
            "choose exactly one prompt source",
            "enabled topic requires channels",
        }
        details = []
        for error in exc.errors():
            location = (
                ".".join(
                    str(part) if isinstance(part, int) or part in fields else "<key>"
                    for part in error["loc"]
                )
                or "document"
            )
            reason = error["type"]
            if reason == "value_error":
                message = str(error.get("ctx", {}).get("error", ""))
                if message in safe_messages:
                    reason = message
            details.append(f"{location}: {reason}")
        errors = "; ".join(details)
        raise TopicConfigError(f"{path}: {errors}") from None
    except OSError, UnicodeError:
        raise TopicConfigError(f"{path}: cannot read topics file") from None
    except TopicConfigError as exc:
        raise TopicConfigError(f"{path}: {exc}") from None

    topics: dict[str, Topic] = {}
    by_channel: dict[str, Topic] = {}
    for slug, item in config.topics.items():
        if item.prompt.builtin:
            from tldw.llm import SYSTEM_PROMPT

            prompt = SYSTEM_PROMPT
        else:
            filename = item.prompt.file or ""
            if not filename.strip() or Path(filename).is_absolute():
                raise TopicConfigError(
                    f"{path}: topics.{slug}.prompt.file: expected relative path"
                )
            try:
                with (path.parent / filename).open(encoding="utf-8", newline="") as prompt_file:
                    prompt = prompt_file.read()
            except OSError, UnicodeError:
                raise TopicConfigError(
                    f"{path}: topics.{slug}.prompt.file: cannot read prompt"
                ) from None
            if not prompt.strip():
                raise TopicConfigError(f"{path}: topics.{slug}.prompt.file: prompt is blank")
        url = env.get(item.discord.webhook_env)
        url = url.strip() if url else None
        if item.enabled and not url:
            raise TopicConfigError(
                f"{path}: topics.{slug}.discord.webhook_env: destination is missing or blank"
            )
        topic = Topic(
            topic_id=slug,
            enabled=item.enabled,
            channels=tuple(Source(c.channel_id, c.name) for c in item.channels),
            system_prompt=prompt,
            webhook_env=item.discord.webhook_env,
            webhook_url=url or None,
            _builtin=item.prompt.builtin,
        )
        topics[slug] = topic
        if item.enabled:
            for source in topic.channels:
                if source.channel_id in by_channel:
                    raise TopicConfigError(
                        f"{path}: topics.{slug}.channels: duplicate active source"
                    )
                by_channel[source.channel_id] = topic
    return TopicRegistry(MappingProxyType(topics), MappingProxyType(by_channel))
