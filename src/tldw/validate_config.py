"""Validate legacy or topic configuration without starting the service."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from tldw.config import Settings, resolve_channel_ids
from tldw.topics import TopicConfigError, TopicRegistry, load_topics


def _check_queue(path: Path, registry: TopicRegistry | None) -> int:
    """Inspect pending videos and unsent alerts using a read-only connection."""
    # Immutable mode cannot read a WAL, so refuse it instead of missing backlog.
    if Path(f"{path}-wal").exists():
        raise TopicConfigError("queue: WAL exists; checkpoint the database before inspection")
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(videos)")}
            if not {"video_id", "terminal_state"} <= columns:
                raise TopicConfigError("queue: missing required videos schema")
            topic = 'COALESCE(v.topic_id, \'default\')' if "topic_id" in columns else "'default'"
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            pending = f"SELECT {topic} FROM videos v WHERE v.terminal_state IS NULL"
            if "failure_alerts" in tables:
                pending += f" UNION ALL SELECT {topic} FROM failure_alerts a JOIN videos v ON v.video_id = a.video_id WHERE a.sent_at IS NULL"
            references = [row[0] for row in db.execute(pending)]
    except sqlite3.Error:
        raise TopicConfigError("queue: cannot read requested database or schema") from None
    if registry is not None:
        for topic_id in references:
            if topic_id not in registry.topics:
                raise TopicConfigError("queue: backlog references a missing topic")
            if not registry.topics[topic_id].webhook_url:
                raise TopicConfigError(f"queue: backlog for topic {topic_id} is blocked by a missing destination")
    return len(references)


def main(argv: list[str] | None = None) -> None:
    """Run an offline preflight; comparison mode permits legacy CSV temporarily."""
    parser = argparse.ArgumentParser(description="Validate configuration offline. --compare-legacy is a migration preflight and permits the legacy CSV alongside YAML.")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--queue-file", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        registry = None
        if settings.topics_file is None:
            sources = resolve_channel_ids(settings)
            topic_count = 1
        else:
            if not args.compare_legacy and settings.channel_ids_env_override and any(part.strip() for part in settings.channel_ids_env_override.split(",")):
                raise TopicConfigError("TLDW_CHANNEL_IDS conflicts with TLDW_TOPICS_FILE")
            registry = load_topics(settings.topics_file)
            sources = list(registry.channel_ids)
            topic_count = len(registry.topics)
            if args.compare_legacy:
                from tldw.llm import SYSTEM_PROMPT

                default = registry.topics["default"]
                if (set(resolve_channel_ids(settings)) != set(sources) or
                    default.system_prompt != SYSTEM_PROMPT or
                    default._builtin != "trading" or
                    default.webhook_env != "TLDW_DISCORD_WEBHOOK_URL" or
                    any(t.enabled for key, t in registry.topics.items() if key != "default")):
                    raise TopicConfigError("legacy comparison does not match default trading configuration")
        references = _check_queue(args.queue_file, registry) if args.queue_file else 0
        print(f"Valid configuration: {topic_count} topic(s), {len(sources)} source(s), {references} backlog reference(s)")
    except (TopicConfigError, ValueError, OSError, ValidationError):
        # Legacy and settings exceptions can include raw values; do not print them.
        import sys

        exc = sys.exception()
        message = str(exc) if isinstance(exc, TopicConfigError) else "invalid configuration or unreadable file"
        parser.exit(1, f"Configuration error: {message}\n")
