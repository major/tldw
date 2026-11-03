"""Persist and deliver operational failures without affecting the pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

import httpx2

from tldw import discord
from tldw.config import Settings
from tldw.queue import QueueRecord, QueueStore
from tldw.topics import TopicRegistry

logger = logging.getLogger(__name__)


def record_failure(
    settings: Settings,
    store: QueueStore,
    record: QueueRecord,
    stage: str,
    reason: str,
    retrying: bool,
    now: float | None = None,
) -> None:
    """Record an actual failure before changing the video's queue state."""
    if not settings.discord_alert_user_id:
        return
    try:
        store.enqueue_failure_alert(record.video_id, stage, reason, retrying, now=now)
    except Exception as exc:
        logger.warning(
            "could not persist failure alert: video=%s topic=%s stage=%s error=%s",
            record.video_id,
            record.topic_id,
            stage,
            type(exc).__name__,
        )


async def flush_failure_alert(
    settings: Settings,
    store: QueueStore,
    client: httpx2.AsyncClient,
    now: float | None = None,
    *,
    registry: TopicRegistry | None = None,
    cooldowns: dict[str, float] | None = None,
) -> bool:
    """Attempt at most one pending alert, returning false when delivery fails."""
    if not settings.discord_alert_user_id:
        return True
    current = time.monotonic()
    routes = (
        {
            key: topic.webhook_url
            for key, topic in registry.topics.items()
            if topic.webhook_url
        }
        if registry is not None
        else {"default": settings.discord_webhook_url}
        if settings.discord_webhook_url
        else {}
    )
    eligible = [
        key for key in routes if cooldowns is None or cooldowns.get(key, 0) <= current
    ]
    alert = None
    try:
        alert = store.next_failure_alert(topic_ids=eligible)
        if alert is None:
            return True
        if alert.current_terminal_state == "DONE":
            disposition = (
                "Delivered using a plain transcript fallback."
                if alert.current_detail == "llm_fallback"
                else "Video summary delivered successfully."
            )
        elif alert.current_terminal_state == "SKIPPED_LIVE":
            disposition = "Skipped because live events are excluded; no automatic retry."
        elif alert.current_terminal_state and alert.current_terminal_state.startswith(
            "GIVE_UP"
        ):
            disposition = "Stopped; no automatic retry."
        else:
            disposition = "Video remains queued for processing."
        failure_time = datetime.fromtimestamp(alert.created_at, UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        content = (
            f"Video: {alert.title[:300]}\nChannel: {alert.channel_name[:200]}\n"
            f"URL: {alert.url[:700]}\nStage: {alert.stage[:100]}\n"
            f"Failure time: {failure_time}\nReason: {alert.reason[:300]}\n"
            f"Current status: {disposition}"
        )
        await asyncio.wait_for(
            discord.send_alert(
                client,
                routes[alert.topic_id],
                content,
                user_id=settings.discord_alert_user_id,
            ),
            timeout=10,
        )
        store.mark_failure_alert_sent(alert.video_id, alert.stage, now=now)
        return True
    except Exception as exc:
        logger.warning("could not deliver failure alert: %s", type(exc).__name__)
        if registry is not None and cooldowns is not None and alert is not None:
            cooldowns[alert.topic_id] = current + 60.0
        return False
