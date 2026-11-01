"""Persist and deliver operational failures without affecting the pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx2

from tldw import discord
from tldw.config import Settings
from tldw.queue import QueueRecord, QueueStore

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
    except Exception:
        logger.exception("could not persist failure alert for %s", record.video_id)


async def flush_failure_alert(
    settings: Settings,
    store: QueueStore,
    client: httpx2.AsyncClient,
    now: float | None = None,
) -> bool:
    """Attempt at most one pending alert, returning false when delivery fails."""
    if not settings.discord_alert_user_id or not settings.discord_webhook_url:
        return True
    try:
        alert = store.next_failure_alert()
        if alert is None:
            return True
        if alert.current_terminal_state == "DONE":
            disposition = (
                "Delivered using a plain transcript fallback."
                if alert.current_detail == "llm_fallback"
                else "Video summary delivered successfully."
            )
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
                settings.discord_webhook_url,
                content,
                user_id=settings.discord_alert_user_id,
            ),
            timeout=10,
        )
        store.mark_failure_alert_sent(alert.video_id, alert.stage, now=now)
        return True
    except Exception:
        logger.exception("could not deliver failure alert")
        return False
