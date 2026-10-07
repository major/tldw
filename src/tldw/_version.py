"""Build and runtime identity for tldw.

The Containerfile stamps ``TLDW_GIT_SHA`` and ``TLDW_BUILD_TIME`` at build time
and exposes them to the process. ``started_at`` is captured the first time the
module is imported, so the value is consistent for the lifetime of the process
and can be logged at startup or returned from the ``/version`` endpoint.

Operators read these three values from container logs and from the
``/version`` HTTP endpoint to confirm which commit and build is running in
Kubernetes. When the env vars are missing (for example during local development
without the Containerfile), the values fall back to ``"unknown"`` so logging
and the endpoint keep working.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

__all__ = [
    "BUILD_TIME",
    "GIT_SHA",
    "STARTED_AT",
    "log_banner",
    "snapshot",
]


def _read_env(name: str) -> str:
    """Return the env var or ``"unknown"`` when unset or empty.

    An empty string is treated the same as unset so a build that passes an
    empty ``--build-arg`` does not produce a misleading blank banner line.
    """
    value = os.environ.get(name, "").strip()
    return value or "unknown"


# Captured at import time so every read sees the same value. The container's
# PID 1 process imports this once, so the timestamps match what is logged.
GIT_SHA: str = _read_env("TLDW_GIT_SHA")
BUILD_TIME: str = _read_env("TLDW_BUILD_TIME")
STARTED_AT: str = datetime.now(UTC).isoformat(timespec="seconds")


def snapshot() -> dict[str, str]:
    """Return the current process identity as a plain dict.

    Returning a fresh dict (rather than module globals) keeps the call site
    easy to mock in tests and avoids accidental mutation of the module state.
    """
    return {
        "git_sha": GIT_SHA,
        "build_time": BUILD_TIME,
        "started_at": STARTED_AT,
    }


def log_banner(logger) -> None:
    """Emit a single INFO line that names the running build.

    Kubernetes surfaces stdout from PID 1, so this line lands in
    ``kubectl logs`` on pod start. One line is enough: the values are also
    exposed via ``/version`` for anything that prefers HTTP.
    """
    logger.info(
        "tldw starting git_sha=%s build_time=%s started_at=%s",
        GIT_SHA,
        BUILD_TIME,
        STARTED_AT,
    )
