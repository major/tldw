"""Command line entry point for the tldw PubSubHubbub subscriber.

The CLI's whole job is small: load settings from the environment, build the
FastAPI application, and hand it to uvicorn. Startup and shutdown behavior
lives in the app's lifespan, which subscribes every resolved channel and runs
the renewal loop, so the CLI only has to get uvicorn running.
"""

from __future__ import annotations

import logging

import uvicorn

from tldw import _version
from tldw.app import create_app
from tldw.config import Settings

__all__ = ["main"]

logger = logging.getLogger(__name__)

# Default port uvicorn listens on when nothing overrides it.
_DEFAULT_PORT = 8000

# Path the Kubernetes liveness and readiness probes hit. Each probe otherwise
# produces an INFO line in uvicorn's access log, which floods the container
# log under steady-state probing. The filter below drops just that one path
# so callback hits, queue polls, and other operator traffic still appear.
_PROBE_PATH = "/version"
# Uvicorn formats access records as 'CLIENT - "METHOD PATH HTTP/VER" STATUS'.
# Matching 'PATH HTTP/' (no leading quote) keeps the filter robust if uvicorn
# ever drops the surrounding quotes, and still distinguishes '/version' from
# a hypothetical '/versions' endpoint.
_PROBE_MARKER = f"{_PROBE_PATH} HTTP/"


class _SuppressAccessPath(logging.Filter):
    """Drop uvicorn access log records whose request line matches a path."""

    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = marker

    def filter(self, record: logging.LogRecord) -> bool:
        # ``getMessage`` lazily formats the record with its args, which is what
        # the handler eventually writes. Skipping it here is the supported way
        # to silence a record without touching uvicorn's formatter.
        return self._marker not in record.getMessage()


def _silence_probe_access_log() -> None:
    """Attach a filter to ``uvicorn.access`` that drops probe traffic.

    The filter is idempotent: re-invocation replaces the previous one instead
    of stacking, which keeps the logger's filter list short across reloads.
    """
    access_logger = logging.getLogger("uvicorn.access")
    for existing in list(access_logger.filters):
        if isinstance(existing, _SuppressAccessPath):
            access_logger.removeFilter(existing)
    access_logger.addFilter(_SuppressAccessPath(_PROBE_MARKER))
    # Uvicorn sets the access log level to INFO at startup. If an operator
    # raised it above INFO to silence noise globally, leave that choice alone.
    # Otherwise ensure the level is INFO so the filter, not the level, gates
    # what gets recorded.
    if access_logger.level == logging.NOTSET:
        access_logger.setLevel(logging.INFO)


def main() -> None:
    """Load settings, build the app, and serve it with uvicorn.

    When TLDW_CALLBACK_URL is unset the app still starts, but no subscriptions
    are issued, so a warning names the variable an operator must set.
    """
    # Emit the build identity before anything else so a slow startup (such as
    # waiting on the hub) does not push the banner past other log lines.
    _version.log_banner(logger)
    settings = Settings()
    if settings.callback_url is None:
        logger.warning(
            "TLDW_CALLBACK_URL is not set; the app will start but no "
            "subscriptions will be issued"
        )
    app = create_app(settings)
    _silence_probe_access_log()
    uvicorn.run(app, host="0.0.0.0", port=_DEFAULT_PORT)
