"""Command line entry point for the tldw PubSubHubbub subscriber.

The CLI's whole job is small: load settings from the environment, build the
FastAPI application, and hand it to uvicorn. Startup and shutdown behavior
lives in the app's lifespan, which subscribes every resolved channel and runs
the renewal loop, so the CLI only has to get uvicorn running.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

# Log line format shared by tldw and uvicorn output. ``asctime`` is rendered
# as ISO 8601 in UTC by ``_UtcFormatter`` below. Keeping the format identical
# for both streams means ``kubectl logs`` and any downstream log shipper see
# a single line shape regardless of which component emitted the line.
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S+00:00"


class _UtcFormatter(logging.Formatter):
    """Render ``asctime`` in UTC so log lines agree across machines and zones.

    Python's default ``Formatter.formatTime`` uses ``time.localtime``, which
    produces host-local timestamps. Container logs from a pod running in a
    different zone than the operator would otherwise disagree on the wall
    clock, so we pin the converter to ``time.gmtime`` and use a literal
    ``+00:00`` in the date format. Operators who want a different zone can
    replace the converter without touching the format string.
    """

    converter = time.gmtime


def _configure_root_logging() -> None:
    """Install a UTC-timestamped stream handler on the root logger.

    Every ``logging.getLogger(__name__)`` in the project propagates to the
    root, so this is the single point where the on-disk log line shape is
    defined. The function is idempotent: if a handler whose formatter is
    already ``_UtcFormatter`` is attached, the call is a no-op. Pre-existing
    handlers (for example pytest's ``LogCaptureHandler`` during the test
    suite) are left in place so log capture keeps working.
    """
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing.formatter, _UtcFormatter):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(_UtcFormatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    root.addHandler(handler)
    # ``NOTSET`` is the sentinel for "fall through to whatever the parent
    # chain says". If a real level has already been set (for example by a
    # deployment that wired in a quieter default), leave it alone.
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


# Uvicorn's default ``log_config`` renders the log line without a timestamp
# and uses formatters that strip ``asctime`` even when it is in the format
# string. Passing our own config gives every uvicorn logger the same UTC
# line shape the rest of the app uses, and pins the access log to stdout
# so ``kubectl logs`` can split error and access streams by convention.
_LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": _UtcFormatter,
            "fmt": _LOG_FORMAT,
            "datefmt": _LOG_DATEFMT,
        },
        "access": {
            "()": _UtcFormatter,
            "fmt": _LOG_FORMAT,
            "datefmt": _LOG_DATEFMT,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


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
    # Configure logging first so the banner and the callback warning land
    # on a stream that already has a timestamp. The same config is passed
    # to uvicorn below so its loggers render the same way.
    _configure_root_logging()
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
    uvicorn.run(app, host="0.0.0.0", port=_DEFAULT_PORT, log_config=_LOG_CONFIG)
