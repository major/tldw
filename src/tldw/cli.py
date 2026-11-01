"""Command line entry point for the tldw PubSubHubbub subscriber.

The CLI's whole job is small: load settings from the environment, build the
FastAPI application, and hand it to uvicorn. Startup and shutdown behavior
lives in the app's lifespan, which subscribes every resolved channel and runs
the renewal loop, so the CLI only has to get uvicorn running.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
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

# Exclude built-in fields, including ones another handler may have populated,
# so only log-call extras are promoted to top-level JSON fields.
_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class _JSONFormatter(logging.Formatter):
    """Render stocknews-compatible JSON lines that ``hl -F`` can read."""

    def format(self, record: logging.LogRecord) -> str:
        """Keep extras structured and escape multiline messages and errors."""
        fields: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        fields.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_FIELDS
                and key not in fields
                and not key.startswith("_")
            }
        )
        if record.exc_info:
            fields["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            fields["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(fields, ensure_ascii=False, default=str, separators=(",", ":"))


def _configure_root_logging() -> None:
    """Install an INFO-level JSON stdout handler on the root logger.

    Every ``logging.getLogger(__name__)`` in the project propagates to the
    root, so this is the single point where the on-disk log line shape is
    defined. The function is idempotent: if a handler whose formatter is
    already ``_JSONFormatter`` is attached, the call is a no-op. Pre-existing
    handlers (for example pytest's ``LogCaptureHandler`` during the test
    suite) are left in place so log capture keeps working.
    """
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing.formatter, _JSONFormatter):
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    # A fresh root logger defaults to WARNING, which would hide application
    # startup and worker INFO messages even while Uvicorn emits its own.
    root.setLevel(logging.INFO)


# Use the same JSON format and stdout stream for Uvicorn and application logs.
# Explicit handlers prevent Uvicorn records from propagating to the root and
# appearing twice.
_LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": _JSONFormatter,
        },
        "access": {
            "()": _JSONFormatter,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
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
    # on a stream that already uses JSON. The same config is passed
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
