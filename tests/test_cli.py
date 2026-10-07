"""Tests for the logging configuration in tldw.cli.

The CLI owns the shape of every log line the process writes. It installs one
UTC-timestamped handler on the root logger and hands uvicorn a ``log_config``
whose formatters use the same class, so ``kubectl logs`` sees one timestamp
format no matter which component emitted the line. These tests drive the real
formatter and the real configuration dict instead of mocking them, and they
pin the exact format and date strings so drift breaks loudly.

Tests that call ``_configure_root_logging`` mutate the root logger, which
pytest itself has already populated. The autouse ``_restore_root_logger``
fixture snapshots the root handlers and level before each test and puts them
back afterwards so the module cannot pollute the rest of the suite.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from tldw import cli

# 2025-01-01T00:00:00+00:00 in Unix time. A fixed epoch keeps the rendered
# timestamp deterministic across machines and time zones.
_FIXED_EPOCH = 1735689600


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Snapshot and restore the root logger around every test in this module.

    pytest pre-populates the root logger with capture handlers and a WARNING
    level. ``_configure_root_logging`` clears those handlers, so tests here
    must put the originals back or later tests in the suite lose their capture
    plumbing.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def _make_record(
    msg: str,
    *,
    name: str = "tldw.cli",
    level: int = logging.INFO,
    args: tuple[object, ...] | None = None,
) -> logging.LogRecord:
    """Build a LogRecord with a fixed creation time and no exception info."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    # Override the wall clock captured by LogRecord.__init__ so the rendered
    # asctime is a known UTC moment.
    record.created = _FIXED_EPOCH
    return record


def test_utc_formatter_renders_iso_timestamp_in_utc() -> None:
    """_UtcFormatter renders asctime as ISO 8601 UTC from the fixed epoch."""
    # Arrange
    formatter = cli._UtcFormatter(fmt=cli._LOG_FORMAT, datefmt=cli._LOG_DATEFMT)
    record = _make_record("hello %s", args=("world",))

    # Act
    line = formatter.format(record)

    # Assert
    assert line.startswith("2025-01-01T00:00:00+00:00 ")
    assert " INFO " in line
    assert record.name in line
    assert "hello world" in line


def test_configure_root_logging_adds_utc_handler() -> None:
    """The function appends a UTC stream handler and keeps pre-existing ones."""
    # Arrange
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    starting = len(root.handlers)

    # Act
    cli._configure_root_logging()

    # Assert
    assert len(root.handlers) == starting + 1
    utc_handlers = [
        h for h in root.handlers if isinstance(h.formatter, cli._UtcFormatter)
    ]
    assert len(utc_handlers) == 1
    # Pre-existing handlers are not removed, so test capture plumbing (e.g.
    # pytest's LogCaptureHandler) keeps working.
    assert sentinel in root.handlers
    handler = utc_handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    # Narrow away the ``Formatter | None`` union so the next assertions are
    # type-safe. ``setFormatter`` was called with a real formatter above.
    assert isinstance(handler.formatter, cli._UtcFormatter)
    # Compare the raw strings, not the class, so a changed format fails here.
    assert handler.formatter._fmt == cli._LOG_FORMAT
    assert handler.formatter.datefmt == cli._LOG_DATEFMT


def test_configure_root_logging_is_idempotent() -> None:
    """Calling the function twice leaves exactly one UTC handler on the root."""
    # Arrange
    root = logging.getLogger()
    # pytest leaves the root at WARNING, but the function only promotes the
    # NOTSET sentinel, so reset it to model a fresh interpreter.
    root.setLevel(logging.NOTSET)

    # Act
    cli._configure_root_logging()
    cli._configure_root_logging()

    # Assert
    utc_handlers = [
        h for h in root.handlers if isinstance(h.formatter, cli._UtcFormatter)
    ]
    assert len(utc_handlers) == 1
    assert root.level == logging.INFO


def test_log_config_uses_utc_formatter_for_default_and_access() -> None:
    """The uvicorn log_config points both formatters at _UtcFormatter."""
    # Arrange
    formatters = cli._LOG_CONFIG["formatters"]
    loggers = cli._LOG_CONFIG["loggers"]

    # Act
    # Assert
    for name in ("default", "access"):
        formatter = formatters[name]
        assert formatter["()"] is cli._UtcFormatter
        assert "asctime" in formatter["fmt"]
        assert "+00:00" in formatter["datefmt"]
    # propagate False stops uvicorn records from also hitting the root handler,
    # which is what lets the access formatter own the access line shape.
    assert loggers["uvicorn"]["propagate"] is False
    assert loggers["uvicorn.access"]["propagate"] is False


def test_suppress_access_path_filter_drops_probe_requests() -> None:
    """_SuppressAccessPath drops the probe path but keeps real callbacks."""
    # Arrange
    access_filter = cli._SuppressAccessPath(cli._PROBE_MARKER)
    probe = _make_record(
        '127.0.0.1:52342 - "GET /version HTTP/1.1" 200 OK',
        name="uvicorn.access",
    )
    callback = _make_record(
        '127.0.0.1:52342 - "POST /pubsub/callback HTTP/1.1" 200 OK',
        name="uvicorn.access",
    )

    # Act
    probe_allowed = access_filter.filter(probe)
    callback_allowed = access_filter.filter(callback)

    # Assert
    assert probe_allowed is False
    assert callback_allowed is True
