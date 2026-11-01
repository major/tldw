"""Tests for the logging configuration in tldw.cli.

The CLI installs a JSON handler on the root logger and hands uvicorn a
``log_config`` with the same formatter, so ``hl -F`` can read both streams.
These tests drive the real formatter and configuration instead of mocking them.

Tests that call ``_configure_root_logging`` mutate the root logger, which
pytest itself has already populated. The autouse ``_restore_root_logger``
fixture snapshots the root handlers and level before each test and puts them
back afterwards so the module cannot pollute the rest of the suite.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from helpers import make_log_record
from tldw import cli


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Snapshot and restore the root logger around every test in this module.

    ``_configure_root_logging`` preserves pytest's capture handlers but adds
    its own handler and sets the level. Restore both after each test.
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


@pytest.mark.parametrize(
    ("name", "message", "args", "expected"),
    [
        ("tldw.worker", "hello %s", ("world",), "hello world"),
        ("uvicorn.error", "ready ✅\nnext line", None, "ready ✅\nnext line"),
        (
            "uvicorn.access",
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:1234", "POST", "/pubsub/callback", "1.1", 200),
            '127.0.0.1:1234 - "POST /pubsub/callback HTTP/1.1" 200',
        ),
    ],
)
def test_json_formatter_renders_one_line(
    name: str, message: str, args: tuple[object, ...] | None, expected: str
) -> None:
    """Application and Uvicorn records share the stocknews JSON fields."""
    # Arrange
    formatter = cli._JSONFormatter()
    record = make_log_record(message, name=name, args=args)

    # Act
    line = formatter.format(record)

    # Assert
    assert len(line.splitlines()) == 1
    assert json.loads(line) == {
        "time": "2025-01-01T00:00:00.000+00:00",
        "level": "INFO",
        "msg": expected,
        "logger": name,
    }


@pytest.mark.parametrize(
    ("message", "args", "color_message"),
    [
        (
            "Started server process [%d]",
            (1,),
            "Started server process [\x1b[36m%d\x1b[0m]",
        ),
        (
            "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
            ("http", "0.0.0.0", 8000),
            "Uvicorn running on \x1b[1m%s://%s:%d\x1b[0m (Press CTRL+C to quit)",
        ),
    ],
)
def test_json_formatter_omits_uvicorn_color_template(
    message: str, args: tuple[object, ...], color_message: str
) -> None:
    """Uvicorn's console template stays out of JSON without changing the record."""
    # Arrange
    record = make_log_record(message, name="uvicorn.error", args=args)
    record.__dict__.update(color_message=color_message, server="main")
    original = record.__dict__.copy()

    # Act
    fields = json.loads(cli._JSONFormatter().format(record))

    # Assert
    assert fields["msg"] == message % args
    assert fields["server"] == "main"
    assert "color_message" not in fields
    assert all("\x1b" not in value for value in fields.values() if isinstance(value, str))
    assert record.__dict__ == original


def test_json_formatter_includes_extras_without_overwriting_core_fields() -> None:
    """Extras stay structured and fields from earlier formatters stay hidden."""
    # Arrange
    record = make_log_record("queued")
    record.__dict__.update(
        video_id="abc123",
        context={"attempt": 2},
        path=Path("transcripts/video.txt"),
        _private="hidden",
        time="wrong",
        level="wrong",
        logger="wrong",
    )
    logging.Formatter("%(asctime)s %(message)s").format(record)

    # Act
    fields = json.loads(cli._JSONFormatter().format(record))

    # Assert
    assert fields == {
        "time": "2025-01-01T00:00:00.000+00:00",
        "level": "INFO",
        "msg": "queued",
        "logger": "tldw.cli",
        "video_id": "abc123",
        "context": {"attempt": 2},
        "path": "transcripts/video.txt",
    }


def test_json_formatter_keeps_exception_and_stack_on_one_output_line() -> None:
    """Tracebacks and stack information are JSON strings, not extra log lines."""
    # Arrange
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(cli._JSONFormatter())
    record = make_log_record("failed\nretry later", level=logging.ERROR)
    try:
        raise ValueError("bad input\ntry again")
    except ValueError:
        record.exc_info = sys.exc_info()
    record.stack_info = "Stack (most recent call last):\n  caller"

    # Act
    handler.handle(record)

    # Assert
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    fields = json.loads(lines[0])
    assert fields["msg"] == "failed\nretry later"
    assert fields["level"] == "ERROR"
    assert "Traceback (most recent call last):" in fields["exception"]
    assert "ValueError: bad input\ntry again" in fields["exception"]
    assert fields["stack_info"] == record.stack_info


def test_configure_root_logging_adds_json_stdout_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep existing handlers and emit application INFO logs as JSON to stdout."""
    # Arrange
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    root.setLevel(logging.WARNING)
    starting = len(root.handlers)

    # Act
    cli._configure_root_logging()
    logging.getLogger("tldw.worker").info("queued %s", "abc123")

    # Assert
    assert len(root.handlers) == starting + 1
    json_handlers = [
        h for h in root.handlers if isinstance(h.formatter, cli._JSONFormatter)
    ]
    assert len(json_handlers) == 1
    # Pre-existing handlers are not removed, so test capture plumbing (e.g.
    # pytest's LogCaptureHandler) keeps working.
    assert sentinel in root.handlers
    handler = json_handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout
    output = capsys.readouterr()
    assert output.err == ""
    fields = json.loads(output.out)
    assert fields["level"] == "INFO"
    assert fields["logger"] == "tldw.worker"
    assert fields["msg"] == "queued abc123"


def test_configure_root_logging_is_idempotent() -> None:
    """Calling the function twice leaves exactly one JSON handler on the root."""
    # Arrange
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)

    # Act
    cli._configure_root_logging()
    cli._configure_root_logging()

    # Assert
    json_handlers = [
        h for h in root.handlers if isinstance(h.formatter, cli._JSONFormatter)
    ]
    assert len(json_handlers) == 1
    assert root.level == logging.INFO


def test_log_config_uses_json_stdout_for_default_and_access() -> None:
    """Uvicorn uses the same JSON formatter and stdout without duplicates."""
    # Arrange
    formatters = cli._LOG_CONFIG["formatters"]
    handlers = cli._LOG_CONFIG["handlers"]
    loggers = cli._LOG_CONFIG["loggers"]

    # Act
    # Assert
    for name in ("default", "access"):
        formatter = formatters[name]
        assert formatter["()"] is cli._JSONFormatter
        assert handlers[name]["formatter"] == name
        assert handlers[name]["class"] == "logging.StreamHandler"
        assert handlers[name]["stream"] == "ext://sys.stdout"
    # propagate False stops uvicorn records from also hitting the root handler,
    # which is what lets the access formatter own the access line shape.
    assert loggers["uvicorn"]["propagate"] is False
    assert loggers["uvicorn.access"]["propagate"] is False


def test_suppress_access_path_filter_drops_probe_requests() -> None:
    """_SuppressAccessPath drops the probe path but keeps real callbacks."""
    # Arrange
    access_filter = cli._SuppressAccessPath(cli._PROBE_MARKER)
    probe = make_log_record(
        '127.0.0.1:52342 - "GET /version HTTP/1.1" 200 OK',
        name="uvicorn.access",
    )
    callback = make_log_record(
        '127.0.0.1:52342 - "POST /pubsub/callback HTTP/1.1" 200 OK',
        name="uvicorn.access",
    )

    # Act
    probe_allowed = access_filter.filter(probe)
    callback_allowed = access_filter.filter(callback)

    # Assert
    assert probe_allowed is False
    assert callback_allowed is True
