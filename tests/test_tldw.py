"""Tests for the tldw CLI entry point and module re-exports."""

from __future__ import annotations

import logging
import runpy
from typing import Any

import fastapi
import pytest

import tldw.cli as tldw_cli


@pytest.fixture
def captured_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch tldw.cli.uvicorn.run so it does not bind a port.

    Returns a dict tests can read to inspect the call. ``run`` is the one
    legitimately solitary patch in this suite; everything else is sociable.
    """
    captured: dict[str, Any] = {}

    def fake_run(app: Any, **run_kwargs: Any) -> None:
        captured["app"] = app
        captured["kwargs"] = run_kwargs

    monkeypatch.setattr("tldw.cli.uvicorn.run", fake_run)
    return captured


def test_main_calls_uvicorn_run_with_default_port(
    captured_uvicorn: dict[str, Any],
) -> None:
    """main() runs uvicorn on 0.0.0.0:8000 with the built FastAPI app."""
    # Arrange
    expected_kwargs = {"host": "0.0.0.0", "port": 8000}

    # Act
    tldw_cli.main()

    # Assert
    assert captured_uvicorn["kwargs"] == expected_kwargs
    assert isinstance(captured_uvicorn["app"], fastapi.FastAPI)


def test_main_installs_probe_filter_on_uvicorn_access(
    captured_uvicorn: dict[str, Any],
) -> None:
    """main() attaches a path-suppressing filter to uvicorn.access.

    Without this, the readiness and liveness probes flood the container log
    with one INFO line per probe; the filter drops those records while
    leaving every other endpoint's access line visible.
    """
    # Arrange: clear any filter left over from a prior test in this session.
    access_logger = logging.getLogger("uvicorn.access")
    for existing in list(access_logger.filters):
        access_logger.removeFilter(existing)

    # Act
    tldw_cli.main()

    # Assert: the filter drops ``GET /version`` records but keeps others.
    matching = [
        filt
        for filt in access_logger.filters
        if isinstance(filt, tldw_cli._SuppressAccessPath)
    ]
    assert len(matching) == 1
    probe_filter = matching[0]
    probe_record = _make_access_record('10.0.0.1:1234 - "GET /version HTTP/1.1" 200')
    other_record = _make_access_record('10.0.0.1:1234 - "GET /queue HTTP/1.1" 200')
    assert probe_filter.filter(probe_record) is False
    assert probe_filter.filter(other_record) is True


def _make_access_record(message: str) -> logging.LogRecord:
    """Build a LogRecord that mimics what uvicorn hands to its access logger."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_main_loads_callback_url_from_environment(
    captured_uvicorn: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() passes the TLDW_CALLBACK_URL value into the app settings."""
    # Arrange
    callback_url = "https://cb.example/pubsub/callback"
    monkeypatch.setenv("TLDW_CALLBACK_URL", callback_url)

    # Act
    tldw_cli.main()

    # Assert
    assert captured_uvicorn["app"].state.settings.callback_url == callback_url


def test_main_warns_when_callback_url_missing(
    captured_uvicorn: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() warns operators about the missing TLDW_CALLBACK_URL variable."""
    # Arrange
    monkeypatch.delenv("TLDW_CALLBACK_URL", raising=False)

    with caplog.at_level(logging.WARNING):
        # Act
        tldw_cli.main()

    # Assert
    messages = [record.message for record in caplog.records]
    assert any("TLDW_CALLBACK_URL" in message for message in messages)


def test_dunder_main_invokes_cli_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """python -m tldw runs the CLI, so uvicorn.run is reached once."""
    # Arrange
    calls: list[tuple[Any, ...]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        calls.append(args)

    monkeypatch.setattr("tldw.cli.uvicorn.run", fake_run)

    # Act
    runpy.run_module("tldw", run_name="__main__")

    # Assert
    assert len(calls) == 1


def test_module_init_exports_main() -> None:
    """The package re-exports the CLI main callable."""
    # Arrange
    import tldw
    from tldw import main

    # Act
    # (the import itself is the action under test)

    # Assert
    assert main is tldw_cli.main


def test_module_init_dunder_main_invokes_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tldw.__main__ drives the CLI main when run as a module."""
    # Arrange
    calls: list[tuple[Any, ...]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        calls.append(args)

    monkeypatch.setattr("tldw.cli.uvicorn.run", fake_run)

    # Act
    runpy.run_module("tldw.__main__", run_name="__main__")

    # Assert
    assert len(calls) == 1
