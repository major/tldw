"""Guards for the async test harness configuration.

These tests protect the pytest settings in pyproject.toml. If someone removes
--allow-unix-socket, every async test in the suite fails during setup with an
opaque SocketBlockedError. These two tests fail first and say why.
"""

import asyncio
import socket

import pytest


async def test_event_loop_starts_under_socket_policy() -> None:
    """An async test can create an event loop and run on it."""
    # Arrange
    marker: list[str] = []

    # Act
    await asyncio.sleep(0)
    marker.append("loop-ran")

    # Assert
    assert marker == ["loop-ran"]


def test_tcp_and_dns_stay_blocked() -> None:
    """--allow-unix-socket did not reopen real network access."""
    # Arrange / Act
    with pytest.raises(Exception) as excinfo:
        socket.getaddrinfo("example.com", 443)

    # Assert
    assert type(excinfo.value).__name__ == "SocketBlockedError"
