"""Tests for the tldw backoff helper.

These tests pin the pure math of ``backoff_delay``: the first retry waits
``base`` seconds, each later attempt doubles the delay, and the delay stops
growing at ``cap``. They also pin the jitter behavior, which must stay inside
``base * (1 +/- jitter)`` and never exceed ``cap * (1 + jitter)``. The jitter
tests seed the RNG so a failure is reproducible. Validation tests cover the
rejected inputs. No fixtures are needed because the function is standalone.
"""

from __future__ import annotations

import random

import pytest

from tldw.backoff import backoff_delay


def test_backoff_delay_first_attempt_is_base_with_zero_jitter() -> None:
    """The first retry waits exactly base seconds when jitter is off."""
    # Arrange
    # attempt=1 with jitter=0.0 is the fully deterministic case.

    # Act
    delay = backoff_delay(1, jitter=0.0)

    # Assert
    assert delay == 600.0


@pytest.mark.parametrize(
    "attempt, expected",
    [
        (1, 600.0),
        (2, 1200.0),
        (3, 2400.0),
        (4, 3600.0),  # 4800 unjittered, clamped to the default cap of 3600.
    ],
)
def test_backoff_delay_doubles_per_attempt(attempt: int, expected: float) -> None:
    """Each attempt doubles the delay until the cap is reached."""
    # Arrange
    # The default base and cap are used; jitter is off for a deterministic value.

    # Act
    delay = backoff_delay(attempt, jitter=0.0)

    # Assert
    assert delay == expected


def test_backoff_delay_caps_at_cap() -> None:
    """A large attempt count never exceeds the configured cap."""
    # Arrange
    # 100 * 2**9 is far above the 500 second cap.

    # Act
    delay = backoff_delay(10, base=100, cap=500, jitter=0.0)

    # Assert
    assert delay == 500.0


def test_backoff_delay_with_zero_jitter_is_deterministic() -> None:
    """Two calls with jitter off return the same value."""
    # Arrange
    # Any attempt works; pick one above the first doubling.

    # Act
    first = backoff_delay(3, jitter=0.0)
    second = backoff_delay(3, jitter=0.0)

    # Assert
    assert first == second


def test_backoff_delay_jitter_stays_within_bounds() -> None:
    """With jitter=0.2 every result stays inside the +/- 20% band."""
    # Arrange
    random.seed(0)
    results = []

    # Act
    for _ in range(200):
        results.append(backoff_delay(2, jitter=0.2))

    # Assert
    # attempt=2 is 1200 seconds unjittered, so the band is [960, 1440].
    assert all(960.0 <= delay <= 1440.0 for delay in results)


def test_backoff_delay_jitter_cannot_exceed_cap_times_factor() -> None:
    """Jitter on a capped delay never exceeds cap * (1 + jitter)."""
    # Arrange
    random.seed(0)
    results = []

    # Act
    for _ in range(200):
        results.append(backoff_delay(10, base=100, cap=500, jitter=0.5))

    # Assert
    assert max(results) <= 500.0 * 1.5


@pytest.mark.parametrize("attempt", [0, -1])
def test_backoff_delay_rejects_invalid_attempt(attempt: int) -> None:
    """An attempt below 1 is not a retry and is rejected."""
    # Arrange
    # attempt=0 and attempt=-1 are both invalid.

    # Act / Assert
    with pytest.raises(ValueError):
        backoff_delay(attempt, jitter=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": -1.0},
        {"cap": -1.0},
        {"jitter": -0.1},
        {"jitter": 1.1},
    ],
)
def test_backoff_delay_rejects_negative_base_cap_jitter(
    kwargs: dict[str, float],
) -> None:
    """A negative base, negative cap, or out-of-range jitter is rejected."""
    # Arrange
    # Each case carries one invalid override; attempt stays valid at 1.

    # Act / Assert
    with pytest.raises(ValueError):
        backoff_delay(1, **kwargs)
