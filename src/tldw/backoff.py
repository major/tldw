"""Exponential backoff delay for the transcript worker.

The worker retries a failed unit of work instead of giving up. Sleeping the
same amount after every failure would hammer a struggling upstream, so this
module computes a delay that grows exponentially with the attempt number and
stops growing once it reaches a cap. A small, bounded jitter spreads out
retries from many workers so they do not all wake at the same instant.

The module is pure: it reads no settings, performs no I/O, and keeps no state.
The caller owns the clock and the loop. When ``jitter`` is 0.0 the result is
deterministic, which keeps the arithmetic easy to test and reason about.
"""

from __future__ import annotations

import logging
import random

__all__ = ["backoff_delay"]

logger = logging.getLogger(__name__)

# Ten minutes for the first retry, doubling each attempt up to one hour.
DEFAULT_BASE_SECONDS: float = 600.0
DEFAULT_CAP_SECONDS: float = 3600.0
# Spread retries by +/- 20% so a fleet of workers does not retry in lockstep.
DEFAULT_JITTER: float = 0.2


def backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BASE_SECONDS,
    cap: float = DEFAULT_CAP_SECONDS,
    jitter: float = DEFAULT_JITTER,
) -> float:
    """Return the next backoff delay in seconds.

    ``attempt`` is 1-based: the first retry passes 1, the second passes 2. The
    unjittered delay is ``min(cap, base * 2 ** (attempt - 1))``. When jitter is
    greater than zero a uniformly random multiplier in
    ``[1 - jitter, 1 + jitter]`` is applied to the unjittered delay. The final
    value is clamped so jitter can never push it above ``cap * (1 + jitter)``.

    Pass ``jitter=0.0`` for a deterministic result in tests. Raises ValueError
    when ``attempt`` is less than 1 or when ``base``, ``cap``, or ``jitter`` is
    negative, or when ``jitter`` is greater than 1.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    if base < 0:
        raise ValueError(f"base must be >= 0, got {base}")
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    if jitter < 0 or jitter > 1:
        raise ValueError(f"jitter must be between 0 and 1, got {jitter}")

    delay = min(cap, base * 2 ** (attempt - 1))
    if jitter > 0:
        delay *= random.uniform(1.0 - jitter, 1.0 + jitter)
        # Clamp after jitter so a wide multiplier cannot blow past the cap.
        delay = min(delay, cap * (1.0 + jitter))
    return float(delay)
