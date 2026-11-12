"""In-memory sliding-window rate limiter.

Each key (an authenticated key id, or ``ip:<address>`` for unauthenticated
traffic) may make *limit* requests in any *window*-second span; the next one is
refused with the seconds until a slot frees up (sent as ``Retry-After``).

Per process only: several workers each keep their own counts. At most
*max_keys* keys are tracked; the least recently used one is dropped beyond
that, which bounds memory under traffic from many addresses.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_MAX_KEYS: Final = 10_000


@dataclass(frozen=True, slots=True)
class RateDecision:
    """Outcome of one :meth:`SlidingWindowRateLimiter.check`."""

    allowed: bool
    retry_after: int = 0  # whole seconds, >= 1 when refused


class SlidingWindowRateLimiter:
    """Exact sliding-window log: a deque of request times per key."""

    def __init__(
        self,
        limit: int,
        window: float,
        *,
        max_keys: int = DEFAULT_MAX_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window <= 0 or max_keys < 1:
            raise ValueError("rate limit, window and max_keys must be positive")
        self._limit = limit
        self._window = window
        self._max_keys = max_keys
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str) -> RateDecision:
        """Count one request for *key*, or refuse it when the window is full."""
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
            self._hits[key] = hits
            if len(self._hits) > self._max_keys:
                self._hits.popitem(last=False)
        else:
            self._hits.move_to_end(key)
        horizon = now - self._window
        while hits and hits[0] <= horizon:
            hits.popleft()
        if len(hits) >= self._limit:
            retry_after = max(1, math.ceil(hits[0] + self._window - now))
            return RateDecision(allowed=False, retry_after=retry_after)
        hits.append(now)
        return RateDecision(allowed=True)
