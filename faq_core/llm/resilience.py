"""A small circuit breaker for LLM provider calls.

Replaces the breaker that came with the dropped ``api_client_base`` wheel.
After ``failure_threshold`` consecutive failures the breaker opens and calls
fail fast (no network) for ``recovery_timeout`` seconds; then one trial call is
let through (half-open) — success closes the breaker, failure reopens it.
``on_state_change`` (see :mod:`observability.alerting`) is told about every
transition.

Single event loop, no awaits inside: the state changes are atomic with respect
to other coroutines, so no lock is needed.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


type StateChangeCallback = Callable[[str, CircuitState, CircuitState], None]
"""``(breaker_name, old_state, new_state) -> None``."""


class CircuitBreaker:
    """Consecutive-failure circuit breaker.

    Args:
        name: Shown in alerts, e.g. ``"llm:anthropic"``.
        failure_threshold: Consecutive failures that open the breaker.
        recovery_timeout: Seconds the breaker stays open before a trial call.
        on_state_change: Called on every transition (public, may be replaced).
        clock: Monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        on_state_change: StateChangeCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.on_state_change = on_state_change
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    def allow_request(self) -> bool:
        """Whether a call may go out now (``False`` while the breaker is open)."""
        return self.state is not CircuitState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        if self._state is not CircuitState.CLOSED:
            self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        self._failures += 1
        if self._state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            if self._state is not CircuitState.OPEN:
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old_state, self._state = self._state, new_state
        if self.on_state_change is not None:
            self.on_state_change(self.name, old_state, new_state)
