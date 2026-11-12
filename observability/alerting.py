"""LLM circuit-breaker alerting through the standard logging pipeline.

Off by default (``AppConfig.ALERTING_ENABLED = False``). When enabled, every
circuit-breaker transition on an LLM client is logged; an opening breaker is
logged at CRITICAL so any log-based alert rule (or a handler added to the
``observability.alerting`` logger) can page an operator. This replaces the former
``notification_manager`` console channel with zero extra dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faq_core.llm.resilience import CircuitState
from observability.log import get_logger

if TYPE_CHECKING:
    from faq_core.config import AppConfig
    from faq_core.llm.resilience import StateChangeCallback

_log = get_logger(__name__)


def log_circuit_state_change(name: str, old_state: CircuitState, new_state: CircuitState) -> None:
    """:class:`~faq_core.llm.resilience.CircuitBreaker` callback: log the transition."""
    level = logging.CRITICAL if new_state == CircuitState.OPEN else logging.WARNING
    _log.log(
        level,
        "LLM circuit breaker changed state",
        extra={"circuit": name, "old_state": old_state.value, "new_state": new_state.value},
    )


def build_circuit_breaker_alert(config: AppConfig) -> StateChangeCallback | None:
    """Return the alert callback when alerting is enabled, else ``None``."""
    return log_circuit_state_change if config.ALERTING_ENABLED else None
