from __future__ import annotations

import logging

import pytest

from faq_core.config import AppConfig
from faq_core.llm.factory import create_llm_gateway
from faq_core.llm.gateway import ChatCompletionGateway
from faq_core.llm.resilience import CircuitBreaker, CircuitState
from observability.alerting import build_circuit_breaker_alert, log_circuit_state_change


def test_alerting_is_disabled_by_default(app_config: AppConfig) -> None:
    assert build_circuit_breaker_alert(app_config) is None


def test_open_breaker_logs_critical(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="observability.alerting"):
        log_circuit_state_change("__global__", CircuitState.CLOSED, CircuitState.OPEN)

    [record] = caplog.records
    assert record.levelno == logging.CRITICAL
    assert record.new_state == "open"  # type: ignore[attr-defined]


def test_enabled_alerting_is_wired_into_the_llm_circuit_breaker(app_config: AppConfig) -> None:
    app_config.ALERTING_ENABLED = True

    gateway = create_llm_gateway(app_config)

    assert isinstance(gateway, ChatCompletionGateway)
    assert gateway.circuit_breaker.on_state_change is log_circuit_state_change


def test_disabled_alerting_leaves_the_breaker_without_a_callback(app_config: AppConfig) -> None:
    gateway = create_llm_gateway(app_config)

    assert isinstance(gateway, ChatCompletionGateway)
    assert gateway.circuit_breaker.on_state_change is None


def test_breaker_opening_logs_critical_through_the_alert_callback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    breaker = CircuitBreaker(
        "llm:test", failure_threshold=2, on_state_change=log_circuit_state_change
    )

    with caplog.at_level(logging.WARNING, logger="observability.alerting"):
        breaker.record_failure()
        breaker.record_failure()

    [record] = caplog.records
    assert record.levelno == logging.CRITICAL
    assert record.circuit == "llm:test"  # type: ignore[attr-defined]
