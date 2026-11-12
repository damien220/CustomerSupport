"""Escalation sinks: signed webhook (fake transport, no network) and the log sink."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from pydantic import SecretStr

from faq_core.config import AppConfig
from faq_core.handoff import (
    EVENT_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    LogSink,
    WebhookSink,
    create_escalation_sink,
    sign_payload,
    verify_signature,
)
from faq_core.models import Escalation

_SECRET = "test-webhook-signing-secret"  # noqa: S105 — fake test value
_URL = "https://hooks.example.com/support"


def _escalation() -> Escalation:
    return Escalation(
        id=uuid.uuid4(),
        reason="requested,low_confidence",
        packet={"schema": "support.escalation/v1", "customer_goal": "refund [EMAIL]"},
    )


async def _no_sleep(_seconds: float) -> None:
    return None


def _sink(handler: httpx.MockTransport, *, max_retries: int = 2) -> WebhookSink:
    return WebhookSink(
        url=_URL,
        secret=SecretStr(_SECRET),
        max_retries=max_retries,
        transport=handler,
        sleep=_no_sleep,
    )


async def test_webhook_posts_a_signed_packet_the_receiver_can_verify() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(202)

    escalation = _escalation()
    delivery = await _sink(httpx.MockTransport(handler)).deliver(escalation)

    [request] = received
    body = request.content
    assert verify_signature(
        _SECRET.encode(), request.headers[TIMESTAMP_HEADER], body, request.headers[SIGNATURE_HEADER]
    )
    assert request.headers[EVENT_HEADER] == "escalation.created"
    assert request.headers["Idempotency-Key"] == str(escalation.id)
    assert json.loads(body) == {"event": "escalation.created", "packet": escalation.packet}
    assert (delivery.sink, delivery.delivered, delivery.attempts) == ("webhook", True, 1)
    assert _SECRET not in body.decode() and _SECRET not in str(request.headers)


async def test_tampered_body_fails_verification() -> None:
    signature = sign_payload(b"k", "1", b'{"a":1}')

    assert not verify_signature(b"k", "1", b'{"a":2}', signature)
    assert not verify_signature(b"other", "1", b'{"a":1}', signature)


async def test_webhook_retries_5xx_and_transport_errors_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
        if calls == 2:
            return httpx.Response(503)
        return httpx.Response(200)

    delivery = await _sink(httpx.MockTransport(handler)).deliver(_escalation())

    assert (delivery.delivered, delivery.attempts, calls) == (True, 3, 3)


async def test_webhook_retries_are_bounded_and_failure_is_reported_not_raised() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    delivery = await _sink(httpx.MockTransport(handler), max_retries=1).deliver(_escalation())

    assert (delivery.delivered, delivery.attempts, delivery.error, calls) == (
        False,
        2,
        "HTTP 500",
        2,
    )


async def test_webhook_does_not_retry_client_errors() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    delivery = await _sink(httpx.MockTransport(handler)).deliver(_escalation())

    assert (delivery.delivered, calls) == (False, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"url": "http://hooks.example.com/x"}, "https"),
        ({"secret": SecretStr("")}, "SECRET"),
        ({"max_retries": 9}, "RETRIES"),
        ({"timeout": 0.0}, "TIMEOUT"),
    ],
)
def test_webhook_settings_are_validated(kwargs: dict[str, object], match: str) -> None:
    settings: dict[str, object] = {"url": _URL, "secret": SecretStr(_SECRET)}
    settings.update(kwargs)

    with pytest.raises(ValueError, match=match):
        WebhookSink(**settings)  # type: ignore[arg-type]


def test_local_http_webhook_is_allowed() -> None:
    WebhookSink(url="http://127.0.0.1:9999/hook", secret=SecretStr(_SECRET))


def test_sink_factory() -> None:
    def config(**values: object) -> AppConfig:
        return AppConfig(_env_file=None, **values)  # type: ignore[arg-type]

    assert isinstance(create_escalation_sink(config()), LogSink)
    webhook = create_escalation_sink(
        config(
            ESCALATION_SINK="webhook",
            ESCALATION_WEBHOOK_URL=_URL,
            ESCALATION_WEBHOOK_SECRET=_SECRET,
        )
    )
    assert isinstance(webhook, WebhookSink) and webhook.forwards
    with pytest.raises(ValueError, match="ESCALATION_WEBHOOK_URL"):
        create_escalation_sink(config(ESCALATION_SINK="webhook"))
    with pytest.raises(ValueError, match="Unknown"):
        create_escalation_sink(config(ESCALATION_SINK="zendesk"))


async def test_log_sink_logs_ids_not_packet_text(caplog: pytest.LogCaptureFixture) -> None:
    escalation = _escalation()

    delivery = await LogSink().deliver(escalation)

    assert delivery.delivered and not LogSink().forwards
    assert "refund" not in caplog.text
