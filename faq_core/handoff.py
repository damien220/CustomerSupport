"""Escalation hand-off: the context packet and pluggable delivery sinks.

Plan decision #3 defers the helpdesk choice, so only two sinks ship:

* :class:`LogSink` (``ESCALATION_SINK=log``, default) — records a structured
  log line (ids and reason codes only, never the packet text); the packet
  stays in the ``escalations`` table for an operator to pick up.
* :class:`WebhookSink` (``ESCALATION_SINK=webhook``) — ``POST``s the packet as
  JSON to ``ESCALATION_WEBHOOK_URL``, signed with HMAC-SHA256 over
  ``"{timestamp}.{body}"`` using ``ESCALATION_WEBHOOK_SECRET``. Headers:
  ``X-Support-Signature: sha256=<hex>``, ``X-Support-Timestamp`` (unix
  seconds), ``X-Support-Event: escalation.created`` and ``Idempotency-Key``
  (the escalation id, so receivers can drop retried duplicates). Every request
  has a timeout; transport errors, 429 and 5xx are retried a bounded number of
  times with exponential backoff; TLS is required for non-local hosts.

The destination is configuration only — never tool/REST input — so an agent
cannot point the server at an arbitrary URL (no SSRF). In ``LLM_MODE=offline``
the network guard also refuses a non-local webhook URL.

A future helpdesk adapter implements :class:`EscalationSink`.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, Protocol

from faq_core.models import EscalationDelivery, utc_now
from faq_core.retrieval.http_json import require_tls
from observability.log import get_logger

if TYPE_CHECKING:
    import httpx
    from pydantic import SecretStr

    from faq_core.config import AppConfig
    from faq_core.models import Escalation, QueryLogRecord, SourceRef, TranscriptTurn

_log = get_logger(__name__)

PACKET_SCHEMA: Final = "support.escalation/v1"
SIGNATURE_HEADER: Final = "X-Support-Signature"
TIMESTAMP_HEADER: Final = "X-Support-Timestamp"
EVENT_HEADER: Final = "X-Support-Event"
MAX_WEBHOOK_RETRIES: Final = 5
_RETRY_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_BACKOFF_SECONDS: Final = 0.5
_MAX_ATTEMPTS_IN_PACKET: Final = 20


class EscalationSink(Protocol):
    """Delivers a stored escalation to whoever handles hand-offs."""

    @property
    def name(self) -> str: ...

    @property
    def forwards(self) -> bool:
        """``True`` when delivery hands the packet to an external system."""
        ...

    async def deliver(self, escalation: Escalation) -> EscalationDelivery:
        """Deliver *escalation*; failures are reported in the receipt, not raised."""
        ...


class LogSink:
    """Logs that an escalation exists (no packet text); the DB row is the hand-off."""

    @property
    def name(self) -> str:
        return "log"

    @property
    def forwards(self) -> bool:
        return False

    async def deliver(self, escalation: Escalation) -> EscalationDelivery:
        _log.warning(
            "escalation created",
            extra={
                "escalation_id": str(escalation.id),
                "kb_id": escalation.kb_id,
                "reasons": escalation.reason,
            },
        )
        return EscalationDelivery(
            sink=self.name, delivered=True, attempts=1, delivered_at=utc_now()
        )


def sign_payload(secret: bytes, timestamp: str, body: bytes) -> str:
    """``sha256=<hex>`` HMAC of ``"{timestamp}.{body}"`` — what receivers recompute."""
    digest = hmac.new(secret, timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return "sha256=" + digest.hexdigest()


def verify_signature(secret: bytes, timestamp: str, body: bytes, signature: str) -> bool:
    """Constant-time check of a received :data:`SIGNATURE_HEADER` value."""
    return hmac.compare_digest(sign_payload(secret, timestamp, body), signature)


class WebhookSink:
    """Signed JSON ``POST`` of the packet (see module docstring).

    Args:
        url: ``ESCALATION_WEBHOOK_URL`` (https, or http to a local host).
        secret: ``ESCALATION_WEBHOOK_SECRET``; required — unsigned hand-offs
            are refused.
        timeout: Seconds per request.
        max_retries: Extra attempts after the first (0-5).
        transport: Injectable httpx transport (tests).
        sleep: Injectable backoff sleep (tests).

    Raises:
        ValueError: bad URL, missing secret, or retries out of range.
    """

    def __init__(
        self,
        *,
        url: str,
        secret: SecretStr,
        timeout: float = 5.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        require_tls(url, setting="ESCALATION_WEBHOOK_URL")
        key = secret.get_secret_value()
        if not key:
            raise ValueError("ESCALATION_WEBHOOK_SECRET is required for ESCALATION_SINK=webhook")
        if not 0 <= max_retries <= MAX_WEBHOOK_RETRIES:
            raise ValueError(f"ESCALATION_WEBHOOK_MAX_RETRIES must be 0..{MAX_WEBHOOK_RETRIES}")
        if timeout <= 0:
            raise ValueError("ESCALATION_WEBHOOK_TIMEOUT must be positive")
        self._url = url
        self._key = key.encode("utf-8")
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "webhook"

    @property
    def forwards(self) -> bool:
        return True

    async def deliver(self, escalation: Escalation) -> EscalationDelivery:
        import httpx

        body = json.dumps(
            {"event": "escalation.created", "packet": escalation.packet},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        attempts = 0
        error = ""
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            while attempts <= self._max_retries:
                attempts += 1
                timestamp = str(int(time.time()))
                headers = {
                    "Content-Type": "application/json",
                    SIGNATURE_HEADER: sign_payload(self._key, timestamp, body),
                    TIMESTAMP_HEADER: timestamp,
                    EVENT_HEADER: "escalation.created",
                    "Idempotency-Key": str(escalation.id),
                }
                try:
                    response = await client.post(self._url, content=body, headers=headers)
                except httpx.TransportError as err:
                    error = f"transport error: {type(err).__name__}"
                else:
                    if response.status_code < 300:  # noqa: PLR2004
                        return EscalationDelivery(
                            sink=self.name,
                            delivered=True,
                            attempts=attempts,
                            delivered_at=utc_now(),
                        )
                    error = f"HTTP {response.status_code}"
                    if response.status_code not in _RETRY_STATUSES:
                        break
                if attempts <= self._max_retries:
                    await self._sleep(_BACKOFF_SECONDS * 2 ** (attempts - 1))
        _log.warning(
            "escalation webhook delivery failed",
            extra={"escalation_id": str(escalation.id), "attempts": attempts, "error": error},
        )
        return EscalationDelivery(sink=self.name, delivered=False, attempts=attempts, error=error)


def create_escalation_sink(config: AppConfig) -> EscalationSink:
    """Build the sink selected by ``ESCALATION_SINK`` (``log`` | ``webhook``).

    Raises:
        ValueError: unknown sink, or webhook settings that are missing/invalid.
    """
    kind = config.ESCALATION_SINK.strip().lower()
    if kind in ("", "log", "none"):
        return LogSink()
    if kind == "webhook":
        if not config.ESCALATION_WEBHOOK_URL.strip():
            raise ValueError("ESCALATION_SINK=webhook needs ESCALATION_WEBHOOK_URL")
        return WebhookSink(
            url=config.ESCALATION_WEBHOOK_URL.strip(),
            secret=config.ESCALATION_WEBHOOK_SECRET,
            timeout=config.ESCALATION_WEBHOOK_TIMEOUT,
            max_retries=config.ESCALATION_WEBHOOK_MAX_RETRIES,
        )
    raise ValueError(f"Unknown ESCALATION_SINK: {kind!r}")


def build_packet(
    *,
    escalation_id: str,
    kb_id: str,
    created_at: str,
    customer_goal: str,
    reasons: list[str],
    reason_detail: str,
    attempts: list[QueryLogRecord],
    transcript: list[TranscriptTurn],
    query_id: str | None,
    session_id: str | None,
    pii_detected: list[str],
) -> dict[str, Any]:
    """The hand-off packet (schema :data:`PACKET_SCHEMA`); every text is already redacted.

    ``sources`` are the distinct citations of the attempts (latest first);
    ``attempts`` the session's queries with their outcome and confidence;
    ``transcript`` the caller's turns, or — when none were given — one
    customer turn per attempt plus the logged answer preview.
    """
    recent = attempts[-_MAX_ATTEMPTS_IN_PACKET:]
    sources: dict[str, SourceRef] = {}
    for attempt in reversed(recent):
        for ref in attempt.citations:
            sources.setdefault(str(ref.citation_id), ref)
    if transcript:
        turns = [{"role": turn.role, "text": turn.text} for turn in transcript]
    else:
        turns = []
        for attempt in recent:
            turns.append(
                {"role": "customer", "text": attempt.query, "at": attempt.created_at.isoformat()}
            )
            if attempt.answer_preview:
                turns.append({"role": "assistant", "text": attempt.answer_preview})
    return {
        "schema": PACKET_SCHEMA,
        "escalation_id": escalation_id,
        "kb_id": kb_id,
        "created_at": created_at,
        "customer_goal": customer_goal,
        "reasons": reasons,
        "reason_detail": reason_detail,
        "query_id": query_id,
        "session_id": session_id,
        "pii_detected": pii_detected,
        "sources": [ref.model_dump(mode="json") for ref in sources.values()],
        "attempts": [
            {
                "query_id": str(attempt.id),
                "query": attempt.query,
                "outcome": attempt.outcome.value,
                "confidence": attempt.confidence,
                "reasons": attempt.reasons,
                "at": attempt.created_at.isoformat(),
            }
            for attempt in recent
        ],
        "transcript": turns,
    }
