"""Tiny typed in-process event bus.

Publishers call :meth:`EventBus.emit` (fire-and-forget: each subscribed handler
runs as its own task, so a slow subscriber such as the re-index scheduler never
delays the caller) or ``await`` :meth:`EventBus.publish` (runs every handler
and waits for all of them). Handler failures are logged, never propagated to the
publisher. Subscriptions match an exact kind, a ``prefix.*`` wildcard, or ``*``.

Payload values are JSON scalars so events stay serialisable if they ever need to
leave the process.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from faq_core.models import utc_now
from observability.log import get_logger

_log = get_logger(__name__)

FAQ_ADDED: Final = "faq.added"
FAQ_INDEXED: Final = "faq.indexed"
QUERY_ANSWERED: Final = "query.answered"
QUERY_ESCALATED: Final = "query.escalated"
GAP_LOGGED: Final = "gap.logged"
ESCALATION_CREATED: Final = "escalation.created"
FEEDBACK_RECEIVED: Final = "feedback.received"
METRICS_REPORTED: Final = "metrics.reported"

_PREVIEW_CHARS: Final = 200

type PayloadValue = str | int | float | bool | None
type EventHandler = Callable[[Event], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable domain event."""

    kind: str
    source: str
    payload: Mapping[str, PayloadValue] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=utc_now)


class EventBus:
    """Async publish/subscribe bus scoped to one process (one per runtime)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._pending: set[asyncio.Task[None]] = set()

    def subscribe(self, pattern: str, handler: EventHandler) -> None:
        """Register *handler* for *pattern* (exact kind, ``prefix.*`` or ``*``).

        Subscribing the same handler to the same pattern twice is a no-op.
        """
        handlers = self._handlers.setdefault(pattern, [])
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(pattern, [])
        if handler in handlers:
            handlers.remove(handler)

    def handlers_for(self, kind: str) -> list[EventHandler]:
        """Return the handlers whose pattern matches *kind*, in subscription order."""
        matched: list[EventHandler] = []
        for pattern, handlers in self._handlers.items():
            if _matches(pattern, kind):
                matched.extend(handlers)
        return matched

    def emit(self, event: Event) -> None:
        """Schedule every matching handler without waiting for it.

        Must be called from inside a running event loop.
        """
        for handler in self.handlers_for(event.kind):
            task = asyncio.create_task(self._invoke(handler, event))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def publish(self, event: Event) -> None:
        """Run every matching handler concurrently and wait for all of them."""
        await asyncio.gather(*(self._invoke(h, event) for h in self.handlers_for(event.kind)))

    async def drain(self) -> None:
        """Wait for every handler scheduled by :meth:`emit` so far (shutdown/tests)."""
        while self._pending:
            await asyncio.gather(*list(self._pending))

    @staticmethod
    async def _invoke(handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:
            # A subscriber bug must not break the publisher's request.
            _log.exception(
                "event handler failed",
                extra={"event_kind": event.kind, "event_id": str(event.id)},
            )


def _matches(pattern: str, kind: str) -> bool:
    if pattern in ("*", kind):
        return True
    return pattern.endswith(".*") and kind.startswith(pattern[:-1])


# ── Event builders (one per published kind) ──────────────────────────────────


def build_faq_added_event(faq_id: str, category: str, kb_id: str) -> Event:
    return Event(
        kind=FAQ_ADDED,
        source="add_faq_entry",
        payload={"faq_id": faq_id, "category": category, "kb_id": kb_id},
    )


def build_faq_indexed_event(faq_id: str, kb_id: str) -> Event:
    """The background job finished indexing an added entry (answer cache is stale)."""
    return Event(kind=FAQ_INDEXED, source="reindex_faq", payload={"faq_id": faq_id, "kb_id": kb_id})


def build_query_answered_event(
    query_id: str,
    query: str,
    answer: str,
    *,
    cached: bool,
    outcome: str = "answer",
    confidence: float = 0.0,
    source: str = "search_faq",
) -> Event:
    """*query* and *answer* must already be PII-redacted."""
    return Event(
        kind=QUERY_ANSWERED,
        source=source,
        payload={
            "query_id": query_id,
            "query": query,
            "answer_preview": answer[:_PREVIEW_CHARS],
            "cached": cached,
            "outcome": outcome,
            "confidence": confidence,
        },
    )


def build_query_escalated_event(
    query_id: str, reasons: str, *, source: str, session_id: str | None
) -> Event:
    """A query's outcome was ``escalate`` (*reasons*: comma-joined codes)."""
    return Event(
        kind=QUERY_ESCALATED,
        source=source,
        payload={"query_id": query_id, "reasons": reasons, "session_id": session_id},
    )


def build_gap_logged_event(query_id: str, query: str, outcome: str, kb_id: str) -> Event:
    """A query entered the knowledge-gap log (*query* is redacted)."""
    return Event(
        kind=GAP_LOGGED,
        source="query_log",
        payload={"query_id": query_id, "query": query, "outcome": outcome, "kb_id": kb_id},
    )


def build_escalation_created_event(
    escalation_id: str, reasons: str, *, sink: str, delivered: bool
) -> Event:
    return Event(
        kind=ESCALATION_CREATED,
        source="create_escalation",
        payload={
            "escalation_id": escalation_id,
            "reasons": reasons,
            "sink": sink,
            "delivered": delivered,
        },
    )


def build_feedback_received_event(feedback_id: str, query_id: str, rating: int) -> Event:
    return Event(
        kind=FEEDBACK_RECEIVED,
        source="record_feedback",
        payload={"feedback_id": feedback_id, "query_id": query_id, "rating": rating},
    )


def build_metrics_reported_event(
    kb_id: str,
    window_days: int,
    *,
    source: str,
    total_queries: int,
    escalation_rate: float | None,
    recontact_rate: float | None,
    knowledge_gaps: int,
) -> Event:
    """Resolution metrics were computed (aggregates only — no query text)."""
    return Event(
        kind=METRICS_REPORTED,
        source=source,
        payload={
            "kb_id": kb_id,
            "window_days": window_days,
            "total_queries": total_queries,
            "escalation_rate": escalation_rate,
            "recontact_rate": recontact_rate,
            "knowledge_gaps": knowledge_gaps,
        },
    )
