"""Structural interfaces the support core depends on.

:class:`faq_core.service.SupportService` and the background re-indexer depend on
these protocols rather than on concrete classes, so the hybrid-retrieval
``RetrievalAnswerEngine`` (``faq_core.retrieval.engine``) and the SQLite
``FAQRepository`` can be swapped — and faked in tests — without inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from faq_core.models import (
        AnswerResponse,
        Document,
        DocumentChange,
        DocumentDraft,
        Escalation,
        EscalationActivity,
        EscalationDelivery,
        EscalationStatus,
        FAQEntry,
        FeedbackActivity,
        FeedbackEntry,
        IndexSyncReport,
        QueryActivity,
        QueryLogRecord,
        RetrievalResponse,
    )


class AnswerEngine(Protocol):
    """Retrieves cited passages for a query, produces a grounded answer, owns the index."""

    @property
    def generates_answers(self) -> bool:
        """Whether :meth:`search` can generate (``False`` in ``LLM_MODE=client``)."""
        ...

    async def search(self, query: str, top_k: int = 3) -> AnswerResponse:
        """Retrieve and generate a grounded answer (only when :attr:`generates_answers`)."""
        ...

    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        """Retrieve cited passages + confidence evidence — no generation, every mode."""
        ...

    async def generate(
        self, query: str, retrieval: RetrievalResponse, *, caveat: bool = False
    ) -> str:
        """Generate an answer from already-retrieved passages (api / offline only)."""
        ...

    async def verify(self, query: str, retrieval: RetrievalResponse, answer: str) -> bool:
        """LLM self-check: is *answer* supported by the passages (api / offline only)."""
        ...

    async def add_document(self, entry: FAQEntry) -> None:
        """Index (or re-index) one FAQ entry."""
        ...

    async def seed_from_entries(self, entries: list[FAQEntry]) -> None: ...

    async def index_document(self, document: Document) -> None:
        """Index (or re-index) one ingested document."""
        ...

    async def sync_index(
        self, entries: Sequence[FAQEntry], documents: Sequence[Document]
    ) -> IndexSyncReport:
        """Reconcile the index with the database contents (startup)."""
        ...


class MetricsStore(Protocol):
    """Read-only access to the support tables for knowledge gaps and resolution metrics."""

    async def list_gap_queries(
        self, kb_id: str, since: datetime, limit: int
    ) -> list[QueryLogRecord]:
        """Logged queries since *since* whose outcome was not ``answer`` or whose
        answer got negative feedback (rating <= 2 or not helpful), newest first."""
        ...

    async def list_query_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[QueryActivity]:
        """The newest *limit* queries logged in ``[since, until)``, oldest first."""
        ...

    async def list_escalation_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[EscalationActivity]:
        """The newest *limit* escalation records created in ``[since, until)``, oldest first."""
        ...

    async def list_feedback_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[FeedbackActivity]:
        """The newest *limit* ratings given in ``[since, until)`` (one per query), oldest first."""
        ...


class FAQStore(MetricsStore, Protocol):
    """Persistence operations the support core needs."""

    async def create(self, entry: FAQEntry) -> FAQEntry: ...

    async def create_many(self, entries: Sequence[FAQEntry]) -> None: ...

    async def get(self, faq_id: str) -> FAQEntry | None: ...

    async def list_all(self, kb_id: str = ...) -> list[FAQEntry]: ...

    async def list_categories(self, kb_id: str = ...) -> list[str]: ...

    async def count(self, kb_id: str = ...) -> int: ...

    async def save_feedback(self, feedback: FeedbackEntry) -> None:
        """Store *feedback*, replacing an earlier rating of the same ``(kb_id, query_id)``."""
        ...

    async def upsert_document(
        self, draft: DocumentDraft, kb_id: str = ...
    ) -> tuple[Document, DocumentChange]: ...

    async def list_documents(self, kb_id: str = ...) -> list[Document]: ...

    # ── support workflow (Phase 4) ───────────────────────────────────────────

    async def log_query(self, record: QueryLogRecord) -> None: ...

    async def get_query(self, query_id: str) -> QueryLogRecord | None: ...

    async def list_session_queries(
        self, kb_id: str, session_id: str, limit: int
    ) -> list[QueryLogRecord]:
        """The session's latest *limit* queries, oldest first."""
        ...

    async def create_escalation(self, escalation: Escalation) -> Escalation: ...

    async def get_escalation(self, escalation_id: str) -> Escalation | None: ...

    async def update_escalation_delivery(
        self, escalation_id: str, status: EscalationStatus, delivery: EscalationDelivery
    ) -> None: ...
