"""Shared fixtures and boundary fakes.

Fakes stand in only at system boundaries (LLM, answer engine, persistence); the
service, event bus, cache and adapters under test are always the real ones.
"""

from __future__ import annotations

import importlib.util
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pytest

from faq_core.confidence import HitEvidence, Outcome
from faq_core.config import AppConfig
from faq_core.events import EventBus
from faq_core.models import (
    DEFAULT_KB_ID,
    AnswerResponse,
    Citation,
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
    Passage,
    QueryActivity,
    QueryLogRecord,
    RetrievalResponse,
    SourceType,
)
from faq_core.repository import FAQRepository
from faq_core.retrieval.engine import RetrievalStack, build_retrieval_stack
from faq_core.security.policy import SecurityPolicy, build_security
from faq_core.service import ServiceLifespan, SupportService

requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec("sqlite_vec") is None, reason="sqlite-vec is not installed"
)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """AppConfig isolated from the real .env file and pointed at a temp DB."""
    return AppConfig(
        _env_file=None,  # type: ignore[call-arg]
        DB_PATH=str(tmp_path / "test.db"),
        ANTHROPIC_API_KEY="test-anthropic-key",  # type: ignore[arg-type]
        LLM_API_KEY="test-llm-key",  # type: ignore[arg-type]
    )


class FakeLLMClient:
    """Stand-in ``LLMGateway`` — records calls, returns canned answers/intents."""

    name = "fake:stub"
    can_generate = True

    def __init__(
        self,
        *,
        answer: str = "stub answer",
        intent: str = "search_faq",
        supported: bool = True,
    ) -> None:
        self._answer = answer
        self._intent = intent
        self._supported = supported
        self.generate_calls: list[str] = []
        self.generate_contexts: list[list[str]] = []
        self.caveat_flags: list[bool] = []
        self.classify_calls: list[str] = []
        self.verify_calls: list[str] = []
        self.closed = False

    async def generate_answer(
        self, query: str, context_docs: Sequence[str], *, caveat: bool = False
    ) -> str:
        self.generate_calls.append(query)
        self.generate_contexts.append(list(context_docs))
        self.caveat_flags.append(caveat)
        return self._answer

    async def verify_answer(self, query: str, context_docs: Sequence[str], answer: str) -> bool:
        self.verify_calls.append(query)
        return self._supported

    async def classify_intent(self, query: str) -> str:
        self.classify_calls.append(query)
        return self._intent

    async def aclose(self) -> None:
        self.closed = True


FakeGateway = FakeLLMClient
"""Phase-3 name for the fake ``LLMGateway``."""


STRONG_EVIDENCE = HitEvidence(
    term_coverage=1.0,
    retrievers=2,
    bm25_rank=1,
    dense_rank=1,
    similarity=0.9,
    runner_up_similarity=0.2,
    bm25_score=5.0,
    runner_up_bm25=1.0,
)
"""Evidence that scores ``high`` → outcome ``answer``."""

MEDIUM_EVIDENCE = HitEvidence(
    term_coverage=0.5,
    retrievers=2,
    bm25_rank=1,
    dense_rank=2,
    similarity=0.15,
    runner_up_similarity=0.14,
)
"""Evidence that scores ``medium`` → outcome ``answer_with_caveat``."""

WEAK_EVIDENCE = HitEvidence(
    term_coverage=0.0, retrievers=2, dense_rank=1, similarity=0.06, runner_up_similarity=0.05
)
"""Evidence that scores ``low`` → outcome ``escalate``."""


class FakeRAGPipeline:
    """Answer-engine stand-in (``faq_core.ports.AnswerEngine``) — no retrieval stack.

    ``generates_answers=False`` models ``LLM_MODE=client`` (NullGateway).
    ``evidence`` is what :meth:`retrieve` reports for the top passage when
    there are sources (default: strong → outcome ``answer``).
    """

    def __init__(
        self,
        sources: list[FAQEntry] | None = None,
        *,
        generates_answers: bool = True,
        evidence: HitEvidence = STRONG_EVIDENCE,
        supported: bool = True,
    ) -> None:
        self.sources = sources or []
        self.generates_answers = generates_answers
        self.evidence = evidence
        self.supported = supported
        self.generate_caveats: list[bool] = []
        self.verify_calls: list[str] = []
        self.retrieve_calls: list[tuple[str, int]] = []
        self.added: list[FAQEntry] = []
        self.seeded: list[FAQEntry] = []
        self.indexed_documents: list[Document] = []
        self.sync_calls: list[tuple[list[FAQEntry], list[Document]]] = []
        self.search_calls: list[tuple[str, int]] = []

    async def add_document(self, entry: FAQEntry) -> None:
        self.added.append(entry)

    async def seed_from_entries(self, entries: list[FAQEntry]) -> None:
        self.seeded.extend(entries)

    async def index_document(self, document: Document) -> None:
        self.indexed_documents.append(document)

    async def sync_index(
        self, entries: Sequence[FAQEntry], documents: Sequence[Document]
    ) -> IndexSyncReport:
        self.sync_calls.append((list(entries), list(documents)))
        return IndexSyncReport(sources_indexed=len(entries) + len(documents))

    async def search(self, query: str, top_k: int = 3) -> AnswerResponse:
        if not self.generates_answers:
            raise AssertionError("search() must not be called when generation is disabled")
        self.search_calls.append((query, top_k))
        return AnswerResponse(
            query=query,
            answer=f"answer to {query}",
            sources=self.sources[:top_k],
            confidence=0.9 if self.sources else 0.0,
            citations=[_citation(entry) for entry in self.sources[:top_k]],
        )

    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        self.retrieve_calls.append((query, top_k))
        sources = self.sources[:top_k]
        citations = [_citation(entry) for entry in sources]
        passages = [
            Passage(
                rank=rank,
                citation_id=citation.id,
                source_type=citation.source_type,
                source_id=citation.source_id,
                title=citation.title,
                text=entry.answer,
                updated_at=citation.updated_at,
                score=citation.score,
            )
            for rank, (entry, citation) in enumerate(zip(sources, citations, strict=True), 1)
        ]
        return RetrievalResponse(
            query=query,
            passages=passages,
            citations=citations,
            sources=sources,
            confidence=0.9 if sources else 0.0,
            evidence=self.evidence if sources else None,
        )

    async def generate(
        self, query: str, retrieval: RetrievalResponse, *, caveat: bool = False
    ) -> str:
        if not self.generates_answers:
            raise AssertionError("generate() must not be called when generation is disabled")
        # (query, top_k) of the generation — the shape the pre-Phase-4 tests assert on.
        self.search_calls.append((query, self.retrieve_calls[-1][1]))
        self.generate_caveats.append(caveat)
        return f"answer to {query}"

    async def verify(self, query: str, retrieval: RetrievalResponse, answer: str) -> bool:
        if not self.generates_answers:
            raise AssertionError("verify() must not be called when generation is disabled")
        self.verify_calls.append(query)
        return self.supported


def _citation(entry: FAQEntry) -> Citation:
    return Citation(
        id=entry.id,
        source_type=SourceType.FAQ,
        source_id=entry.id,
        title=entry.question,
        snippet=entry.answer,
        updated_at=entry.created_at,
        score=0.9,
    )


class FakeFAQRepository:
    """In-memory ``faq_core.ports.FAQStore`` — no sqlite."""

    def __init__(self, entries: dict[str, FAQEntry] | None = None) -> None:
        self._entries: dict[str, FAQEntry] = dict(entries or {})
        self._documents: dict[tuple[str, str], Document] = {}
        self.saved_feedback: list[FeedbackEntry] = []
        self.queries: list[QueryLogRecord] = []
        self.escalations: dict[str, Escalation] = {}
        self.closed = False

    async def create(self, entry: FAQEntry) -> FAQEntry:
        self._entries[str(entry.id)] = entry
        return entry

    async def create_many(self, entries: Sequence[FAQEntry]) -> None:
        for entry in entries:
            await self.create(entry)

    async def get(self, faq_id: str) -> FAQEntry | None:
        return self._entries.get(faq_id)

    async def list_categories(self, kb_id: str = DEFAULT_KB_ID) -> list[str]:
        return sorted({e.category for e in self._entries.values() if e.kb_id == kb_id})

    async def list_all(self, kb_id: str = DEFAULT_KB_ID) -> list[FAQEntry]:
        return [e for e in self._entries.values() if e.kb_id == kb_id]

    async def count(self, kb_id: str = DEFAULT_KB_ID) -> int:
        return len(await self.list_all(kb_id))

    async def save_feedback(self, feedback: FeedbackEntry) -> None:
        # Mirrors the SQLite upsert: one rating per (kb_id, query_id), latest wins.
        self.saved_feedback = [
            f
            for f in self.saved_feedback
            if (f.kb_id, f.query_id) != (feedback.kb_id, feedback.query_id)
        ]
        self.saved_feedback.append(feedback)

    async def upsert_document(
        self, draft: DocumentDraft, kb_id: str = DEFAULT_KB_ID
    ) -> tuple[Document, DocumentChange]:
        key = (kb_id, draft.source_uri)
        existing = self._documents.get(key)
        fields = draft.model_dump()
        if existing is not None and existing.body == draft.body:
            return existing, DocumentChange.UNCHANGED
        if existing is not None:
            document = existing.model_copy(update=fields)
            change = DocumentChange.UPDATED
        else:
            document = Document(**fields, kb_id=kb_id)
            change = DocumentChange.ADDED
        self._documents[key] = document
        return document, change

    async def list_documents(self, kb_id: str = DEFAULT_KB_ID) -> list[Document]:
        return [d for (kb, _), d in self._documents.items() if kb == kb_id]

    async def log_query(self, record: QueryLogRecord) -> None:
        self.queries.append(record)

    async def get_query(self, query_id: str) -> QueryLogRecord | None:
        return next((q for q in self.queries if str(q.id) == query_id), None)

    async def list_session_queries(
        self, kb_id: str, session_id: str, limit: int
    ) -> list[QueryLogRecord]:
        rows = [q for q in self.queries if q.kb_id == kb_id and q.session_id == session_id]
        return rows[-limit:]

    async def list_gap_queries(
        self, kb_id: str, since: datetime, limit: int
    ) -> list[QueryLogRecord]:
        bad = {str(f.query_id) for f in self.saved_feedback if not f.helpful or f.rating <= 2}
        rows = [
            q
            for q in self.queries
            if q.kb_id == kb_id
            and q.created_at >= since
            and (q.outcome is not Outcome.ANSWER or str(q.id) in bad)
        ]
        return list(reversed(rows))[:limit]

    async def list_query_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[QueryActivity]:
        rows = [q for q in self.queries if q.kb_id == kb_id and since <= q.created_at < until]
        return [
            QueryActivity(
                id=q.id,
                session_id=q.session_id,
                outcome=q.outcome,
                confidence=q.confidence,
                created_at=q.created_at,
            )
            for q in sorted(rows, key=lambda q: q.created_at)[-limit:]
        ]

    async def list_escalation_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[EscalationActivity]:
        return [
            EscalationActivity(status=e.status, created_at=e.created_at)
            for e in self.escalations.values()
            if e.kb_id == kb_id and since <= e.created_at < until
        ][-limit:]

    async def list_feedback_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[FeedbackActivity]:
        return [
            FeedbackActivity(rating=f.rating, helpful=f.helpful, created_at=f.created_at)
            for f in self.saved_feedback
            if f.kb_id == kb_id and since <= f.created_at < until
        ][-limit:]

    async def create_escalation(self, escalation: Escalation) -> Escalation:
        self.escalations[str(escalation.id)] = escalation
        return escalation

    async def get_escalation(self, escalation_id: str) -> Escalation | None:
        return self.escalations.get(escalation_id)

    async def update_escalation_delivery(
        self, escalation_id: str, status: EscalationStatus, delivery: EscalationDelivery
    ) -> None:
        current = self.escalations[escalation_id]
        self.escalations[escalation_id] = current.model_copy(
            update={"status": status, "delivery": delivery}
        )

    async def close(self) -> None:
        self.closed = True


def make_entry(question: str = "Reset password?", category: str = "account") -> FAQEntry:
    return FAQEntry(question=question, answer="Use the settings page.", category=category)


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_rag_pipeline() -> FakeRAGPipeline:
    """An engine with one confidently matching source (outcome ``answer``)."""
    return FakeRAGPipeline(sources=[make_entry()])


@pytest.fixture
def fake_faq_repository() -> FakeFAQRepository:
    return FakeFAQRepository()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def service(
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> SupportService:
    return SupportService(engine=fake_rag_pipeline, store=fake_faq_repository, events=event_bus)


@pytest.fixture
async def sqlite_repo(app_config: AppConfig) -> AsyncIterator[FAQRepository]:
    """A real, migrated repository on a temp file (closed after the test)."""
    repository = FAQRepository(app_config)
    yield repository
    await repository.close()


@pytest.fixture
def retrieval_stack(app_config: AppConfig, sqlite_repo: FAQRepository) -> RetrievalStack:
    """The real lite retrieval stack (hashing embedder, sqlite-vec, no reranker)."""
    return build_retrieval_stack(app_config, sqlite_repo)


def sample_entries() -> list[FAQEntry]:
    return [
        FAQEntry(
            question="How do I reset my password?",
            answer="Open the login page and click 'Forgot Password' to receive a reset link.",
            category="Account",
        ),
        FAQEntry(
            question="How long does shipping take?",
            answer="Standard shipping takes 5-7 business days.",
            category="Shipping",
        ),
        FAQEntry(
            question="What payment methods do you accept?",
            answer="We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
            category="Billing",
        ),
    ]


def static_lifespan(service: SupportService) -> ServiceLifespan:
    """A ``ServiceLifespan`` that yields an already-built *service*."""

    @asynccontextmanager
    async def _lifespan() -> AsyncIterator[SupportService]:
        yield service

    return _lifespan


# ── HTTP-surface security (Phase 5) ──────────────────────────────────────────
# Tokens are generated per run (never literals), so no test file carries a
# credential-shaped string.

TEST_KEY_NAME = "tests"


def make_api_token() -> str:
    """A fresh static bearer token (43 URL-safe characters)."""
    return secrets.token_urlsafe(32)


def bearer(token: str) -> dict[str, str]:
    """The Authorization header for *token*."""
    scheme = "Bearer"
    return {"Authorization": f"{scheme} {token}"}


def security_config(**overrides: object) -> AppConfig:
    """AppConfig for security tests: isolated from .env, TestClient's host allowed."""
    values: dict[str, object] = {"ALLOWED_HOSTS": "testserver", "AUTH_MODE": "token"}
    values.update(overrides)
    return AppConfig(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def token_security(
    token: str, *, bind_host: str = "127.0.0.1", **overrides: object
) -> SecurityPolicy:
    """A static-token policy accepting exactly *token* (key name ``tests``)."""
    config = security_config(API_KEYS=f"{TEST_KEY_NAME}:{token}", **overrides)
    return build_security(config, bind_host=bind_host)


@pytest.fixture
def api_token() -> str:
    return make_api_token()
