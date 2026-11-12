"""SupportService behavior (replaces the former FAQController tests)."""

from __future__ import annotations

import uuid

import pytest

from faq_core.errors import FAQNotFoundError, InvalidRequestError, QueryNotFoundError
from faq_core.events import FAQ_ADDED, FEEDBACK_RECEIVED, QUERY_ANSWERED, Event, EventBus
from faq_core.models import (
    Citation,
    DocumentDraft,
    FAQDraft,
    IngestBatch,
    RetrievalResponse,
    SkippedSource,
    SourceType,
)
from faq_core.service import SupportService
from tests.conftest import FakeFAQRepository, FakeRAGPipeline, make_entry


@pytest.fixture
def recorded(event_bus: EventBus) -> list[Event]:
    events: list[Event] = []

    async def _record(event: Event) -> None:
        events.append(event)

    event_bus.subscribe("*", _record)
    return events


# ── search ───────────────────────────────────────────────────────────────────


async def test_search_returns_answer_with_sources_and_query_id(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    source = make_entry()
    engine = FakeRAGPipeline(sources=[source])
    service = SupportService(engine=engine, store=fake_faq_repository, events=event_bus)

    result = await service.search("hello", top_k=3)

    assert result.answer == "answer to hello"
    assert result.sources == [source]
    assert result.confidence == result.confidence_detail.score > 0.55  # type: ignore[union-attr]
    assert (result.outcome, result.escalate) == ("answer", False)
    assert result.cached is False
    assert isinstance(result.query_id, uuid.UUID)
    assert engine.search_calls == [("hello", 3)]


async def test_search_defaults_top_k_to_three(
    service: SupportService, fake_rag_pipeline: FakeRAGPipeline
) -> None:
    await service.search("hours")

    assert fake_rag_pipeline.search_calls == [("hours", 3)]


async def test_repeated_search_is_served_from_cache_with_fresh_query_id(
    service: SupportService, fake_rag_pipeline: FakeRAGPipeline
) -> None:
    first = await service.search("hello")
    second = await service.search("hello")

    assert len(fake_rag_pipeline.search_calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert first.answer == second.answer
    assert first.query_id != second.query_id


async def test_distinct_queries_and_top_k_are_not_shared(
    service: SupportService, fake_rag_pipeline: FakeRAGPipeline
) -> None:
    await service.search("a")
    await service.search("b")
    await service.search("a", top_k=5)

    assert len(fake_rag_pipeline.search_calls) == 3


@pytest.mark.parametrize(("query", "top_k"), [("", 3), ("   ", 3), ("ok", 0), ("ok", 21)])
async def test_search_rejects_invalid_input(
    service: SupportService, query: str, top_k: int
) -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        await service.search(query, top_k)

    assert exc_info.value.to_dict()["error"]["code"] == "invalid_request"


async def test_search_emits_query_answered(
    service: SupportService, event_bus: EventBus, recorded: list[Event]
) -> None:
    result = await service.search("hello")
    await event_bus.drain()

    assert [e.kind for e in recorded] == [QUERY_ANSWERED]
    assert recorded[0].payload["query_id"] == str(result.query_id)


# ── entries ──────────────────────────────────────────────────────────────────


async def test_add_entry_persists_and_emits_faq_added(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
    recorded: list[Event],
) -> None:
    entry = await service.add_entry("Do you ship overseas?", "Yes.", "Shipping")
    await event_bus.drain()

    assert await fake_faq_repository.get(str(entry.id)) == entry
    assert entry.kb_id == "default"
    assert [e.kind for e in recorded] == [FAQ_ADDED]
    assert recorded[0].payload == {
        "faq_id": str(entry.id),
        "category": "Shipping",
        "kb_id": "default",
    }


async def test_add_entry_rejects_blank_fields(service: SupportService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.add_entry("Q?", " ", "Billing")


async def test_get_entry_returns_matching_entry(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    entry = await fake_faq_repository.create(make_entry())

    assert await service.get_entry(str(entry.id)) == entry


async def test_get_entry_unknown_id_raises_not_found(service: SupportService) -> None:
    missing = str(uuid.uuid4())

    with pytest.raises(FAQNotFoundError) as exc_info:
        await service.get_entry(missing)

    assert exc_info.value.status_code == 404
    assert exc_info.value.to_dict()["error"] == {
        "code": "faq_not_found",
        "message": f"FAQ entry {missing!r} not found",
        "faq_id": missing,
    }


async def test_list_categories_returns_sorted_unique(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    for category in ("Shipping", "Account", "Shipping", "Billing"):
        await fake_faq_repository.create(make_entry(category=category))

    result = await service.list_categories()

    assert result.categories == ["Account", "Billing", "Shipping"]
    assert result.kb_id == "default"


# ── feedback ─────────────────────────────────────────────────────────────────


async def test_record_feedback_persists_and_emits(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
    recorded: list[Event],
) -> None:
    query_id = (await service.search_kb("hello")).query_id
    await event_bus.drain()
    recorded.clear()

    receipt = await service.record_feedback(str(query_id), 5, "Great")
    await event_bus.drain()

    [saved] = fake_faq_repository.saved_feedback
    assert receipt.status == "recorded"
    assert receipt.feedback_id == saved.id
    assert saved.query_id == query_id
    assert [e.kind for e in recorded] == [FEEDBACK_RECEIVED]


async def test_record_feedback_keeps_only_the_latest_rating_per_query(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    query_id = (await service.search_kb("hello")).query_id

    await service.record_feedback(query_id, 1, helpful=False)
    latest = await service.record_feedback(query_id, 5)

    [saved] = fake_faq_repository.saved_feedback
    assert (saved.id, saved.rating, saved.helpful) == (latest.feedback_id, 5, True)


async def test_record_feedback_rejects_an_unknown_or_foreign_query(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    foreign = (await service.search_kb("hello")).query_id
    fake_faq_repository.queries[0] = fake_faq_repository.queries[0].model_copy(
        update={"kb_id": "other"}
    )

    for query_id in (uuid.uuid4(), foreign):
        with pytest.raises(QueryNotFoundError) as caught:
            await service.record_feedback(query_id, 1)
        assert caught.value.to_dict()["error"]["code"] == "query_not_found"
    assert fake_faq_repository.saved_feedback == []


@pytest.mark.parametrize(("query_id", "rating"), [("not-a-uuid", 3), (str(uuid.uuid4()), 6)])
async def test_record_feedback_rejects_invalid_input(
    service: SupportService, query_id: str, rating: int
) -> None:
    with pytest.raises(InvalidRequestError):
        await service.record_feedback(query_id, rating)


# ── seeding ──────────────────────────────────────────────────────────────────


async def test_seed_if_empty_loads_store_and_engine_once(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    fake_rag_pipeline: FakeRAGPipeline,
) -> None:
    drafts = [
        FAQDraft(question="Q1", answer="A1", category="billing"),
        FAQDraft(question="Q2", answer="A2", category="account"),
    ]

    first = await service.seed_if_empty(drafts)
    second = await service.seed_if_empty(drafts)

    assert (first, second) == (2, 0)
    assert await fake_faq_repository.count() == 2
    assert len(fake_rag_pipeline.seeded) == 2


# ── citations, index sync, ingest ────────────────────────────────────────────


async def test_search_passes_engine_citations_through(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    source = make_entry()
    citation = Citation(
        id=uuid.uuid4(),
        source_type=SourceType.FAQ,
        source_id=source.id,
        title=source.question,
        snippet="Use the settings page.",
        updated_at=source.created_at,
        score=0.8,
    )

    class _CitingEngine(FakeRAGPipeline):
        async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
            retrieved = await super().retrieve(query, top_k)
            return retrieved.model_copy(update={"citations": [citation]})

    service = SupportService(
        engine=_CitingEngine(sources=[source]), store=fake_faq_repository, events=event_bus
    )

    first = await service.search("q")
    cached = await service.search("q")

    assert first.citations == [citation]
    assert cached.cached is True and cached.citations == [citation]


async def test_sync_index_hands_every_entry_and_document_to_the_engine(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    fake_rag_pipeline: FakeRAGPipeline,
) -> None:
    entry = await fake_faq_repository.create(make_entry())
    document, _ = await fake_faq_repository.upsert_document(
        DocumentDraft(title="T", body="B", source_uri="file:///t.md")
    )

    report = await service.sync_index()

    assert fake_rag_pipeline.sync_calls == [([entry], [document])]
    assert report.sources_indexed == 2


async def test_ingest_adds_new_faqs_skips_duplicates_and_indexes_changed_documents(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    fake_rag_pipeline: FakeRAGPipeline,
) -> None:
    await fake_faq_repository.create(make_entry("Reset password?"))
    doc = DocumentDraft(title="Guide", body="# Guide", source_uri="file:///g.md")
    batch = IngestBatch(
        source="file:///kb",
        faqs=[
            FAQDraft(question="reset  PASSWORD", answer="dup", category="a"),
            FAQDraft(question="New question?", answer="New.", category="a"),
            FAQDraft(question="new question", answer="dup in batch", category="a"),
        ],
        documents=[doc],
        skipped=[SkippedSource(source="x.png", reason="unsupported file type")],
    )

    first = await service.ingest(batch)
    second = await service.ingest(batch.model_copy(update={"faqs": []}))

    assert (first.faqs_added, first.faqs_duplicate, first.documents_added) == (1, 2, 1)
    assert first.skipped == batch.skipped
    assert [e.question for e in fake_rag_pipeline.seeded] == ["New question?"]
    assert len(fake_rag_pipeline.indexed_documents) == 1  # unchanged doc not re-indexed
    assert second.documents_unchanged == 1
