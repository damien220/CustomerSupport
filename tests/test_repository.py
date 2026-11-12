from __future__ import annotations

import sqlite3
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from faq_core.confidence import Outcome
from faq_core.config import AppConfig
from faq_core.migrations import LATEST_VERSION
from faq_core.models import (
    DocumentChange,
    DocumentDraft,
    DocumentFormat,
    Escalation,
    EscalationDelivery,
    EscalationStatus,
    FAQEntry,
    FeedbackEntry,
    QueryLogRecord,
    SourceRef,
    SourceType,
    utc_now,
)
from faq_core.repository import FAQRepository


@pytest.fixture
async def repo(app_config: AppConfig) -> AsyncIterator[FAQRepository]:
    repository = FAQRepository(app_config)
    yield repository
    await repository.close()


# ── FAQ entries ──────────────────────────────────────────────────────────────


async def test_create_then_get_round_trips_entry(repo: FAQRepository) -> None:
    entry = FAQEntry(question="How do I cancel?", answer="Go to settings.", category="billing")

    created = await repo.create(entry)
    fetched = await repo.get(str(created.id))

    assert fetched == entry
    assert fetched is not None and fetched.kb_id == "default"


@pytest.mark.parametrize("faq_id", [str(uuid.uuid4()), "not-a-uuid"])
async def test_get_returns_none_for_unknown_id(repo: FAQRepository, faq_id: str) -> None:
    assert await repo.get(faq_id) is None


async def test_list_categories_returns_unique_sorted_categories(repo: FAQRepository) -> None:
    await repo.create(FAQEntry(question="Q1", answer="A1", category="billing"))
    await repo.create(FAQEntry(question="Q2", answer="A2", category="account"))
    await repo.create(FAQEntry(question="Q3", answer="A3", category="billing"))

    assert await repo.list_categories() == ["account", "billing"]


async def test_list_all_and_count_return_every_entry(repo: FAQRepository) -> None:
    await repo.create_many(
        [
            FAQEntry(question="Q1", answer="A1", category="billing"),
            FAQEntry(question="Q2", answer="A2", category="account"),
        ]
    )

    assert len(await repo.list_all()) == 2
    assert await repo.count() == 2


async def test_list_all_is_empty_for_fresh_repository(repo: FAQRepository) -> None:
    assert await repo.list_all() == []
    assert await repo.count() == 0


async def test_queries_are_scoped_by_kb_id(repo: FAQRepository) -> None:
    await repo.create(FAQEntry(question="Q1", answer="A1", category="billing"))
    await repo.create(FAQEntry(question="Q2", answer="A2", category="hr", kb_id="internal"))

    assert await repo.list_categories() == ["billing"]
    assert await repo.list_categories("internal") == ["hr"]
    assert await repo.count("internal") == 1


# ── feedback ─────────────────────────────────────────────────────────────────


async def test_save_feedback_round_trips(repo: FAQRepository) -> None:
    feedback = FeedbackEntry(query_id=uuid.uuid4(), rating=5, comment="Great answer")

    await repo.save_feedback(feedback)

    assert await repo.list_feedback(str(feedback.query_id)) == [feedback]


async def test_save_feedback_replaces_the_earlier_rating_of_the_same_query(
    repo: FAQRepository,
) -> None:
    query_id = uuid.uuid4()
    first = FeedbackEntry(query_id=query_id, rating=1, helpful=False)
    latest = FeedbackEntry(query_id=query_id, rating=4, comment="better now")
    other_kb = FeedbackEntry(query_id=query_id, rating=2, kb_id="other")

    for feedback in (first, latest, other_kb):
        await repo.save_feedback(feedback)

    stored = await repo.list_feedback(str(query_id))
    assert sorted(stored, key=lambda f: f.kb_id) == [latest, other_kb]


# ── new Phase-1 tables ───────────────────────────────────────────────────────


def _draft(body: str = "# Guide\n\nText.", uri: str = "file:///kb/guide.md") -> DocumentDraft:
    return DocumentDraft(
        title="Guide",
        body=body,
        format=DocumentFormat.MARKDOWN,
        source_uri=uri,
        updated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


async def test_upsert_document_adds_then_reports_unchanged(repo: FAQRepository) -> None:
    added, first = await repo.upsert_document(_draft())
    same, second = await repo.upsert_document(_draft())

    assert (first, second) == (DocumentChange.ADDED, DocumentChange.UNCHANGED)
    assert same.id == added.id
    assert await repo.get_document(str(added.id)) == added
    assert added.content_hash


async def test_upsert_document_updates_changed_body_in_place(repo: FAQRepository) -> None:
    original, _ = await repo.upsert_document(_draft())

    updated, change = await repo.upsert_document(_draft(body="# Guide\n\nNew text."))

    assert change is DocumentChange.UPDATED
    assert updated.id == original.id
    assert updated.body.endswith("New text.")
    assert updated.content_hash != original.content_hash
    assert [d.body for d in await repo.list_documents()] == [updated.body]


async def test_documents_are_keyed_by_kb_and_source_uri(repo: FAQRepository) -> None:
    await repo.upsert_document(_draft(uri="file:///a.md"))
    await repo.upsert_document(_draft(uri="file:///b.md"))
    await repo.upsert_document(_draft(uri="file:///a.md"), kb_id="other")

    assert len(await repo.list_documents()) == 2
    assert len(await repo.list_documents("other")) == 1


async def test_vector_extension_is_loaded_when_available(repo: FAQRepository) -> None:
    pytest.importorskip("sqlite_vec")

    assert repo.vector_search_available is True


async def test_repository_degrades_when_sqlite_vec_is_missing(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)  # import now raises ImportError

    repository = FAQRepository(app_config)
    try:
        await repository.create(FAQEntry(question="Q", answer="A", category="c"))
        assert repository.vector_search_available is False
        assert await repository.count() == 1
    finally:
        await repository.close()
    assert "dense retrieval disabled, using BM25 only" in caplog.text


async def test_run_in_connection_uses_the_shared_connection(repo: FAQRepository) -> None:
    def _probe(conn: sqlite3.Connection) -> bool:
        return conn is repo._conn

    assert await repo.run_in_connection(_probe) is True


async def test_query_log_round_trips(repo: FAQRepository) -> None:
    record = QueryLogRecord(
        query="refund?", surface="mcp", top_k=3, confidence=0.4, source_ids=[uuid.uuid4()]
    )

    await repo.log_query(record)

    assert await repo.list_queries() == [record]


async def test_escalations_create_list_and_update_status(repo: FAQRepository) -> None:
    escalation = Escalation(reason="low_confidence", packet={"goal": "refund"})
    await repo.create_escalation(escalation)

    updated = await repo.set_escalation_status(str(escalation.id), EscalationStatus.SENT)
    missing = await repo.set_escalation_status(str(uuid.uuid4()), EscalationStatus.SENT)

    assert (updated, missing) == (True, False)
    assert await repo.list_escalations(status=EscalationStatus.OPEN) == []
    [stored] = await repo.list_escalations(status=EscalationStatus.SENT)
    assert stored.packet == {"goal": "refund"}


# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_schema_is_migrated_to_latest_on_open(repo: FAQRepository) -> None:
    assert await repo.schema_version() == LATEST_VERSION


async def test_file_database_uses_wal_and_a_busy_timeout(repo: FAQRepository) -> None:
    def _pragmas(conn: sqlite3.Connection) -> tuple[str, int]:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        return str(mode), int(timeout)

    assert await repo.run_in_connection(_pragmas) == ("wal", 5000)


async def test_in_memory_database_skips_wal(app_config: AppConfig) -> None:
    memory = FAQRepository(app_config.model_copy(update={"DB_PATH": ":memory:"}))
    try:
        mode = await memory.run_in_connection(
            lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0]
        )
    finally:
        await memory.close()

    assert mode == "memory"


async def test_reuses_a_single_connection_across_calls(repo: FAQRepository) -> None:
    await repo.create(FAQEntry(question="Q1", answer="A1", category="billing"))
    connection_after_first_call = repo._conn

    await repo.list_all()
    await repo.list_categories()

    assert connection_after_first_call is not None
    assert repo._conn is connection_after_first_call


async def test_data_persists_across_repository_instances(app_config: AppConfig) -> None:
    first = FAQRepository(app_config)
    entry = await first.create(FAQEntry(question="Q1", answer="A1", category="billing"))
    await first.close()

    second = FAQRepository(app_config)
    try:
        assert await second.get(str(entry.id)) == entry
    finally:
        await second.close()


async def test_close_disconnects_shuts_down_worker_and_is_idempotent(
    app_config: AppConfig,
) -> None:
    repository = FAQRepository(app_config)
    await repository.create(FAQEntry(question="Q1", answer="A1", category="billing"))

    await repository.close()
    await repository.close()

    assert repository._conn is None
    with pytest.raises(RuntimeError):
        repository._executor.submit(lambda: None)


# ── Phase 4: query log as gap log, escalation delivery ───────────────────────


async def test_query_log_keeps_triage_fields_and_session_order(repo: FAQRepository) -> None:
    ref = SourceRef(
        citation_id=uuid.uuid4(), source_type=SourceType.FAQ, source_id=uuid.uuid4(), title="T"
    )
    first = QueryLogRecord(query="q1", surface="search_kb", top_k=3, session_id="s")
    second = QueryLogRecord(
        query="q2",
        surface="search_kb",
        top_k=3,
        session_id="s",
        outcome=Outcome.ESCALATE,
        reasons=["low_confidence"],
        citations=[ref],
        answer_preview="handoff",
        created_at=first.created_at + timedelta(seconds=1),
    )
    await repo.log_query(first)
    await repo.log_query(second)
    await repo.log_query(QueryLogRecord(query="other", surface="mcp", top_k=1, session_id="t"))

    assert await repo.list_session_queries("default", "s", 10) == [first, second]
    assert await repo.list_session_queries("default", "s", 1) == [second]
    assert await repo.get_query(str(second.id)) == second
    assert await repo.get_query(str(uuid.uuid4())) is None


async def test_gap_queries_include_non_answers_and_badly_rated_answers(
    repo: FAQRepository,
) -> None:
    answered = QueryLogRecord(query="fine", surface="mcp", top_k=3)
    disliked = QueryLogRecord(query="disliked", surface="mcp", top_k=3)
    caveat = QueryLogRecord(
        query="partial", surface="mcp", top_k=3, outcome=Outcome.ANSWER_WITH_CAVEAT
    )
    old = QueryLogRecord(
        query="old",
        surface="mcp",
        top_k=3,
        outcome=Outcome.ESCALATE,
        created_at=utc_now() - timedelta(days=60),
    )
    for record in (answered, disliked, caveat, old):
        await repo.log_query(record)
    await repo.save_feedback(FeedbackEntry(query_id=disliked.id, rating=2))
    await repo.save_feedback(FeedbackEntry(query_id=answered.id, rating=5))

    rows = await repo.list_gap_queries("default", utc_now() - timedelta(days=30), 100)

    assert {row.query for row in rows} == {"disliked", "partial"}


async def test_escalation_delivery_round_trips(repo: FAQRepository) -> None:
    escalation = Escalation(reason="requested", packet={"schema": "x"}, session_id="s")
    await repo.create_escalation(escalation)
    delivery = EscalationDelivery(sink="webhook", delivered=True, attempts=2)

    await repo.update_escalation_delivery(str(escalation.id), EscalationStatus.SENT, delivery)

    stored = await repo.get_escalation(str(escalation.id))
    assert stored is not None
    assert (stored.status, stored.delivery, stored.session_id) == (
        EscalationStatus.SENT,
        delivery,
        "s",
    )
    assert stored.updated_at is not None
    assert await repo.get_escalation(str(uuid.uuid4())) is None
