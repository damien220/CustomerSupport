"""Hybrid retrieval end to end on real SQLite: indexing, sync, fusion, dedupe, citations.

Only the LLM (answer text) and, where a test says so, the reranker are fakes.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from faq_core.config import AppConfig
from faq_core.models import (
    DocumentDraft,
    DocumentFormat,
    FAQEntry,
    SourceType,
)
from faq_core.repository import FAQRepository
from faq_core.retrieval import chunking
from faq_core.retrieval.embedders import HashingEmbedder
from faq_core.retrieval.engine import (
    RetrievalAnswerEngine,
    RetrievalStack,
    build_retrieval_stack,
)
from faq_core.retrieval.rerankers import RerankerError, RerankResult
from tests.conftest import FakeLLMClient, requires_sqlite_vec, sample_entries


async def _seed(repo: FAQRepository, stack: RetrievalStack) -> list[FAQEntry]:
    entries = sample_entries()
    await repo.create_many(entries)
    await stack.indexer.index_entries(entries)
    return entries


# ── retrieval ────────────────────────────────────────────────────────────────


@requires_sqlite_vec
async def test_hybrid_retrieval_finds_the_relevant_entry_with_both_signals(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    await _seed(sqlite_repo, retrieval_stack)

    hits = await retrieval_stack.retriever.retrieve("I need to reset my password", 3)

    assert retrieval_stack.retriever.dense_enabled
    assert hits[0].chunk.title == "How do I reset my password?"
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank == 1
    assert hits[0].confidence == pytest.approx(1.0)
    assert hits[0].faq is not None and hits[0].faq.category == "Account"


async def test_retrieval_degrades_to_bm25_only_without_sqlite_vec(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    repo = FAQRepository(app_config)
    try:
        stack = build_retrieval_stack(app_config, repo)
        await _seed(repo, stack)
        report = await stack.indexer.sync(await repo.list_all(), [])

        hits = await stack.retriever.retrieve("which payment methods are accepted", 2)
    finally:
        await repo.close()

    assert stack.vectors is None and report.dense_enabled is False
    assert hits[0].chunk.title == "What payment methods do you accept?"
    assert hits[0].dense_rank is None
    assert "using BM25 only" in caplog.text


async def test_vector_store_none_disables_dense_retrieval(
    app_config: AppConfig, sqlite_repo: FAQRepository
) -> None:
    stack = build_retrieval_stack(
        app_config.model_copy(update={"VECTOR_STORE": "none"}), sqlite_repo
    )

    assert stack.vectors is None
    with pytest.raises(ValueError, match="VECTOR_STORE"):
        build_retrieval_stack(app_config.model_copy(update={"VECTOR_STORE": "x"}), sqlite_repo)


@requires_sqlite_vec
async def test_dense_only_mode_uses_vectors(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    await _seed(sqlite_repo, retrieval_stack)

    hits = await retrieval_stack.retriever.retrieve("shipping days", 1, mode="dense")

    assert hits[0].chunk.title == "How long does shipping take?"
    assert hits[0].bm25_rank is None and hits[0].dense_rank == 1


async def test_near_duplicate_entries_collapse_to_one_hit(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    answer = "Refunds are issued to the original card within 5 business days."
    copies = [
        FAQEntry(question="How do refunds work?", answer=answer, category="Billing"),
        FAQEntry(question="How do refunds work?", answer=answer + " ", category="Returns"),
        FAQEntry(
            question="When will I get my refund?", answer="Refunds take 5 days.", category="B"
        ),
    ]
    await sqlite_repo.create_many(copies)
    await retrieval_stack.indexer.index_entries(copies)

    hits = await retrieval_stack.retriever.retrieve("refunds", 5)

    titles = [hit.chunk.title for hit in hits]
    assert titles.count("How do refunds work?") == 1
    assert "When will I get my refund?" in titles


async def test_retrieve_returns_nothing_for_blank_or_unmatched_queries(
    sqlite_repo: FAQRepository, app_config: AppConfig
) -> None:
    stack = build_retrieval_stack(
        app_config.model_copy(update={"VECTOR_STORE": "none"}), sqlite_repo
    )
    await _seed(sqlite_repo, stack)

    assert await stack.retriever.retrieve("   ", 3) == []
    assert await stack.retriever.retrieve("zebra xylophone", 3) == []


class _ReversingReranker:
    name = "reverse"

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        order = list(reversed(range(len(documents))))[:top_n]
        return [RerankResult(index=i, score=0.5) for i in order]


class _FailingReranker:
    name = "failing"

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        raise RerankerError("down")


async def test_reranker_reorders_and_sets_confidence(
    app_config: AppConfig, sqlite_repo: FAQRepository
) -> None:
    stack = build_retrieval_stack(app_config, sqlite_repo, reranker=_ReversingReranker())
    await _seed(sqlite_repo, stack)
    plain = build_retrieval_stack(app_config, sqlite_repo)

    baseline = await plain.retriever.retrieve("password shipping payment", 3)
    reranked = await stack.retriever.retrieve("password shipping payment", 3)

    assert [h.chunk.id for h in reranked] == [h.chunk.id for h in reversed(baseline)]
    assert {h.confidence for h in reranked} == {0.5}


async def test_reranker_failure_keeps_fused_order(
    app_config: AppConfig, sqlite_repo: FAQRepository, caplog: pytest.LogCaptureFixture
) -> None:
    stack = build_retrieval_stack(app_config, sqlite_repo, reranker=_FailingReranker())
    await _seed(sqlite_repo, stack)

    hits = await stack.retriever.retrieve("reset password", 2)

    assert hits[0].chunk.title == "How do I reset my password?"
    assert "reranker failed" in caplog.text


# ── documents + citations ────────────────────────────────────────────────────


async def test_document_chunks_become_citations_with_heading_and_freshness(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    updated = datetime(2026, 9, 1, 12, tzinfo=UTC)
    document, _ = await sqlite_repo.upsert_document(
        DocumentDraft(
            title="Billing guide",
            body="# Billing guide\n\nGeneral.\n\n## Refunds\n\nRefunds reach your card in 5 days.",
            format=DocumentFormat.MARKDOWN,
            source_uri="https://kb.example.com/billing",
            updated_at=updated,
        )
    )
    await retrieval_stack.indexer.index_documents([document])
    engine = RetrievalAnswerEngine(
        retriever=retrieval_stack.retriever,
        indexer=retrieval_stack.indexer,
        llm=FakeLLMClient(answer="Five days."),
    )

    response = await engine.search("how long do refunds take", top_k=1)

    [citation] = response.citations
    assert citation.source_type is SourceType.DOCUMENT
    assert citation.source_id == document.id
    assert (citation.title, citation.heading) == ("Billing guide", "Billing guide > Refunds")
    assert citation.snippet == "Refunds reach your card in 5 days."
    assert citation.source_uri == "https://kb.example.com/billing"
    assert citation.updated_at == updated
    assert 0.0 < citation.score <= 1.0
    assert response.sources == []  # documents are cited, not returned as FAQ entries
    assert response.answer == "Five days."


async def test_engine_search_returns_faq_sources_and_passes_context_to_llm(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    entries = await _seed(sqlite_repo, retrieval_stack)
    llm = FakeLLMClient()
    engine = RetrievalAnswerEngine(
        retriever=retrieval_stack.retriever, indexer=retrieval_stack.indexer, llm=llm
    )

    response = await engine.search("how long does shipping take", top_k=2)

    assert response.sources[0] == entries[1]
    assert response.citations[0].source_id == entries[1].id
    assert response.citations[0].updated_at == entries[1].created_at
    assert response.confidence == response.citations[0].score
    assert llm.generate_calls == ["how long does shipping take"]


# ── sync (startup re-index) ──────────────────────────────────────────────────


async def test_sync_indexes_unindexed_entries_and_is_idempotent(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    entries = sample_entries()
    await sqlite_repo.create_many(entries)  # e.g. Phase-1 rows, or a lost re-index job

    first = await retrieval_stack.indexer.sync(entries, [])
    second = await retrieval_stack.indexer.sync(entries, [])

    assert first.sources_indexed == 3 and first.chunks == 3
    assert first.vectors_embedded == (3 if retrieval_stack.vectors else 0)
    assert (second.sources_indexed, second.vectors_embedded, second.fts_repaired) == (0, 0, 0)
    hits = await retrieval_stack.retriever.retrieve("reset password", 1)
    assert hits[0].chunk.title == "How do I reset my password?"


async def test_sync_removes_chunks_of_deleted_sources(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    entries = await _seed(sqlite_repo, retrieval_stack)

    report = await retrieval_stack.indexer.sync(entries[:2], [])

    assert report.sources_removed == 1 and report.chunks == 2
    if retrieval_stack.vectors is not None:
        assert report.vectors_removed == 1
    hits = await retrieval_stack.retriever.retrieve("payment methods visa", 3)
    assert "What payment methods do you accept?" not in {hit.chunk.title for hit in hits}


async def test_sync_rechunks_everything_after_a_chunker_upgrade(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = await _seed(sqlite_repo, retrieval_stack)
    monkeypatch.setattr(
        "faq_core.retrieval.indexer.CHUNKER_VERSION", chunking.CHUNKER_VERSION + "x"
    )

    report = await retrieval_stack.indexer.sync(entries, [])

    assert report.sources_indexed == 3 and report.chunks == 3


@requires_sqlite_vec
async def test_sync_repairs_missing_fts_rows_and_vectors(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    entries = await _seed(sqlite_repo, retrieval_stack)

    def _wipe(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM chunk_vec")
        conn.commit()

    await sqlite_repo.run_in_connection(_wipe)
    report = await retrieval_stack.indexer.sync(entries, [])

    assert (report.sources_indexed, report.fts_repaired, report.vectors_embedded) == (0, 3, 3)


@requires_sqlite_vec
async def test_sync_reembeds_everything_when_the_embedder_changes(
    app_config: AppConfig, sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    entries = await _seed(sqlite_repo, retrieval_stack)
    other = build_retrieval_stack(app_config, sqlite_repo, embedder=HashingEmbedder(dimension=64))

    report = await other.indexer.sync(entries, [])

    assert report.sources_indexed == 0
    assert report.vectors_embedded == 3
    hits = await other.retriever.retrieve("shipping", 1, mode="dense")
    assert hits[0].chunk.title == "How long does shipping take?"


async def test_sync_indexes_documents_and_reindexes_changed_ones(
    sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack
) -> None:
    draft = DocumentDraft(title="Doc", body="# Doc\n\nAlpha topic.", source_uri="file:///doc.md")
    document, _ = await sqlite_repo.upsert_document(draft)
    await retrieval_stack.indexer.sync([], [document])
    changed, _ = await sqlite_repo.upsert_document(
        draft.model_copy(update={"body": "# Doc\n\nBeta topic."})
    )

    report = await retrieval_stack.indexer.sync([], [changed])

    assert report.sources_indexed == 1
    # Dense KNN always returns neighbours, so check the keyword side for removal.
    assert await retrieval_stack.retriever.retrieve("alpha", 3, mode="bm25") == []
    [hit] = await retrieval_stack.retriever.retrieve("beta", 3)
    assert hit.chunk.document_id == document.id
    assert hit.chunk.content == "Beta topic."
