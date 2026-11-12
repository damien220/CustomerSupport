"""Chunk persistence + FTS5 BM25 search and sqlite-vec vector storage, on real SQLite."""

from __future__ import annotations

import sqlite3
import uuid

import pytest

from faq_core.models import Chunk, FAQEntry, SourceType
from faq_core.repository import FAQRepository
from faq_core.retrieval.chunk_store import ChunkStore
from faq_core.retrieval.vector_store import SqliteVecStore
from tests.conftest import requires_sqlite_vec


async def _entry(repo: FAQRepository, question: str, kb_id: str = "default") -> FAQEntry:
    return await repo.create(FAQEntry(question=question, answer="A", category="c", kb_id=kb_id))


def _chunk(entry: FAQEntry, content: str, ordinal: int = 0, **extra: object) -> Chunk:
    return Chunk(
        faq_id=entry.id,
        kb_id=entry.kb_id,
        ordinal=ordinal,
        title=entry.question,
        content=content,
        source_hash="v1",
        **extra,  # type: ignore[arg-type]
    )


# ── chunks + FTS5 ────────────────────────────────────────────────────────────


async def test_replace_sources_round_trips_chunks_in_ordinal_order(
    sqlite_repo: FAQRepository,
) -> None:
    store = ChunkStore(sqlite_repo)
    entry = await _entry(sqlite_repo, "Q")
    key = (SourceType.FAQ, str(entry.id))

    await store.replace_sources(
        [(key, [_chunk(entry, "second", 1, heading="Details"), _chunk(entry, "first", 0)])]
    )

    chunks = await store.list_chunks("default", key)
    assert [c.content for c in chunks] == ["first", "second"]
    assert chunks[1].heading == "Details" and chunks[1].title == "Q"
    assert await store.source_hashes("default") == {key: "v1"}


async def test_replacing_a_source_removes_its_old_chunks_and_fts_rows(
    sqlite_repo: FAQRepository,
) -> None:
    store = ChunkStore(sqlite_repo)
    entry = await _entry(sqlite_repo, "Q")
    key = (SourceType.FAQ, str(entry.id))
    old = _chunk(entry, "obsolete wording")
    await store.replace_sources([(key, [old])])

    removed = await store.replace_sources([(key, [_chunk(entry, "fresh wording")])])

    assert removed == [str(old.id)]
    assert await store.search_bm25("default", "obsolete", 5) == []
    assert len(await store.search_bm25("default", "fresh", 5)) == 1


async def test_bm25_ranks_title_matches_and_scopes_by_kb(sqlite_repo: FAQRepository) -> None:
    store = ChunkStore(sqlite_repo)
    password = await _entry(sqlite_repo, "How do I reset my password?")
    shipping = await _entry(sqlite_repo, "How long does shipping take?")
    other_kb = await _entry(sqlite_repo, "Password policy", kb_id="other")
    await store.replace_sources(
        [
            ((SourceType.FAQ, str(password.id)), [_chunk(password, "Use the login page.")]),
            ((SourceType.FAQ, str(shipping.id)), [_chunk(shipping, "Reset? No, 5 days.")]),
            ((SourceType.FAQ, str(other_kb.id)), [_chunk(other_kb, "Rotate passwords.")]),
        ]
    )

    hits = await store.search_bm25("default", "resetting passwords", 5)

    stored = await store.fetch([chunk_id for chunk_id, _ in hits])
    titles = [stored[chunk_id].chunk.title for chunk_id, _ in hits]
    assert titles == ["How do I reset my password?", "How long does shipping take?"]
    assert hits[0][1] > hits[1][1]
    assert stored[hits[0][0]].faq == password


async def test_bm25_tolerates_fts5_syntax_in_user_queries(sqlite_repo: FAQRepository) -> None:
    store = ChunkStore(sqlite_repo)
    entry = await _entry(sqlite_repo, "Q")
    await store.replace_sources([((SourceType.FAQ, str(entry.id)), [_chunk(entry, "near me")])])

    hits = await store.search_bm25("default", 'NEAR( "unbalanced * content:', 5)

    assert len(hits) == 1


async def test_delete_sources_and_repair_fts(sqlite_repo: FAQRepository) -> None:
    store = ChunkStore(sqlite_repo)
    entry = await _entry(sqlite_repo, "Q")
    key = (SourceType.FAQ, str(entry.id))
    chunk = _chunk(entry, "alpha beta")
    await store.replace_sources([(key, [chunk])])

    def _break_fts(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM chunks_fts")
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, kb_id, title, heading, content) "
            "VALUES ('ghost', 'default', '', '', 'ghost text')"
        )
        conn.commit()

    await sqlite_repo.run_in_connection(_break_fts)
    repaired = await store.repair_fts("default")

    assert repaired == 2  # one orphan dropped, one missing row re-added
    assert [cid for cid, _ in await store.search_bm25("default", "alpha", 5)] == [str(chunk.id)]
    assert await store.search_bm25("default", "ghost", 5) == []
    assert await store.delete_sources([key]) == [str(chunk.id)]
    assert await store.count("default") == 0


async def test_chunks_with_disagreeing_source_hashes_are_reported_stale(
    sqlite_repo: FAQRepository,
) -> None:
    store = ChunkStore(sqlite_repo)
    entry = await _entry(sqlite_repo, "Q")
    key = (SourceType.FAQ, str(entry.id))
    mixed = [
        _chunk(entry, "one", 0),
        _chunk(entry, "two", 1).model_copy(update={"source_hash": "x"}),
    ]
    await store.replace_sources([(key, mixed)])

    assert await store.source_hashes("default") == {key: ""}


# ── sqlite-vec ───────────────────────────────────────────────────────────────


@requires_sqlite_vec
async def test_vector_store_upsert_search_delete(sqlite_repo: FAQRepository) -> None:
    store = SqliteVecStore(sqlite_repo, signature="test", dimension=3)

    await store.upsert("default", [("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0])])
    await store.upsert("other", [("c", [1.0, 0.0, 0.0])])
    await store.upsert("default", [("b", [0.9, 0.1, 0.0])])  # replace

    hits = await store.search("default", [1.0, 0.0, 0.0], 5)
    assert [chunk_id for chunk_id, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(1.0)
    assert await store.ids("default") == {"a", "b"}

    await store.delete(["a"])
    assert await store.ids("default") == {"b"}


@requires_sqlite_vec
async def test_vector_store_recreates_table_when_the_embedder_changes(
    sqlite_repo: FAQRepository,
) -> None:
    first = SqliteVecStore(sqlite_repo, signature="model-a", dimension=3)
    assert await first.prepare() is True
    await first.upsert("default", [("a", [1.0, 0.0, 0.0])])

    same = SqliteVecStore(sqlite_repo, signature="model-a", dimension=3)
    assert await same.prepare() is False
    assert await same.ids("default") == {"a"}

    changed = SqliteVecStore(sqlite_repo, signature="model-b", dimension=4)
    assert await changed.prepare() is True
    assert await changed.ids("default") == set()


@requires_sqlite_vec
async def test_vector_store_rejects_wrong_dimension(sqlite_repo: FAQRepository) -> None:
    store = SqliteVecStore(sqlite_repo, signature="test", dimension=3)

    with pytest.raises(ValueError, match="expected 3"):
        await store.upsert("default", [(str(uuid.uuid4()), [1.0, 0.0])])
