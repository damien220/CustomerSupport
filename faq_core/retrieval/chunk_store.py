"""Chunk persistence and BM25 keyword search (SQLite FTS5).

Chunks live in the ``chunks`` table; ``chunks_fts`` (migration 4) holds a copy
of each chunk's title, heading and content for full-text search, keyed by
``chunk_id``. Both are written in the same transaction, so they cannot drift
apart during normal operation; :meth:`ChunkStore.repair_fts` re-aligns them
after anything unusual (a crash between processes, manual SQL).

All SQL runs on the repository's single connection via
:meth:`~faq_core.repository.FAQRepository.run_in_connection`. Queries are fully
parameterised; the only interpolated identifiers come from the fixed
``_SOURCE_COLUMN`` map.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from faq_core.models import Chunk, FAQEntry, SourceType
from faq_core.repository import parse_timestamp
from faq_core.retrieval.ranking import STOPWORDS, normalize_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

type SourceKey = tuple[SourceType, str]
"""``(source_type, source_id)`` — one FAQ entry or one document."""

_SOURCE_COLUMN: Final[dict[SourceType, str]] = {
    SourceType.FAQ: "faq_id",
    SourceType.DOCUMENT: "document_id",
}
_CHUNK_COLUMNS: Final = (
    "id, kb_id, faq_id, document_id, ordinal, title, heading, content, source_uri, "
    "content_hash, source_hash, updated_at, created_at"
)
_INSERT_CHUNK: Final = (
    f"INSERT INTO chunks ({_CHUNK_COLUMNS}) "  # noqa: S608 — constant column list
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_FTS: Final = (
    "INSERT INTO chunks_fts (chunk_id, kb_id, title, heading, content) VALUES (?, ?, ?, ?, ?)"
)
# bm25() weights follow the FTS5 column order: chunk_id, kb_id (unindexed), title,
# heading, content. A hit in the FAQ question / document title or the section
# heading says more about the topic than one in the body.
_BM25_WEIGHTS: Final = "0.0, 0.0, 2.5, 2.0, 1.0"
_MAX_QUERY_TERMS: Final = 32
_ID_BATCH: Final = 500


class ConnectionRunner(Protocol):
    """Runs a function with the shared SQLite connection on its owning thread."""

    async def run_in_connection[T](self, fn: Callable[[sqlite3.Connection], T]) -> T: ...


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk plus the FAQ entry it was cut from (``None`` for document chunks)."""

    chunk: Chunk
    faq: FAQEntry | None


def build_match_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 query: quoted terms joined with ``OR``.

    Every term is a double-quoted FTS5 string, so user input can never inject
    FTS5 operators (``NEAR``, ``-``, ``*``, column filters). Stopwords are
    dropped unless the query consists of nothing else. Returns ``None`` when the
    query has no word characters at all.
    """
    words = normalize_text(query).split()
    terms = [w for w in words if w not in STOPWORDS] or words
    unique = list(dict.fromkeys(terms))[:_MAX_QUERY_TERMS]
    if not unique:
        return None
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in unique)


class ChunkStore:
    """CRUD for chunks + their FTS5 mirror, and BM25 search over them."""

    def __init__(self, runner: ConnectionRunner) -> None:
        self._runner = runner

    # ── reads ────────────────────────────────────────────────────────────────

    async def source_hashes(self, kb_id: str) -> dict[SourceKey, str]:
        """Map every source that has chunks to the ``source_hash`` they were cut from.

        A source whose chunks disagree on the hash maps to ``""`` (stale).
        """

        def _do(conn: sqlite3.Connection) -> dict[SourceKey, str]:
            rows = conn.execute(
                "SELECT faq_id, document_id, MIN(source_hash) AS source_hash, "
                "COUNT(DISTINCT source_hash) AS versions FROM chunks WHERE kb_id = ? "
                "GROUP BY faq_id, document_id",
                (kb_id,),
            ).fetchall()
            hashes: dict[SourceKey, str] = {}
            for row in rows:
                if row["document_id"]:
                    key: SourceKey = (SourceType.DOCUMENT, row["document_id"])
                elif row["faq_id"]:
                    key = (SourceType.FAQ, row["faq_id"])
                else:
                    continue
                hashes[key] = row["source_hash"] if row["versions"] == 1 else ""
            return hashes

        return await self._runner.run_in_connection(_do)

    async def chunk_ids(self, kb_id: str) -> set[str]:
        def _do(conn: sqlite3.Connection) -> set[str]:
            rows = conn.execute("SELECT id FROM chunks WHERE kb_id = ?", (kb_id,)).fetchall()
            return {row["id"] for row in rows}

        return await self._runner.run_in_connection(_do)

    async def count(self, kb_id: str) -> int:
        def _do(conn: sqlite3.Connection) -> int:
            row = conn.execute("SELECT COUNT(*) FROM chunks WHERE kb_id = ?", (kb_id,)).fetchone()
            return int(row[0])

        return await self._runner.run_in_connection(_do)

    async def list_chunks(self, kb_id: str, source: SourceKey | None = None) -> list[Chunk]:
        """Chunks of *kb_id* (optionally of one *source*), by source then ordinal."""

        def _do(conn: sqlite3.Connection) -> list[Chunk]:
            sql = f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE kb_id = ?"  # noqa: S608 — constant column list
            params: tuple[str, ...] = (kb_id,)
            if source is not None:
                sql += f" AND {_SOURCE_COLUMN[source[0]]} = ?"
                params = (kb_id, source[1])
            rows = conn.execute(sql + " ORDER BY faq_id, document_id, ordinal", params)
            return [_row_to_chunk(row) for row in rows.fetchall()]

        return await self._runner.run_in_connection(_do)

    async def get_chunks(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        """Chunks with the given ids (unknown ids are ignored), in no particular order."""
        ids = list(chunk_ids)

        def _do(conn: sqlite3.Connection) -> list[Chunk]:
            chunks: list[Chunk] = []
            for start in range(0, len(ids), _ID_BATCH):
                batch = ids[start : start + _ID_BATCH]
                rows = conn.execute(
                    f"SELECT {_CHUNK_COLUMNS} FROM chunks "  # noqa: S608 — constant columns + placeholders
                    f"WHERE id IN ({', '.join('?' * len(batch))})",
                    batch,
                ).fetchall()
                chunks.extend(_row_to_chunk(row) for row in rows)
            return chunks

        return await self._runner.run_in_connection(_do)

    async def fetch(self, chunk_ids: Sequence[str]) -> dict[str, StoredChunk]:
        """Chunks by id joined with their FAQ entry; unknown ids are absent."""
        ids = list(dict.fromkeys(chunk_ids))[:_ID_BATCH]
        if not ids:
            return {}

        def _do(conn: sqlite3.Connection) -> dict[str, StoredChunk]:
            columns = ", ".join(f"c.{name.strip()}" for name in _CHUNK_COLUMNS.split(","))
            rows = conn.execute(
                f"SELECT {columns}, f.question AS faq_question, f.answer AS faq_answer, "  # noqa: S608 — constant columns + placeholders
                "f.category AS faq_category, f.kb_id AS faq_kb_id, "
                "f.created_at AS faq_created_at "
                "FROM chunks c LEFT JOIN faq_entries f ON f.id = c.faq_id "
                f"WHERE c.id IN ({', '.join('?' * len(ids))})",
                ids,
            ).fetchall()
            return {row["id"]: _row_to_stored(row) for row in rows}

        return await self._runner.run_in_connection(_do)

    async def search_bm25(self, kb_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` best first; score is ``-bm25`` (higher = better)."""
        match = build_match_query(query)
        if match is None or limit < 1:
            return []

        def _do(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            rows = conn.execute(
                f"SELECT chunk_id, bm25(chunks_fts, {_BM25_WEIGHTS}) AS score "  # noqa: S608 — constant weights
                "FROM chunks_fts WHERE chunks_fts MATCH ? AND kb_id = ? "
                "ORDER BY score LIMIT ?",
                (match, kb_id, limit),
            ).fetchall()
            return [(row["chunk_id"], -float(row["score"])) for row in rows]

        return await self._runner.run_in_connection(_do)

    # ── writes ───────────────────────────────────────────────────────────────

    async def replace_sources(
        self, replacements: Sequence[tuple[SourceKey, Sequence[Chunk]]]
    ) -> list[str]:
        """Atomically swap each source's chunks (and FTS rows) for new ones.

        Returns the ids of the chunks that were removed, so the caller can drop
        their vectors.
        """
        if not replacements:
            return []

        def _do(conn: sqlite3.Connection) -> list[str]:
            removed: list[str] = []
            try:
                for (source_type, source_id), chunks in replacements:
                    removed.extend(_delete_source(conn, source_type, source_id))
                    conn.executemany(_INSERT_CHUNK, [_chunk_to_row(c) for c in chunks])
                    conn.executemany(
                        _INSERT_FTS,
                        [(str(c.id), c.kb_id, c.title, c.heading, c.content) for c in chunks],
                    )
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
            return removed

        return await self._runner.run_in_connection(_do)

    async def delete_sources(self, keys: Sequence[SourceKey]) -> list[str]:
        """Delete every chunk (and FTS row) of *keys*; return the removed chunk ids."""
        if not keys:
            return []

        def _do(conn: sqlite3.Connection) -> list[str]:
            removed: list[str] = []
            try:
                for source_type, source_id in keys:
                    removed.extend(_delete_source(conn, source_type, source_id))
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
            return removed

        return await self._runner.run_in_connection(_do)

    async def repair_fts(self, kb_id: str) -> int:
        """Drop FTS rows without a chunk and add missing ones; return rows touched."""

        def _do(conn: sqlite3.Connection) -> int:
            try:
                dropped = conn.execute(
                    "DELETE FROM chunks_fts WHERE kb_id = ? AND chunk_id NOT IN "
                    "(SELECT id FROM chunks WHERE kb_id = ?)",
                    (kb_id, kb_id),
                ).rowcount
                added = conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, kb_id, title, heading, content) "
                    "SELECT id, kb_id, title, heading, content FROM chunks WHERE kb_id = ? "
                    "AND id NOT IN (SELECT chunk_id FROM chunks_fts WHERE kb_id = ?)",
                    (kb_id, kb_id),
                ).rowcount
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
            return max(dropped, 0) + max(added, 0)

        return await self._runner.run_in_connection(_do)


# ── row mapping ──────────────────────────────────────────────────────────────


def _delete_source(conn: sqlite3.Connection, source_type: SourceType, source_id: str) -> list[str]:
    column = _SOURCE_COLUMN[source_type]
    ids = [
        row[0]
        for row in conn.execute(
            f"SELECT id FROM chunks WHERE {column} = ?",  # noqa: S608 — column from fixed map
            (source_id,),
        )
    ]
    if ids:  # FTS deletes scan the table; skip them for first-time indexing
        conn.execute(
            f"DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE {column} = ?)",  # noqa: S608 — column from fixed map
            (source_id,),
        )
        conn.execute(f"DELETE FROM chunks WHERE {column} = ?", (source_id,))  # noqa: S608 — column from fixed map
    return ids


def _chunk_to_row(chunk: Chunk) -> tuple[object, ...]:
    return (
        str(chunk.id),
        chunk.kb_id,
        str(chunk.faq_id) if chunk.faq_id else None,
        str(chunk.document_id) if chunk.document_id else None,
        chunk.ordinal,
        chunk.title,
        chunk.heading,
        chunk.content,
        chunk.source_uri,
        chunk.content_hash,
        chunk.source_hash,
        chunk.updated_at.isoformat(),
        chunk.created_at.isoformat(),
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    created_at = parse_timestamp(row["created_at"])
    return Chunk(
        id=uuid.UUID(row["id"]),
        kb_id=row["kb_id"],
        faq_id=uuid.UUID(row["faq_id"]) if row["faq_id"] else None,
        document_id=uuid.UUID(row["document_id"]) if row["document_id"] else None,
        ordinal=row["ordinal"],
        title=row["title"],
        heading=row["heading"],
        content=row["content"],
        source_uri=row["source_uri"],
        content_hash=row["content_hash"],
        source_hash=row["source_hash"],
        # Chunks written before migration 4 have no freshness; use their creation.
        updated_at=parse_timestamp(row["updated_at"]) if row["updated_at"] else created_at,
        created_at=created_at,
    )


def _row_to_stored(row: sqlite3.Row) -> StoredChunk:
    chunk = _row_to_chunk(row)
    faq = None
    if chunk.faq_id is not None and row["faq_question"] is not None:
        faq = FAQEntry(
            id=chunk.faq_id,
            question=row["faq_question"],
            answer=row["faq_answer"],
            category=row["faq_category"],
            kb_id=row["faq_kb_id"],
            created_at=parse_timestamp(row["faq_created_at"]),
        )
    return StoredChunk(chunk=chunk, faq=faq)
