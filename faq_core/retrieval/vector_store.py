"""Dense-vector storage behind the :class:`VectorStore` interface.

:class:`SqliteVecStore` keeps vectors in a ``sqlite-vec`` ``vec0`` virtual table
(``chunk_vec``) inside the support database, partitioned by ``kb_id`` and using
cosine distance. The table is created at runtime rather than in a migration,
because it needs the optional extension and the embedder's dimension. Its
*signature* (embedder name + dimension) is stored in ``index_meta``; when the
configured embedder changes, the table is dropped and re-created empty, and the
startup sync re-embeds every chunk — vectors from different models are never
mixed.

Vectors are written outside the chunk transaction (as an external store such as
Qdrant would be), so the indexer treats them as eventually consistent: search
results whose chunk no longer exists are dropped, and the startup sync adds
missing vectors and removes orphans. Qdrant is not implemented; it would be a
second class satisfying :class:`VectorStore`.
"""

from __future__ import annotations

import sqlite3
import struct
from typing import TYPE_CHECKING, Final, Protocol

from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faq_core.retrieval.chunk_store import ConnectionRunner

_log = get_logger(__name__)

VECTOR_TABLE: Final = "chunk_vec"
_SIGNATURE_KEY: Final = "vector_signature"
_MAX_K: Final = 4096  # sqlite-vec's KNN limit
_ID_BATCH: Final = 500


class VectorStore(Protocol):
    """Nearest-neighbour index of chunk embeddings, keyed by chunk id."""

    @property
    def name(self) -> str: ...

    async def upsert(self, kb_id: str, items: Sequence[tuple[str, Sequence[float]]]) -> None: ...

    async def delete(self, chunk_ids: Sequence[str]) -> None: ...

    async def ids(self, kb_id: str) -> set[str]: ...

    async def search(
        self, kb_id: str, vector: Sequence[float], limit: int
    ) -> list[tuple[str, float]]: ...


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Pack *vector* as little-endian float32, sqlite-vec's compact BLOB format."""
    return struct.pack(f"<{len(vector)}f", *vector)


class SqliteVecStore:
    """``vec0``-backed :class:`VectorStore` on the repository's connection.

    Args:
        runner: The repository (its connection must have sqlite-vec loaded).
        signature: Embedder identity; a change re-creates the table.
        dimension: Vector size of that embedder.
    """

    def __init__(self, runner: ConnectionRunner, *, signature: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._runner = runner
        self._signature = f"{signature}:{dimension}"
        self._dimension = dimension
        self._ready = False

    @property
    def name(self) -> str:
        return "sqlite-vec"

    async def prepare(self) -> bool:
        """Create the table if missing or built for another embedder (idempotent).

        Returns ``True`` when the table was (re-)created and is empty. No lock
        is needed: the check-and-create runs on the single database thread, so
        concurrent callers are serialised and later ones see the new table.
        """
        if self._ready:
            return False
        recreated = await self._runner.run_in_connection(self._prepare_sync)
        self._ready = True
        return recreated

    def _prepare_sync(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (_SIGNATURE_KEY,))
        stored = row.fetchone()
        exists = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (VECTOR_TABLE,)
            ).fetchone()
            is not None
        )
        if exists and stored is not None and stored["value"] == self._signature:
            return False
        try:
            if exists:
                _log.warning(
                    "embedder changed; dropping dense vectors for a full re-embed",
                    extra={"old": stored["value"] if stored else None, "new": self._signature},
                )
                conn.execute(f"DROP TABLE {VECTOR_TABLE}")
            # The dimension is a validated int; vec0 cannot bind DDL parameters.
            conn.execute(
                f"CREATE VIRTUAL TABLE {VECTOR_TABLE} USING vec0("
                "chunk_id TEXT PRIMARY KEY, kb_id TEXT PARTITION KEY, "
                f"embedding FLOAT[{int(self._dimension)}] distance_metric=cosine)"
            )
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (_SIGNATURE_KEY, self._signature),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        return True

    async def upsert(self, kb_id: str, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        if not items:
            return
        await self.prepare()
        for chunk_id, vector in items:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"vector for {chunk_id} has {len(vector)} dims, expected {self._dimension}"
                )

        def _do(conn: sqlite3.Connection) -> None:
            try:
                # vec0 has no UPSERT: delete then insert, in one transaction.
                ids = [chunk_id for chunk_id, _ in items]
                for start in range(0, len(ids), _ID_BATCH):
                    batch = ids[start : start + _ID_BATCH]
                    conn.execute(
                        f"DELETE FROM {VECTOR_TABLE} "  # noqa: S608 — constant table + placeholders
                        f"WHERE chunk_id IN ({', '.join('?' * len(batch))})",
                        batch,
                    )
                conn.executemany(
                    f"INSERT INTO {VECTOR_TABLE} (chunk_id, kb_id, embedding) VALUES (?, ?, ?)",  # noqa: S608 — constant table
                    [(chunk_id, kb_id, serialize_vector(vector)) for chunk_id, vector in items],
                )
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise

        await self._runner.run_in_connection(_do)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        await self.prepare()
        ids = list(chunk_ids)

        def _do(conn: sqlite3.Connection) -> None:
            try:
                for start in range(0, len(ids), _ID_BATCH):
                    batch = ids[start : start + _ID_BATCH]
                    conn.execute(
                        f"DELETE FROM {VECTOR_TABLE} "  # noqa: S608 — constant table + placeholders
                        f"WHERE chunk_id IN ({', '.join('?' * len(batch))})",
                        batch,
                    )
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise

        await self._runner.run_in_connection(_do)

    async def ids(self, kb_id: str) -> set[str]:
        await self.prepare()

        def _do(conn: sqlite3.Connection) -> set[str]:
            rows = conn.execute(
                f"SELECT chunk_id FROM {VECTOR_TABLE} WHERE kb_id = ?",  # noqa: S608 — constant table
                (kb_id,),
            ).fetchall()
            return {row[0] for row in rows}

        return await self._runner.run_in_connection(_do)

    async def search(
        self, kb_id: str, vector: Sequence[float], limit: int
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` nearest first."""
        if limit < 1:
            return []
        await self.prepare()
        blob = serialize_vector(vector)
        k = min(limit, _MAX_K)

        def _do(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            rows = conn.execute(
                f"SELECT chunk_id, distance FROM {VECTOR_TABLE} "  # noqa: S608 — constant table
                "WHERE embedding MATCH ? AND k = ? AND kb_id = ? ORDER BY distance",
                (blob, k, kb_id),
            ).fetchall()
            return [(row[0], 1.0 - float(row[1])) for row in rows]

        return await self._runner.run_in_connection(_do)
