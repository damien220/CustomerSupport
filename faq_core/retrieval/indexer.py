"""Turns FAQ entries and documents into indexed chunks, and keeps the index in sync.

Each source (one FAQ entry or one document) is chunked, its chunks are written
with their FTS5 rows in one transaction, and their embeddings are stored in the
vector store. Every chunk records the ``source_hash`` of the source version it
was cut from — a fingerprint of the content, freshness date, chunker version
and chunk size — so :meth:`KnowledgeIndexer.sync` can tell exactly which
sources are missing, stale or gone.

:meth:`KnowledgeIndexer.sync` runs at every startup (see :func:`bootstrap.serve`).
It fixes the Phase-1 bug where the index was only built while seeding an empty
database (so an in-memory vector store came back empty after a restart): the
index now lives in the database file, and the sync re-chunks stale or missing
sources, drops chunks of deleted ones, repairs FTS rows, and embeds every chunk
that has no vector (all of them after an embedder change).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from faq_core.models import Chunk, DocumentFormat, FAQEntry, IndexSyncReport, SourceType
from faq_core.retrieval.chunking import (
    CHUNKER_VERSION,
    DEFAULT_MAX_CHARS,
    TextChunk,
    chunk_markdown,
    chunk_plain_text,
)
from faq_core.retrieval.ranking import content_fingerprint, fingerprint
from faq_core.retrieval.retriever import chunk_text
from observability.log import get_logger

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from datetime import datetime

    from faq_core.models import Document
    from faq_core.retrieval.chunk_store import ChunkStore, SourceKey
    from faq_core.retrieval.embedders import Embedder
    from faq_core.retrieval.vector_store import VectorStore

_log = get_logger(__name__)

_EMBED_BATCH: Final = 64


class KnowledgeIndexer:
    """Writes chunks (BM25) and vectors (dense) for one knowledge base.

    Args:
        chunks: Chunk + FTS5 store.
        embedder: Document embedder (its signature names the vector table).
        vectors: Dense store, or ``None`` for BM25-only.
        kb_id: Knowledge base being indexed; sources of other KBs are ignored.
        max_chunk_chars: Upper bound on chunk size (``CHUNK_MAX_CHARS``).
    """

    def __init__(
        self,
        *,
        chunks: ChunkStore,
        embedder: Embedder,
        vectors: VectorStore | None,
        kb_id: str,
        max_chunk_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._vectors = vectors
        self._kb_id = kb_id
        self._max_chars = max_chunk_chars

    @property
    def dense_enabled(self) -> bool:
        return self._vectors is not None

    # ── incremental indexing ─────────────────────────────────────────────────

    async def index_entries(self, entries: Sequence[FAQEntry]) -> int:
        """(Re-)index *entries*; return the number of chunks written."""
        return await self._index(
            [((SourceType.FAQ, str(e.id)), self.chunk_entry(e)) for e in entries]
        )

    async def index_documents(self, documents: Sequence[Document]) -> int:
        """(Re-)index *documents*; return the number of chunks written."""
        return await self._index(
            [((SourceType.DOCUMENT, str(d.id)), self.chunk_document(d)) for d in documents]
        )

    async def _index(self, replacements: list[tuple[SourceKey, list[Chunk]]]) -> int:
        removed = await self._chunks.replace_sources(replacements)
        new_chunks = [chunk for _, chunks in replacements for chunk in chunks]
        if self._vectors is not None:
            await self._vectors.delete(removed)
            await self._embed_and_store(new_chunks)
        return len(new_chunks)

    # ── startup reconciliation ───────────────────────────────────────────────

    async def sync(
        self, entries: Sequence[FAQEntry], documents: Sequence[Document]
    ) -> IndexSyncReport:
        """Make the index match *entries* and *documents* (the database contents)."""
        expected: dict[SourceKey, str] = {}
        builders: dict[SourceKey, FAQEntry | Document] = {}
        for entry in entries:
            if entry.kb_id == self._kb_id:
                key: SourceKey = (SourceType.FAQ, str(entry.id))
                expected[key] = self.entry_source_hash(entry)
                builders[key] = entry
        for document in documents:
            if document.kb_id == self._kb_id:
                key = (SourceType.DOCUMENT, str(document.id))
                expected[key] = self.document_source_hash(document)
                builders[key] = document

        existing = await self._chunks.source_hashes(self._kb_id)
        stale = [key for key, digest in expected.items() if existing.get(key) != digest]
        gone = [key for key in existing if key not in expected]

        removed = await self._chunks.delete_sources(gone)
        replacements = [(key, self._chunk_source(builders[key])) for key in stale]
        removed += await self._chunks.replace_sources(replacements)
        fts_repaired = await self._chunks.repair_fts(self._kb_id)

        embedded = vectors_removed = 0
        if self._vectors is not None:
            chunk_ids = await self._chunks.chunk_ids(self._kb_id)
            have = await self._vectors.ids(self._kb_id)
            orphans = sorted(have - chunk_ids)
            await self._vectors.delete(orphans)
            missing = await self._chunks.get_chunks(chunk_ids - have)
            await self._embed_and_store(missing)
            embedded, vectors_removed = len(missing), len(orphans)

        report = IndexSyncReport(
            kb_id=self._kb_id,
            sources_indexed=len(stale),
            sources_removed=len(gone),
            fts_repaired=fts_repaired,
            vectors_embedded=embedded,
            vectors_removed=vectors_removed,
            chunks=await self._chunks.count(self._kb_id),
            dense_enabled=self._vectors is not None,
        )
        _log.info("retrieval index synced", extra=report.model_dump())
        return report

    # ── chunk construction ───────────────────────────────────────────────────

    def entry_source_hash(self, entry: FAQEntry) -> str:
        return fingerprint(
            CHUNKER_VERSION,
            str(self._max_chars),
            entry.question,
            entry.answer,
            entry.created_at.isoformat(),
        )

    def document_source_hash(self, document: Document) -> str:
        return fingerprint(
            CHUNKER_VERSION,
            str(self._max_chars),
            document.title,
            document.format.value,
            document.source_uri,
            document.updated_at.isoformat(),
            document.body,
        )

    def chunk_entry(self, entry: FAQEntry) -> list[Chunk]:
        """One chunk per FAQ entry (more only for answers over ``max_chunk_chars``)."""
        pieces = chunk_plain_text(entry.answer, max_chars=self._max_chars)
        return _build_chunks(
            pieces,
            kb_id=entry.kb_id,
            title=entry.question,
            source_uri="",
            source_hash=self.entry_source_hash(entry),
            updated_at=entry.created_at,
            faq_id=entry.id,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Heading-aware chunks for markdown, paragraph chunks for plain text."""
        if document.format is DocumentFormat.MARKDOWN:
            pieces = chunk_markdown(document.body, max_chars=self._max_chars)
        else:
            pieces = chunk_plain_text(document.body, max_chars=self._max_chars)
        return _build_chunks(
            pieces,
            kb_id=document.kb_id,
            title=document.title,
            source_uri=document.source_uri,
            source_hash=self.document_source_hash(document),
            updated_at=document.updated_at,
            document_id=document.id,
        )

    def _chunk_source(self, source: FAQEntry | Document) -> list[Chunk]:
        if isinstance(source, FAQEntry):
            return self.chunk_entry(source)
        return self.chunk_document(source)

    async def _embed_and_store(self, chunks: Sequence[Chunk]) -> None:
        if self._vectors is None or not chunks:
            return
        for start in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[start : start + _EMBED_BATCH]
            vectors = await self._embedder.embed_documents([chunk_text(c) for c in batch])
            await self._vectors.upsert(
                self._kb_id,
                [(str(c.id), v) for c, v in zip(batch, vectors, strict=True)],
            )


def _build_chunks(
    pieces: Sequence[TextChunk],
    *,
    kb_id: str,
    title: str,
    source_uri: str,
    source_hash: str,
    updated_at: datetime,
    faq_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> list[Chunk]:
    """Wrap text pieces as chunks, skipping exact repeats within the source."""
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for piece in pieces:
        digest = content_fingerprint(f"{piece.heading}\n{piece.content}")
        if digest in seen:
            continue
        seen.add(digest)
        chunks.append(
            Chunk(
                kb_id=kb_id,
                faq_id=faq_id,
                document_id=document_id,
                ordinal=len(chunks),
                title=title,
                heading=piece.heading,
                content=piece.content,
                source_uri=source_uri,
                content_hash=content_fingerprint(piece.content),
                source_hash=source_hash,
                updated_at=updated_at,
            )
        )
    return chunks
