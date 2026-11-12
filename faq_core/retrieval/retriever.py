"""Hybrid retrieval: BM25 + dense → RRF → near-duplicate collapse → rerank → citations.

Per query, both retrievers fetch a candidate pool (``RETRIEVAL_CANDIDATES``,
at least ``top_k``) concurrently; the ranked id lists are fused with Reciprocal
Rank Fusion; candidates whose text near-duplicates a better-ranked one are
dropped; the optional reranker reorders what is left; the best ``top_k``
become :class:`RetrievedChunk` hits.

``confidence`` of a hit (0..1, surfaced as ``Citation.score`` and as the
answer's ``confidence``) is the reranker's relevance score when a reranker
scored it, otherwise its fused RRF score divided by the best score possible
with the retrievers that ran — ``1.0`` means "ranked first by every
retriever". It is a rank-agreement signal, not a calibrated probability.
Answer-level confidence (Phase 4) is computed from the raw evidence that
:meth:`HybridRetriever.retrieve_with_evidence` returns alongside the hits — BM25
scores, dense similarities, ranks and query-term coverage — by
:mod:`faq_core.confidence`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from faq_core.confidence import HitEvidence, term_coverage
from faq_core.models import Citation, SourceType
from faq_core.retrieval.ranking import (
    DEFAULT_DEDUPE_THRESHOLD,
    DEFAULT_RRF_K,
    dedupe_ranked,
    make_snippet,
    rrf_fuse,
    rrf_max_score,
)
from faq_core.retrieval.rerankers import NoOpReranker, RerankerError
from observability.log import get_logger
from observability.tracing import observe

if TYPE_CHECKING:
    from faq_core.models import Chunk, FAQEntry
    from faq_core.retrieval.chunk_store import ChunkStore, StoredChunk
    from faq_core.retrieval.embedders import Embedder
    from faq_core.retrieval.rerankers import Reranker
    from faq_core.retrieval.vector_store import VectorStore

_log = get_logger(__name__)

type RetrievalMode = Literal["hybrid", "bm25", "dense"]

DEFAULT_CANDIDATES: Final = 30


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Tuning knobs (``RETRIEVAL_*`` in :class:`~faq_core.config.AppConfig`)."""

    rrf_k: int = DEFAULT_RRF_K
    candidates: int = DEFAULT_CANDIDATES
    dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One hit: the chunk, its FAQ entry (if any), ranks and scores."""

    chunk: Chunk
    faq: FAQEntry | None
    fused_score: float
    confidence: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    bm25_score: float | None = None
    dense_similarity: float | None = None

    @property
    def text(self) -> str:
        """Title, heading and content — what dedupe, reranking and embeddings see."""
        return chunk_text(self.chunk)

    def context(self) -> str:
        """The passage as given to the answer-generating LLM."""
        if self.chunk.source_type is SourceType.FAQ:
            return f"Q: {self.chunk.title}\nA: {self.chunk.content}"
        location = " > ".join(p for p in (self.chunk.title, self.chunk.heading) if p)
        return f"{location}\n{self.chunk.content}" if location else self.chunk.content

    def citation(self, query: str) -> Citation:
        source_id = self.chunk.source_id
        if source_id is None:  # chunk rows always carry a source; guard the type
            raise ValueError(f"chunk {self.chunk.id} has no source")
        return Citation(
            id=self.chunk.id,
            source_type=self.chunk.source_type,
            source_id=source_id,
            title=self.chunk.title,
            heading=self.chunk.heading,
            snippet=make_snippet(self.chunk.content, query),
            source_uri=self.chunk.source_uri,
            updated_at=self.chunk.updated_at,
            score=round(self.confidence, 4),
        )


def chunk_text(chunk: Chunk) -> str:
    """The text of *chunk* used for embeddings, dedupe and reranking."""
    return "\n".join(part for part in (chunk.title, chunk.heading, chunk.content) if part)


class HybridRetriever:
    """Runs hybrid search over one knowledge base.

    Args:
        chunks: Chunk + FTS5 store (BM25 side).
        embedder: Query embedder; must match the one used for indexing.
        vectors: Dense store, or ``None`` for BM25-only (sqlite-vec missing or
            ``VECTOR_STORE=none``).
        reranker: Second stage; :class:`NoOpReranker` keeps the fused order.
        kb_id: Knowledge base searched.
        settings: RRF ``k``, candidate pool size, dedupe threshold.
    """

    def __init__(
        self,
        *,
        chunks: ChunkStore,
        embedder: Embedder,
        vectors: VectorStore | None,
        reranker: Reranker | None = None,
        kb_id: str,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self._chunks = chunks
        self._embedder = embedder
        self._vectors = vectors
        self._reranker = reranker or NoOpReranker()
        self._kb_id = kb_id
        self._settings = settings or RetrievalSettings()

    @property
    def dense_enabled(self) -> bool:
        return self._vectors is not None

    @observe(name="retrieval.retrieve")
    async def retrieve(
        self, query: str, top_k: int, *, mode: RetrievalMode = "hybrid"
    ) -> list[RetrievedChunk]:
        """Return up to *top_k* deduplicated hits for *query*, best first.

        ``mode`` restricts retrieval to one side (benchmarks); ``"dense"``
        without a vector store returns nothing.
        """
        hits, _evidence = await self._run(query, top_k, mode)
        return hits

    @observe(name="retrieval.retrieve_with_evidence")
    async def retrieve_with_evidence(
        self, query: str, top_k: int
    ) -> tuple[list[RetrievedChunk], HitEvidence | None]:
        """Hybrid hits plus the raw evidence for the top one (``None`` without hits)."""
        return await self._run(query, top_k, "hybrid")

    async def _run(
        self, query: str, top_k: int, mode: RetrievalMode
    ) -> tuple[list[RetrievedChunk], HitEvidence | None]:
        if top_k < 1 or not query.strip():
            return [], None
        pool = max(self._settings.candidates, top_k)
        use_bm25 = mode in ("hybrid", "bm25")
        use_dense = mode in ("hybrid", "dense") and self._vectors is not None

        bm25_hits, dense_hits = await asyncio.gather(
            self._chunks.search_bm25(self._kb_id, query, pool) if use_bm25 else _no_hits(),
            self._search_dense(query, pool) if use_dense else _no_hits(),
        )
        bm25_ids = [chunk_id for chunk_id, _ in bm25_hits]
        dense_ids = [chunk_id for chunk_id, _ in dense_hits]
        rankings = [ids for ids, active in ((bm25_ids, use_bm25), (dense_ids, use_dense)) if active]
        fused = rrf_fuse(rankings, k=self._settings.rrf_k)[:pool]
        if not fused:
            return [], None

        stored = await self._chunks.fetch([chunk_id for chunk_id, _ in fused])
        best_possible = rrf_max_score(len(rankings), k=self._settings.rrf_k)
        scores = _RawScores(
            bm25_rank={chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)},
            dense_rank={chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)},
            bm25=dict(bm25_hits),
            dense=dict(dense_hits),
        )
        candidates = [
            _hit(stored[chunk_id], score, best_possible, scores)
            for chunk_id, score in fused
            if chunk_id in stored  # an orphan vector whose chunk is gone
        ]
        candidates = dedupe_ranked(
            candidates, lambda hit: hit.text, threshold=self._settings.dedupe_threshold
        )
        hits = await self._rerank(query, candidates, top_k)
        if not hits:
            return [], None
        return hits, _evidence(query, hits[0], candidates, retrievers=len(rankings))

    async def _search_dense(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self._vectors is None:
            return []
        vector = await self._embedder.embed_query(query)
        return await self._vectors.search(self._kb_id, vector, limit)

    async def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if isinstance(self._reranker, NoOpReranker) or len(candidates) <= 1:
            return candidates[:top_k]
        try:
            results = await self._reranker.rerank(query, [c.text for c in candidates], top_k)
        except RerankerError:
            _log.warning(
                "reranker failed; keeping fused order",
                extra={"reranker": self._reranker.name},
                exc_info=True,
            )
            return candidates[:top_k]
        reranked: list[RetrievedChunk] = []
        for result in results[:top_k]:
            hit = candidates[result.index]
            if result.score is not None:
                hit = replace(hit, rerank_score=result.score, confidence=result.score)
            reranked.append(hit)
        return reranked


async def _no_hits() -> list[tuple[str, float]]:
    return []


@dataclass(frozen=True, slots=True)
class _RawScores:
    bm25_rank: dict[str, int]
    dense_rank: dict[str, int]
    bm25: dict[str, float]
    dense: dict[str, float]


def _hit(
    stored: StoredChunk, fused_score: float, best_possible: float, scores: _RawScores
) -> RetrievedChunk:
    chunk_id = str(stored.chunk.id)
    return RetrievedChunk(
        chunk=stored.chunk,
        faq=stored.faq,
        fused_score=fused_score,
        confidence=min(fused_score / best_possible, 1.0) if best_possible else 0.0,
        bm25_rank=scores.bm25_rank.get(chunk_id),
        dense_rank=scores.dense_rank.get(chunk_id),
        bm25_score=scores.bm25.get(chunk_id),
        dense_similarity=scores.dense.get(chunk_id),
    )


def _evidence(
    query: str, top: RetrievedChunk, candidates: list[RetrievedChunk], *, retrievers: int
) -> HitEvidence:
    """Evidence for *top*; runner-ups come from the deduplicated candidate pool."""
    others = [hit for hit in candidates if hit.chunk.id != top.chunk.id]
    dense_others = [hit.dense_similarity for hit in others if hit.dense_similarity is not None]
    bm25_others = [hit.bm25_score for hit in others if hit.bm25_score is not None]
    return HitEvidence(
        term_coverage=term_coverage(query, top.text),
        retrievers=retrievers,
        bm25_rank=top.bm25_rank,
        dense_rank=top.dense_rank,
        similarity=top.dense_similarity,
        runner_up_similarity=max(dense_others, default=None),
        bm25_score=top.bm25_score,
        runner_up_bm25=max(bm25_others, default=None),
        rerank_score=top.rerank_score,
    )
