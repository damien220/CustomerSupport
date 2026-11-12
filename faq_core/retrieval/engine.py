"""The answer engine (``faq_core.ports.AnswerEngine``) on top of hybrid retrieval.

:class:`RetrievalAnswerEngine` retrieves passages with
:class:`~faq_core.retrieval.retriever.HybridRetriever`, asks the LLM gateway for
a grounded answer, and returns the passages as citation objects plus the FAQ
entries behind them as ``sources``. :meth:`RetrievalAnswerEngine.retrieve` is
the passages-only path (``search_kb``; the only search path in
``LLM_MODE=client``). Indexing is delegated to
:class:`~faq_core.retrieval.indexer.KnowledgeIndexer`.

:func:`build_retrieval_stack` wires the retrieval pieces from configuration
(also used by the benchmark); :func:`create_answer_engine` adds the LLM. Both
are called only from :mod:`bootstrap`, after Langfuse is initialised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from faq_core.models import DEFAULT_KB_ID, AnswerResponse, Passage, RetrievalResponse, SourceType
from faq_core.retrieval.chunk_store import ChunkStore
from faq_core.retrieval.embedders import create_embedder
from faq_core.retrieval.indexer import KnowledgeIndexer
from faq_core.retrieval.rerankers import create_reranker
from faq_core.retrieval.retriever import HybridRetriever, RetrievalSettings
from faq_core.retrieval.vector_store import SqliteVecStore
from observability.log import get_logger
from observability.tracing import observe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faq_core.config import AppConfig
    from faq_core.llm.gateway import LLMGateway
    from faq_core.models import Document, FAQEntry, IndexSyncReport
    from faq_core.repository import FAQRepository
    from faq_core.retrieval.embedders import Embedder
    from faq_core.retrieval.rerankers import Reranker
    from faq_core.retrieval.retriever import RetrievedChunk
    from faq_core.retrieval.vector_store import VectorStore

_log = get_logger(__name__)

_SQLITE_VEC: Final = frozenset({"", "sqlite-vec", "sqlite_vec", "sqlitevec"})
_DISABLED: Final = frozenset({"none", "off", "bm25"})
# Phase-1 values: chroma/inmemory were removed with Haystack; qdrant is not built yet.
_LEGACY: Final = frozenset({"chroma", "inmemory", "in-memory", "qdrant"})


@dataclass(frozen=True, slots=True)
class RetrievalStack:
    """The wired retrieval pieces for one knowledge base."""

    retriever: HybridRetriever
    indexer: KnowledgeIndexer
    embedder: Embedder
    vectors: VectorStore | None


class RetrievalAnswerEngine:
    """Retrieve → generate → cite. Satisfies :class:`faq_core.ports.AnswerEngine`."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        indexer: KnowledgeIndexer,
        llm: LLMGateway,
        embedder: Embedder | None = None,
    ) -> None:
        self._retriever = retriever
        self._indexer = indexer
        self._llm = llm
        self._embedder = embedder

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def embedder(self) -> Embedder | None:
        """The index's embedder (reused to cluster knowledge gaps)."""
        return self._embedder

    @property
    def generates_answers(self) -> bool:
        """``False`` with a :class:`~faq_core.llm.gateway.NullGateway` (client mode)."""
        return self._llm.can_generate

    @observe(name="rag.search")
    async def search(self, query: str, top_k: int = 3) -> AnswerResponse:
        hits = await self._retriever.retrieve(query, top_k)
        answer = await self._llm.generate_answer(query, [hit.context() for hit in hits])
        return AnswerResponse(
            query=query,
            answer=answer,
            sources=_faq_sources(hits),
            confidence=_top_confidence(hits),
            citations=[hit.citation(query) for hit in hits],
        )

    @observe(name="rag.generate")
    async def generate(
        self, query: str, retrieval: RetrievalResponse, *, caveat: bool = False
    ) -> str:
        """Generate an answer to *query* (already redacted) from *retrieval*'s passages."""
        contexts = [passage_context(passage) for passage in retrieval.passages]
        return await self._llm.generate_answer(query, contexts, caveat=caveat)

    @observe(name="rag.verify")
    async def verify(self, query: str, retrieval: RetrievalResponse, answer: str) -> bool:
        """Optional LLM self-check of *answer* against *retrieval*'s passages."""
        contexts = [passage_context(passage) for passage in retrieval.passages]
        return await self._llm.verify_answer(query, contexts, answer)

    @observe(name="rag.retrieve")
    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResponse:
        """Retrieve passages and confidence evidence only — no LLM call."""
        hits, evidence = await self._retriever.retrieve_with_evidence(query, top_k)
        citations = [hit.citation(query) for hit in hits]
        passages = [
            Passage(
                rank=rank,
                citation_id=citation.id,
                source_type=citation.source_type,
                source_id=citation.source_id,
                title=citation.title,
                heading=citation.heading,
                text=hit.chunk.content,
                source_uri=citation.source_uri,
                updated_at=citation.updated_at,
                score=citation.score,
            )
            for rank, (hit, citation) in enumerate(zip(hits, citations, strict=True), start=1)
        ]
        return RetrievalResponse(
            query=query,
            passages=passages,
            citations=citations,
            sources=_faq_sources(hits),
            confidence=_top_confidence(hits),
            evidence=evidence,
        )

    @observe(name="rag.add_document")
    async def add_document(self, entry: FAQEntry) -> None:
        """Index (or re-index) one FAQ entry — the ``faq.added`` background job."""
        await self._indexer.index_entries([entry])

    async def seed_from_entries(self, entries: list[FAQEntry]) -> None:
        await self._indexer.index_entries(entries)

    @observe(name="rag.index_document")
    async def index_document(self, document: Document) -> None:
        await self._indexer.index_documents([document])

    async def sync_index(
        self, entries: Sequence[FAQEntry], documents: Sequence[Document]
    ) -> IndexSyncReport:
        return await self._indexer.sync(entries, documents)


def build_retrieval_stack(
    config: AppConfig,
    repo: FAQRepository,
    *,
    kb_id: str = DEFAULT_KB_ID,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RetrievalStack:
    """Wire chunk store, embedder, vector store, reranker, retriever and indexer.

    Raises:
        ValueError: an unknown ``VECTOR_STORE`` / ``EMBED_PROVIDER`` / ``RERANKER``.
    """
    embedder = embedder or create_embedder(config)
    reranker = reranker or create_reranker(config)
    chunks = ChunkStore(repo)
    vectors: VectorStore | None = None
    if _dense_requested(config.VECTOR_STORE):
        if repo.vector_search_available:
            vectors = SqliteVecStore(repo, signature=embedder.name, dimension=embedder.dimension)
        else:
            _log.warning("dense retrieval disabled: sqlite-vec is not loaded; using BM25 only")
    retriever = HybridRetriever(
        chunks=chunks,
        embedder=embedder,
        vectors=vectors,
        reranker=reranker,
        kb_id=kb_id,
        settings=RetrievalSettings(
            rrf_k=config.RETRIEVAL_RRF_K,
            candidates=config.RETRIEVAL_CANDIDATES,
            dedupe_threshold=config.RETRIEVAL_DEDUPE_THRESHOLD,
        ),
    )
    indexer = KnowledgeIndexer(
        chunks=chunks,
        embedder=embedder,
        vectors=vectors,
        kb_id=kb_id,
        max_chunk_chars=config.CHUNK_MAX_CHARS,
    )
    _log.info(
        "retrieval stack ready",
        extra={
            "embedder": embedder.name,
            "vector_store": vectors.name if vectors is not None else "none",
            "reranker": reranker.name,
        },
    )
    return RetrievalStack(retriever=retriever, indexer=indexer, embedder=embedder, vectors=vectors)


def create_answer_engine(
    config: AppConfig, llm: LLMGateway, repo: FAQRepository
) -> RetrievalAnswerEngine:
    """Build the production answer engine over *repo*'s database."""
    stack = build_retrieval_stack(config, repo)
    return RetrievalAnswerEngine(
        retriever=stack.retriever, indexer=stack.indexer, llm=llm, embedder=stack.embedder
    )


def passage_context(passage: Passage) -> str:
    """A passage as given to the LLM (same shape as ``RetrievedChunk.context``)."""
    if passage.source_type is SourceType.FAQ:
        return f"Q: {passage.title}\nA: {passage.text}"
    location = " > ".join(p for p in (passage.title, passage.heading) if p)
    return f"{location}\n{passage.text}" if location else passage.text


def _faq_sources(hits: Sequence[RetrievedChunk]) -> list[FAQEntry]:
    sources: dict[str, FAQEntry] = {}
    for hit in hits:
        if hit.faq is not None:
            sources.setdefault(str(hit.faq.id), hit.faq)
    return list(sources.values())


def _top_confidence(hits: Sequence[RetrievedChunk]) -> float:
    return round(hits[0].confidence, 4) if hits else 0.0


def _dense_requested(setting: str) -> bool:
    backend = setting.strip().lower()
    if backend in _SQLITE_VEC:
        return True
    if backend in _DISABLED:
        return False
    if backend in _LEGACY:
        _log.warning(
            "VECTOR_STORE backend is not available since Phase 2; using sqlite-vec",
            extra={"vector_store": backend},
        )
        return True
    raise ValueError(f"Unknown VECTOR_STORE: {backend!r}")
