"""In-house hybrid retrieval: chunker → BM25 (SQLite FTS5) + dense (sqlite-vec) → RRF
→ near-duplicate collapse → optional reranker → citation objects.

Replaces the Phase-1 Haystack/Chroma pipeline (plan decision #1). Everything
lives in the support SQLite file, so the index survives restarts; at startup
:meth:`~faq_core.retrieval.indexer.KnowledgeIndexer.sync` reconciles it with
the FAQ entries and documents in the database.

Modules are imported individually (this package imports nothing), so modules
carrying ``@observe`` wrappers are first imported after Langfuse is initialised
by :func:`bootstrap.build_runtime`.
"""
