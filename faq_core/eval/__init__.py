"""Offline answer-quality evaluation (RAGAS) — needs requirements-rag.txt and a live LLM.

Modules are imported individually; nothing here is imported by the service or
its entry points, and ``ragas`` itself is imported lazily inside the functions
that need it, so the lite install never loads it.
"""
