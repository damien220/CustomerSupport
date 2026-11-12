"""Knowledge-gap clustering: turn the gap log into a ranked content backlog.

The gap log is the ``query_log`` rows whose outcome was not ``answer`` (low
confidence, escalated) or whose answer got negative feedback — queries stored
PII-redacted. :func:`cluster_gaps` groups them greedily (cheap, no new
dependencies): queries are embedded once with the knowledge base's own
embedder (hashing by default — lexical similarity) and each joins the first
cluster whose representative is at least ``threshold`` cosine-similar, else
starts a new one. Identical normalised queries always share a cluster.

The greedy pass is O(N·K·d) pure-Python work (N queries, K clusters, d
dimensions), so it runs in a worker thread (:func:`asyncio.to_thread`) and the
event loop keeps serving other requests; vectors are normalised once so each
comparison is a single C-level dot product (:func:`math.sumprod`).

The result is ranked by size, then recency: the most-asked unanswered topics
first — candidates for new FAQ entries or documents.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from faq_core.models import KnowledgeGap
from faq_core.retrieval.ranking import normalize_text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from faq_core.models import QueryLogRecord
    from faq_core.retrieval.embedders import Embedder

MAX_EXAMPLES: Final = 5
DEFAULT_CLUSTER_THRESHOLD: Final = 0.35


@dataclass(slots=True)
class _Cluster:
    representative: str
    vector: list[float]
    key: str
    records: list[QueryLogRecord] = field(default_factory=list)


def unit_vector(vector: Sequence[float]) -> list[float]:
    """*vector* scaled to length 1 (all zeros stay zeros, so their cosine is 0)."""
    norm = math.sqrt(math.sumprod(vector, vector))
    return [value / norm for value in vector] if norm else [0.0] * len(vector)


async def cluster_gaps(
    records: Sequence[QueryLogRecord],
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[KnowledgeGap]:
    """Cluster gap-log *records* (any order) into ranked :class:`KnowledgeGap` items.

    Embedding stays a coroutine (a remote embedder is I/O); the CPU-bound
    clustering runs in a worker thread so it never blocks the event loop.
    """
    if not records:
        return []
    ordered = sorted(records, key=lambda record: record.created_at)
    vectors = await embedder.embed_documents([record.query for record in ordered])
    return await asyncio.to_thread(cluster_embedded, ordered, vectors, threshold=threshold)


def cluster_embedded(
    ordered: Sequence[QueryLogRecord],
    vectors: Sequence[Sequence[float]],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[KnowledgeGap]:
    """Greedy clustering of *ordered* records (oldest first) with their *vectors* (sync)."""
    clusters: list[_Cluster] = []
    by_key: dict[str, _Cluster] = {}
    for record, raw_vector in zip(ordered, vectors, strict=True):
        key = normalize_text(record.query)
        cluster = by_key.get(key)
        if cluster is None:
            vector = unit_vector(raw_vector)
            cluster = next(
                (c for c in clusters if math.sumprod(vector, c.vector) >= threshold),
                None,
            )
            if cluster is None:
                cluster = _Cluster(representative=record.query, vector=vector, key=key)
                clusters.append(cluster)
        by_key.setdefault(key, cluster)
        cluster.records.append(record)
    gaps = [_to_gap(cluster) for cluster in clusters]
    gaps.sort(key=lambda gap: (gap.count, gap.last_seen), reverse=True)
    return gaps


def _to_gap(cluster: _Cluster) -> KnowledgeGap:
    records = cluster.records
    examples = list(dict.fromkeys(record.query for record in reversed(records)))[:MAX_EXAMPLES]
    confidences = [record.confidence for record in records if record.confidence is not None]
    first_seen: datetime = min(record.created_at for record in records)
    last_seen: datetime = max(record.created_at for record in records)
    return KnowledgeGap(
        gap_id=hashlib.sha256(cluster.key.encode("utf-8")).hexdigest()[:16],
        representative_query=cluster.representative,
        count=len(records),
        example_queries=examples,
        outcomes=dict(Counter(record.outcome.value for record in records)),
        reasons=dict(Counter(reason for record in records for reason in record.reasons)),
        avg_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        first_seen=first_seen,
        last_seen=last_seen,
    )
