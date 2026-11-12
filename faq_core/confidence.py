"""Answer confidence from retrieval evidence (plan §4, Phase 4).

Replaces the Phase 2/3 "score of the best passage", which was an RRF rank
signal (a BM25-only top hit always normalised to ``1.0``). The score here is a
weighted mean of the evidence components that are available for the top
passage — never a lone normalised rank score:

=================  ======  =====================================================
component          weight  meaning (each in ``0..1``)
=================  ======  =====================================================
``term_coverage``  0.40    share of the query's content words (stopwords
                           dropped, 5-letter truncation stemming) found in the
                           top passage — embedder-independent lexical evidence
``similarity``     0.30    dense cosine similarity of the top passage, mapped
                           linearly from ``[floor, ceiling]`` onto ``[0, 1]``
``agreement``      0.20    BM25 and dense both rank it first (1.0), both in
                           their top 3 (0.5), both found it (0.25), one only (0)
``margin``         0.10    relative lead of the top passage over the runner-up
                           (dense similarity, else BM25 score)
``rerank``         0.30    the reranker's relevance score, when one ran
=================  ======  =====================================================

Components that cannot be computed (no dense side, one retriever, no
reranker) are left out and the remaining weights are renormalised.

Thresholds (``CONFIDENCE_ANSWER_THRESHOLD`` / ``CONFIDENCE_CAVEAT_THRESHOLD``)
map the score to a level and an outcome: ``high`` → ``answer``, ``medium`` →
``answer_with_caveat``, ``low`` → ``escalate``. The defaults and the
similarity floor/ceiling were tuned on ``data/retrieval_benchmark.json`` with
the default hashing embedder (a test keeps in-KB cases out of ``escalate`` and
the benchmark's ``out_of_kb`` queries in it). The score is a heuristic, **not
a calibrated probability** (``calibrated`` is always ``false``); a semantic
embedder changes the similarity scale, so re-check the floor/ceiling when
switching ``EMBED_PROVIDER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from faq_core.retrieval.ranking import STOPWORDS, normalize_text

STEM_CHARS: Final = 5
"""Truncation-stemming length: ``shipping``/``shipped`` → ``shipp``."""

WEIGHTS: Final[dict[str, float]] = {
    "term_coverage": 0.40,
    "similarity": 0.30,
    "agreement": 0.20,
    "margin": 0.10,
    "rerank": 0.30,
}

_AGREEMENT_TOP: Final = 3
_WEAK: Final = 0.5
_SMALL_MARGIN: Final = 0.15


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Outcome(StrEnum):
    """What the caller should do with a result (stable values)."""

    ANSWER = "answer"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    ESCALATE = "escalate"


LEVEL_OUTCOME: Final[dict[ConfidenceLevel, Outcome]] = {
    ConfidenceLevel.HIGH: Outcome.ANSWER,
    ConfidenceLevel.MEDIUM: Outcome.ANSWER_WITH_CAVEAT,
    ConfidenceLevel.LOW: Outcome.ESCALATE,
}


@dataclass(frozen=True, slots=True)
class HitEvidence:
    """Raw retrieval evidence for the top passage of one query.

    ``bm25_rank`` / ``dense_rank`` are 1-based ranks in each retriever's
    candidate list (``None``: not found by it). ``similarity`` is the top
    passage's dense cosine similarity, ``runner_up_similarity`` the best one
    among the other candidates; ``bm25_score`` / ``runner_up_bm25`` likewise
    for BM25 (higher is better). ``retrievers`` counts the retrievers that ran.
    """

    term_coverage: float
    retrievers: int
    bm25_rank: int | None = None
    dense_rank: int | None = None
    similarity: float | None = None
    runner_up_similarity: float | None = None
    bm25_score: float | None = None
    runner_up_bm25: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Score cut-offs and the similarity scale (``CONFIDENCE_*`` settings)."""

    answer: float = 0.55
    caveat: float = 0.30
    similarity_floor: float = 0.05
    similarity_ceiling: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.caveat <= self.answer <= 1.0:
            raise ValueError("confidence thresholds need 0 <= caveat <= answer <= 1")
        if self.similarity_ceiling <= self.similarity_floor:
            raise ValueError("CONFIDENCE_SIMILARITY_CEILING must exceed the floor")


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """Score (0..1), level, the components used and why it is not high."""

    score: float
    level: ConfidenceLevel
    components: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def outcome(self) -> Outcome:
        return LEVEL_OUTCOME[self.level]


def content_terms(text: str) -> set[str]:
    """Stemmed content words of *text* (stopwords and 1-letter tokens dropped)."""
    return {
        word[:STEM_CHARS]
        for word in normalize_text(text).split()
        if word not in STOPWORDS and len(word) > 1
    }


def term_coverage(query: str, passage: str) -> float:
    """Share of *query*'s content terms that occur in *passage* (0.0 when it has none)."""
    wanted = content_terms(query)
    if not wanted:
        return 0.0
    return len(wanted & content_terms(passage)) / len(wanted)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _margin(best: float | None, runner_up: float | None) -> float | None:
    if best is None or best <= 0.0:
        return None if best is None else 0.0
    return _clip((best - (runner_up or 0.0)) / best)


def _agreement(evidence: HitEvidence) -> float | None:
    if evidence.retrievers < 2:  # noqa: PLR2004 — needs BM25 and dense
        return None
    bm25, dense = evidence.bm25_rank, evidence.dense_rank
    if bm25 is None or dense is None:
        return 0.0
    if bm25 == 1 and dense == 1:
        return 1.0
    if bm25 <= _AGREEMENT_TOP and dense <= _AGREEMENT_TOP:
        return 0.5
    return 0.25


class ConfidenceScorer:
    """Turns :class:`HitEvidence` into a :class:`ConfidenceAssessment`."""

    def __init__(self, thresholds: ConfidenceThresholds | None = None) -> None:
        self._thresholds = thresholds or ConfidenceThresholds()

    @property
    def thresholds(self) -> ConfidenceThresholds:
        return self._thresholds

    def assess(self, evidence: HitEvidence | None) -> ConfidenceAssessment:
        """Score the evidence of the top passage (``None``: nothing was retrieved)."""
        if evidence is None:
            return ConfidenceAssessment(0.0, ConfidenceLevel.LOW, {}, ("no_passages",))
        components = self._components(evidence)
        total_weight = sum(WEIGHTS[name] for name in components)
        score = (
            sum(WEIGHTS[name] * value for name, value in components.items()) / total_weight
            if total_weight
            else 0.0
        )
        score = round(score, 4)
        if score >= self._thresholds.answer:
            level = ConfidenceLevel.HIGH
        elif score >= self._thresholds.caveat:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        reasons = () if level is ConfidenceLevel.HIGH else _weak_signals(evidence, components)
        return ConfidenceAssessment(score, level, components, reasons)

    def _components(self, evidence: HitEvidence) -> dict[str, float]:
        floor, ceiling = self._thresholds.similarity_floor, self._thresholds.similarity_ceiling
        components = {"term_coverage": _clip(evidence.term_coverage)}
        if evidence.similarity is not None:
            components["similarity"] = _clip((evidence.similarity - floor) / (ceiling - floor))
        agreement = _agreement(evidence)
        if agreement is not None:
            components["agreement"] = agreement
        if evidence.similarity is not None:
            margin = _margin(evidence.similarity, evidence.runner_up_similarity)
        else:
            margin = _margin(evidence.bm25_score, evidence.runner_up_bm25)
        if margin is not None:
            components["margin"] = margin
        if evidence.rerank_score is not None:
            components["rerank"] = _clip(evidence.rerank_score)
        return {name: round(value, 4) for name, value in components.items()}


def _weak_signals(evidence: HitEvidence, components: dict[str, float]) -> tuple[str, ...]:
    reasons: list[str] = []
    if evidence.bm25_rank is None:
        reasons.append("no_keyword_match")
    if components.get("term_coverage", 0.0) < _WEAK:
        reasons.append("low_term_coverage")
    if components.get("similarity", 1.0) < _WEAK:
        reasons.append("low_similarity")
    if components.get("agreement", 1.0) < _WEAK:
        reasons.append("retrievers_disagree")
    if components.get("margin", 1.0) < _SMALL_MARGIN:
        reasons.append("small_margin")
    if components.get("rerank", 1.0) < _WEAK:
        reasons.append("low_rerank_score")
    return tuple(reasons)
