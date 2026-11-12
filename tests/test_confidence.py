"""Confidence scoring: formula behaviour and a sanity check on the labeled benchmark.

The benchmark check runs the real lite stack (hashing embedder + sqlite-vec +
FTS5) over ``data/retrieval_benchmark.json``: in-KB cases whose relevant entry is
retrieved first must not be escalated, and the curated ``out_of_kb`` queries
must all be escalated.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from faq_core.confidence import (
    ConfidenceLevel,
    ConfidenceScorer,
    ConfidenceThresholds,
    HitEvidence,
    Outcome,
    term_coverage,
)
from faq_core.config import AppConfig
from faq_core.models import FAQDraft, FAQEntry
from faq_core.repository import FAQRepository
from faq_core.retrieval.benchmark import DEFAULT_CASES, load_cases
from faq_core.retrieval.engine import build_retrieval_stack
from tests.conftest import MEDIUM_EVIDENCE, STRONG_EVIDENCE, WEAK_EVIDENCE, requires_sqlite_vec


def test_term_coverage_uses_stems_and_ignores_stopwords() -> None:
    assert term_coverage("tracking my shipping", "Track the shipped email") == 1.0
    assert term_coverage("how do I", "anything") == 0.0
    assert term_coverage("gift cards", "We accept credit cards") == 0.5


@pytest.mark.parametrize(
    ("evidence", "level", "outcome"),
    [
        (STRONG_EVIDENCE, ConfidenceLevel.HIGH, Outcome.ANSWER),
        (MEDIUM_EVIDENCE, ConfidenceLevel.MEDIUM, Outcome.ANSWER_WITH_CAVEAT),
        (WEAK_EVIDENCE, ConfidenceLevel.LOW, Outcome.ESCALATE),
    ],
)
def test_levels_map_to_outcomes(
    evidence: HitEvidence, level: ConfidenceLevel, outcome: Outcome
) -> None:
    assessment = ConfidenceScorer().assess(evidence)

    assert (assessment.level, assessment.outcome) == (level, outcome)
    assert 0.0 <= assessment.score <= 1.0


def test_nothing_retrieved_is_low_with_no_passages_reason() -> None:
    assessment = ConfidenceScorer().assess(None)

    assert (assessment.score, assessment.level, assessment.reasons) == (
        0.0,
        ConfidenceLevel.LOW,
        ("no_passages",),
    )


def test_a_lone_bm25_hit_is_not_trusted_from_its_rank_alone() -> None:
    # BM25 only (no dense side): rank 1 but the query's words are mostly absent.
    evidence = HitEvidence(
        term_coverage=0.2, retrievers=1, bm25_rank=1, bm25_score=1.0, runner_up_bm25=0.95
    )

    assessment = ConfidenceScorer().assess(evidence)

    assert set(assessment.components) == {"term_coverage", "margin"}
    assert assessment.level is ConfidenceLevel.LOW
    assert "low_term_coverage" in assessment.reasons


def test_reranker_score_is_used_when_present() -> None:
    weak_rerank = HitEvidence(
        term_coverage=0.6, retrievers=1, bm25_rank=1, bm25_score=3.0, rerank_score=0.05
    )

    assessment = ConfidenceScorer().assess(weak_rerank)

    assert assessment.components["rerank"] == 0.05
    assert "low_rerank_score" in assessment.reasons


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError, match="caveat <= answer"):
        ConfidenceThresholds(answer=0.3, caveat=0.5)
    with pytest.raises(ValueError, match="CEILING"):
        ConfidenceThresholds(similarity_floor=0.5, similarity_ceiling=0.4)


# ── benchmark sanity ─────────────────────────────────────────────────────────


@requires_sqlite_vec
async def test_benchmark_in_kb_queries_are_not_escalated_and_out_of_kb_ones_are(
    tmp_path: Path,
) -> None:
    cases, seed_path = load_cases(DEFAULT_CASES)
    out_of_kb: list[str] = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))["out_of_kb"]
    raw_seed = json.loads(seed_path.read_text(encoding="utf-8"))
    entries = [FAQEntry(**FAQDraft.model_validate(item).model_dump()) for item in raw_seed]
    config = AppConfig(_env_file=None, DB_PATH=str(tmp_path / "bench.db"))  # type: ignore[call-arg]
    repo = await asyncio.to_thread(FAQRepository, config)
    scorer = ConfidenceScorer()
    try:
        await repo.create_many(entries)
        stack = build_retrieval_stack(config, repo)
        await stack.indexer.sync(entries, [])

        async def outcome_of(query: str) -> tuple[Outcome, str]:
            hits, evidence = await stack.retriever.retrieve_with_evidence(query, 3)
            return scorer.assess(evidence).outcome, hits[0].chunk.title if hits else ""

        in_kb = [(case, *await outcome_of(case.query)) for case in cases]
        out = [(query, (await outcome_of(query))[0]) for query in out_of_kb]
    finally:
        await repo.close()

    found_first = [(c, o) for c, o, top in in_kb if top in c.relevant]
    wrongly_escalated = [c.query for c, o in found_first if o is Outcome.ESCALATE]
    all_escalated = [c.query for c, o, _ in in_kb if o is Outcome.ESCALATE]
    assert len(found_first) >= 20
    assert wrongly_escalated == []
    assert len(all_escalated) <= 2  # only where retrieval itself missed the answer
    assert len(out) >= 10
    assert [q for q, o in out if o is not Outcome.ESCALATE] == []
