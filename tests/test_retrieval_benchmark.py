"""Retrieval benchmark: metric definitions and a lite-mode quality floor on ``data/``.

Runs the real stack (hashing embedder + sqlite-vec + FTS5) — no torch, no network.
The floors sit a little under the measured scores so a regression in chunking,
fusion or BM25 weighting fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faq_core.retrieval.benchmark import (
    DEFAULT_CASES,
    load_cases,
    main,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
)
from tests.conftest import requires_sqlite_vec


def test_recall_at_k() -> None:
    ranked = ["a", "b", "c"]

    assert recall_at_k(ranked, {"b"}, 1) == 0.0
    assert recall_at_k(ranked, {"b"}, 2) == 1.0
    assert recall_at_k(ranked, {"a", "z"}, 3) == 0.5
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_labeled_cases_reference_questions_in_the_seed_file() -> None:
    cases, seed = load_cases(DEFAULT_CASES)
    questions = {item["question"] for item in json.loads(seed.read_text(encoding="utf-8"))}

    assert len(cases) >= 20
    assert all(case.relevant <= questions for case in cases)


async def test_hybrid_retrieval_meets_the_lite_quality_floor() -> None:
    reports = {report.mode: report for report in await run_benchmark(modes=("hybrid", "bm25"))}

    hybrid = reports["hybrid"]
    assert hybrid.cases >= 20
    assert hybrid.recall_at[1] >= 0.8
    assert hybrid.recall_at[3] >= 0.9
    assert hybrid.recall_at[5] >= 0.95
    assert hybrid.mrr >= 0.85
    assert hybrid.mrr >= reports["bm25"].mrr - 0.02  # fusion must not hurt BM25


@requires_sqlite_vec
async def test_dense_mode_is_benchmarked_when_sqlite_vec_loads() -> None:
    [dense] = await run_benchmark(modes=("dense",))

    assert dense.mode == "dense" and dense.recall_at[5] > 0.5


def test_cli_prints_one_json_report_per_mode(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["--modes", "bm25", "--k", "1", "3"]) == 0

    [line] = capsys.readouterr().out.strip().splitlines()
    report = json.loads(line)
    assert report["mode"] == "bm25" and set(report["recall_at"]) == {"1", "3"}
