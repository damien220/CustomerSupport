"""Retrieval benchmark: recall@k and MRR over a labeled query set.

``python -m faq_core.retrieval.benchmark`` seeds a throw-away SQLite database
with ``data/sample_faq.json``, indexes it with the real retrieval stack and
scores ``data/retrieval_benchmark.json`` in each retrieval mode (``hybrid``,
``bm25``, ``dense``), printing one JSON report per mode. By default it uses the
lite defaults (hashing embedder, sqlite-vec, no reranker) and ignores ``.env``;
``--env`` benchmarks the configured embedder/reranker instead (only ``DB_PATH``
is overridden).

A hit is relevant when its citation title (the FAQ question or document title)
is in the case's ``relevant`` list; repeated titles count once, at their best
rank.

* ``recall@k`` — share of a case's relevant titles found in the top *k*,
  averaged over cases.
* ``MRR`` — mean of ``1 / rank`` of the first relevant hit (0 when none is
  retrieved within the largest *k*).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from faq_core.config import AppConfig, get_config
from faq_core.models import FAQDraft, FAQEntry
from faq_core.repository import FAQRepository
from faq_core.retrieval.engine import build_retrieval_stack

if TYPE_CHECKING:
    from collections.abc import Sequence, Set

    from faq_core.retrieval.retriever import HybridRetriever, RetrievalMode

DATA_DIR: Final = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CASES: Final = DATA_DIR / "retrieval_benchmark.json"
DEFAULT_KS: Final = (1, 3, 5)
ALL_MODES: Final[tuple[RetrievalMode, ...]] = ("hybrid", "bm25", "dense")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    query: str
    relevant: frozenset[str]


@dataclass(slots=True)
class ModeReport:
    """Scores of one retrieval mode."""

    mode: str
    cases: int
    recall_at: dict[int, float]
    mrr: float
    misses: list[str] = field(default_factory=list)
    """Queries with no relevant hit within the largest k."""


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Share of *relevant* items among the first *k* of *ranked* (0 if none relevant)."""
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    """``1 / rank`` of the first relevant item of *ranked* (1-based), 0 when absent."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def load_cases(path: Path = DEFAULT_CASES) -> tuple[list[BenchmarkCase], Path]:
    """Read the labeled cases and the seed file they refer to (relative to *path*)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        BenchmarkCase(query=item["query"], relevant=frozenset(item["relevant"]))
        for item in raw["cases"]
    ]
    return cases, path.parent / raw["seed"]


async def evaluate(
    retriever: HybridRetriever,
    cases: Sequence[BenchmarkCase],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    mode: RetrievalMode = "hybrid",
) -> ModeReport:
    """Score *retriever* on *cases* in one retrieval *mode*."""
    if not cases or not ks:
        raise ValueError("need at least one case and one k")
    depth = max(ks)
    recall_sums = dict.fromkeys(ks, 0.0)
    rr_sum = 0.0
    misses: list[str] = []
    for case in cases:
        hits = await retriever.retrieve(case.query, depth, mode=mode)
        ranked = list(dict.fromkeys(hit.chunk.title for hit in hits))
        for k in ks:
            recall_sums[k] += recall_at_k(ranked, case.relevant, k)
        rr = reciprocal_rank(ranked, case.relevant)
        rr_sum += rr
        if rr == 0.0:
            misses.append(case.query)
    count = len(cases)
    return ModeReport(
        mode=mode,
        cases=count,
        recall_at={k: round(total / count, 4) for k, total in recall_sums.items()},
        mrr=round(rr_sum / count, 4),
        misses=misses,
    )


async def run_benchmark(
    cases_path: Path = DEFAULT_CASES,
    *,
    config: AppConfig | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    modes: Sequence[RetrievalMode] = ALL_MODES,
) -> list[ModeReport]:
    """Index the seed FAQ into a temporary DB and score every requested mode.

    ``dense`` is skipped (absent from the result) when sqlite-vec is unavailable.
    """
    cases, seed_path = load_cases(cases_path)
    raw_seed = json.loads(seed_path.read_text(encoding="utf-8"))
    entries = [FAQEntry(**FAQDraft.model_validate(item).model_dump()) for item in raw_seed]
    with tempfile.TemporaryDirectory(prefix="retrieval-bench-") as tmp:
        base = config or AppConfig(_env_file=None)  # type: ignore[call-arg]
        bench_config = base.model_copy(update={"DB_PATH": str(Path(tmp) / "bench.db")})
        repo = await asyncio.to_thread(FAQRepository, bench_config)
        try:
            await repo.create_many(entries)
            stack = build_retrieval_stack(bench_config, repo)
            await stack.indexer.sync(entries, [])
            return [
                await evaluate(stack.retriever, cases, ks=ks, mode=mode)
                for mode in modes
                if mode != "dense" or stack.retriever.dense_enabled
            ]
        finally:
            await repo.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval benchmark (recall@k, MRR)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    parser.add_argument(
        "--env", action="store_true", help="use the .env configuration (embedder, reranker)"
    )
    args = parser.parse_args(argv)
    reports = asyncio.run(
        run_benchmark(
            args.cases,
            config=get_config() if args.env else None,
            ks=args.k,
            modes=args.modes,
        )
    )
    for report in reports:
        print(json.dumps(asdict(report)))  # noqa: T201 — CLI output
    return 0


if __name__ == "__main__":
    sys.exit(main())
