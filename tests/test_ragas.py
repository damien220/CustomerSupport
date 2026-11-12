"""RAGAS answer-quality gate — end to end through the real ``SupportService``.

Thresholds (mean over the samples below; all must hold):
    * ``faithfulness``      >= 0.85  — answers grounded in the retrieved passages
    * ``answer_relevancy``  >= 0.80  — answers address the question
    * ``context_precision`` >= 0.75  — retrieved passages are relevant to the reference

Pinned to ``ragas==0.4.3`` (``requirements-rag.txt``) and its collections metric
API; the scoring code lives in ``faq_core/eval/ragas_eval.py``. The service is the
production one (``bootstrap.build_runtime`` + ``serve`` over a temporary SQLite DB
seeded from ``data/sample_faq.json``); answers come from ``SupportService.answer``
and the judge is the *configured* LLM (``LLM_PROVIDER`` preset and key from
``.env``), with the retrieval index's embedder for answer relevancy.

Skips cleanly (``rag`` marker, ``importorskip``) under ``pytest -m "not rag"``, when
ragas is not installed, and — with the reason — when the configuration cannot
judge: ``LLM_MODE=client``, no API key / model, or the lexical hashing embedder
(set ``EMBED_PROVIDER=local`` with requirements-rag.txt installed).

``RAGAS_REQUIRE_LIVE=1`` (set by the CI ``ragas-gate`` job) turns every one of
those skips into a failure, so a gate that cannot score anything is red, not
green. Local runs leave it unset and keep skipping.

UNVERIFIED live: the ragas wiring is exercised against a mock transport in
``tests/test_ragas_eval.py``, but this gate has never run against a real LLM, so
the thresholds have never been checked against real scores. Each run makes real,
billed LLM calls (3 samples x about 8 judge requests, plus 3 answers).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REQUIRE_LIVE = os.environ.get("RAGAS_REQUIRE_LIVE", "").strip().lower() in {"1", "true", "yes"}


def _not_live(reason: str) -> None:
    """Skip, or fail when the live gate is required (``RAGAS_REQUIRE_LIVE=1``)."""
    if REQUIRE_LIVE:
        pytest.fail(f"RAGAS_REQUIRE_LIVE=1 but {reason}", pytrace=False)
    pytest.skip(reason)


try:
    import ragas  # noqa: F401 — availability check only
except ImportError:
    if REQUIRE_LIVE:
        raise  # collection error: the required gate cannot run
    pytest.skip("ragas is not installed (requirements-rag.txt)", allow_module_level=True)

from bootstrap import build_runtime, serve  # noqa: E402 — must follow the ragas availability guard
from faq_core.config import get_config  # noqa: E402
from faq_core.eval.ragas_eval import (  # noqa: E402
    EvalSample,
    JudgeUnavailableError,
    build_judge,
    collect_records,
    langfuse_scores,
    resolve_judge_settings,
    score_records,
    threshold_failures,
)
from faq_core.retrieval.engine import RetrievalAnswerEngine  # noqa: E402
from observability.tracing import push_evaluation_scores  # noqa: E402

pytestmark = pytest.mark.rag

# Curated question -> reference pairs; the references mirror data/sample_faq.json.
_SAMPLES: tuple[EvalSample, ...] = (
    EvalSample(
        question="How do I reset my password?",
        reference="Go to the login page and click 'Forgot Password' to receive a reset link.",
    ),
    EvalSample(
        question="How long does shipping take?",
        reference="Standard shipping takes 5-7 business days.",
    ),
    EvalSample(
        question="What payment methods do you accept?",
        reference="We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    ),
)


async def test_rag_quality_meets_thresholds(tmp_path: Path) -> None:
    config = get_config().model_copy(
        update={"DB_PATH": str(tmp_path / "ragas.db"), "SCHEDULER_DB_URL": ""}
    )
    try:
        resolve_judge_settings(config)  # cheap check before anything is built
    except JudgeUnavailableError as err:
        _not_live(f"no live judge configured: {err}")

    runtime = build_runtime(config)
    skip_reason = ""
    async with serve(runtime) as service:  # closes the LLM client and the DB on exit
        engine = runtime.engine
        embedder = engine.embedder if isinstance(engine, RetrievalAnswerEngine) else None
        # Skip only after serve() exits: raising inside its task group hides the reason.
        if embedder is None:
            skip_reason = "the answer engine exposes no index embedder"
        else:
            try:
                judge = build_judge(config, embedder)
            except JudgeUnavailableError as err:
                skip_reason = str(err)
            else:
                records = await collect_records(service, _SAMPLES)
    if skip_reason:
        _not_live(f"no live judge configured: {skip_reason}")
    scores = await score_records(records, judge)

    push_evaluation_scores(langfuse_scores(scores))

    outcomes = {record.user_input: record.outcome for record in records}
    failures = threshold_failures(scores)
    assert not failures, f"RAGAS gate failed: {failures}; scores={scores}; outcomes={outcomes}"
