"""Tests for ``faq_core.eval.ragas_eval`` — no network, no live LLM.

The helper tests run in the lite install. The ``rag``-marked wiring tests need
``ragas`` (requirements-rag.txt) and drive the real ragas -> instructor ->
provider-SDK chain against a mock HTTP transport: they check the request shape
(no sampling parameters / forced tool use for Claude) and that canned judge
replies score 1.0 — not the quality of a real judge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from bootstrap import build_runtime, serve
from faq_core.confidence import Outcome
from faq_core.config import AppConfig
from faq_core.eval.ragas_eval import (
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    FAITHFULNESS,
    RAGAS_THRESHOLDS,
    EvalRecord,
    EvalSample,
    JudgeUnavailableError,
    collect_records,
    langfuse_scores,
    metric_value,
    require_semantic_embedder,
    resolve_judge_settings,
    threshold_failures,
)
from faq_core.llm.presets import WireProtocol
from faq_core.models import LLMMode
from faq_core.retrieval.embedders import HashingEmbedder
from tests.conftest import FakeLLMClient

_QUESTION = "How do I reset my password?"
_CONTEXT = "Go to the login page and click 'Forgot Password' to get a reset link."


def _config(tmp_path: Path, **overrides: object) -> AppConfig:
    values: dict[str, object] = {
        "DB_PATH": str(tmp_path / "eval.db"),
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "LLM_API_KEY": "test-llm-key",
    }
    values.update(overrides)
    return AppConfig(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


# ── fakes ────────────────────────────────────────────────────────────────────


@dataclass
class _Passage:
    text: str


@dataclass
class _Answered:
    answer: str
    outcome: Outcome = Outcome.ANSWER
    generated: bool = True


@dataclass
class _Retrieved:
    passages: list[_Passage]


@dataclass
class _FakeService:
    answers: dict[str, str]
    contexts: dict[str, list[str]]
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    async def answer(self, query: str, top_k: int = 3, *, session_id: str | None = None) -> Any:  # noqa: ANN401
        self.calls.append(("answer", query, top_k))
        return _Answered(answer=self.answers[query])

    async def search_kb(self, query: str, top_k: int = 3, *, session_id: str | None = None) -> Any:  # noqa: ANN401
        self.calls.append(("search_kb", query, top_k))
        return _Retrieved(passages=[_Passage(text) for text in self.contexts[query]])


class _DigestEmbedder:
    """Deterministic non-hashing embedder: equal texts get equal unit vectors."""

    @property
    def name(self) -> str:
        return "fake-digest"

    @property
    def dimension(self) -> int:
        return 8

    async def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.strip().lower().encode()).digest()[: self.dimension]
        vector = [float(byte) + 1.0 for byte in digest]
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


# ── lite helper tests ───────────────────────────────────────────────────────


async def test_collect_records_pairs_answer_with_full_passages() -> None:
    service = _FakeService(
        answers={_QUESTION: "Click 'Forgot Password'."}, contexts={_QUESTION: [_CONTEXT]}
    )

    records = await collect_records(
        service, [EvalSample(_QUESTION, "Use Forgot Password.")], top_k=2
    )

    assert records == [
        EvalRecord(
            user_input=_QUESTION,
            response="Click 'Forgot Password'.",
            retrieved_contexts=(_CONTEXT,),
            reference="Use Forgot Password.",
            outcome="answer",
            generated=True,
        )
    ]
    assert service.calls == [("answer", _QUESTION, 2), ("search_kb", _QUESTION, 2)]


async def test_collect_records_runs_the_real_service(app_config: AppConfig) -> None:
    """The gate's data path: production runtime, seeded KB, fake LLM only."""
    runtime = build_runtime(app_config, llm=FakeLLMClient(answer="Click 'Forgot Password'."))
    async with serve(runtime) as service:
        [record] = await collect_records(service, [EvalSample(_QUESTION, "Use Forgot Password.")])

    assert record.response == "Click 'Forgot Password'."
    assert record.generated and record.outcome == "answer"
    assert any("Forgot Password" in text for text in record.retrieved_contexts)


def test_judge_settings_follow_the_configured_anthropic_preset(tmp_path: Path) -> None:
    settings = resolve_judge_settings(_config(tmp_path, LLM_PROVIDER="anthropic"))

    assert settings.protocol is WireProtocol.ANTHROPIC
    assert settings.model  # preset default model
    assert settings.base_url.startswith("https://")


def test_judge_settings_follow_a_local_openai_compatible_runtime(tmp_path: Path) -> None:
    settings = resolve_judge_settings(
        _config(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="llama3.2", LLM_API_KEY="")
    )

    assert settings.protocol is WireProtocol.OPENAI
    assert (settings.model, settings.base_url) == ("llama3.2", "http://localhost:11434/v1")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"LLM_MODE": LLMMode.CLIENT}, "no server-side LLM"),
        ({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": ""}, "ANTHROPIC_API_KEY"),
        ({"LLM_PROVIDER": "openai", "LLM_API_KEY": ""}, "LLM_API_KEY"),
        ({"LLM_PROVIDER": "ollama", "LLM_MODEL": ""}, "LLM_MODEL"),
        (
            {"LLM_PROVIDER": "openai", "LLM_BASE_URL": "http://llm.example.com/v1"},
            "must use https",
        ),
        (  # a "local" preset pointed at a remote host: decided by the host, not the preset
            {
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "llama3.2",
                "OLLAMA_BASE_URL": "http://gpu-box.corp.example:11434/v1",
            },
            "must use https",
        ),
        (
            {"LLM_PROVIDER": "vllm", "LLM_MODEL": "m", "LLM_BASE_URL": "http://93.184.216.34/v1"},
            "must use https",
        ),
    ],
)
def test_judge_settings_refuse_unusable_configs(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(JudgeUnavailableError, match=reason) as caught:
        resolve_judge_settings(_config(tmp_path, **overrides))

    assert "test-" not in str(caught.value)  # never echoes a key


@pytest.mark.parametrize(
    ("base_url", "allowed"),
    [
        ("http://127.0.0.1:11434/v1", ""),
        ("http://192.168.1.20:11434/v1", ""),  # private IP
        ("http://gpu-box.corp.example:11434/v1", "gpu-box.corp.example"),  # allow-listed
        ("https://llm.example.com/v1", ""),
    ],
)
def test_judge_settings_allow_plain_http_only_to_local_hosts(
    tmp_path: Path, base_url: str, allowed: str
) -> None:
    settings = resolve_judge_settings(
        _config(
            tmp_path,
            LLM_PROVIDER="ollama",
            LLM_MODEL="llama3.2",
            LLM_BASE_URL=base_url,
            OFFLINE_ALLOWED_HOSTS=allowed,
        )
    )

    assert settings.base_url == base_url


def test_hashing_embedder_is_refused_for_answer_relevancy() -> None:
    with pytest.raises(JudgeUnavailableError, match="semantic embedder"):
        require_semantic_embedder(HashingEmbedder(64))

    require_semantic_embedder(_DigestEmbedder())


@dataclass
class _Result:
    value: object


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.9, 0.9), (1, 1.0), (None, 0.0), (float("nan"), 0.0), ("0.9", 0.0), (True, 0.0)],
)
def test_metric_value_never_inflates_a_failed_score(value: object, expected: float) -> None:
    assert metric_value(_Result(value)) == expected


def test_threshold_failures_report_low_missing_and_nan_scores() -> None:
    passing = {FAITHFULNESS: 0.9, ANSWER_RELEVANCY: 0.8, CONTEXT_PRECISION: 0.75}
    assert threshold_failures(passing) == []

    failures = threshold_failures({FAITHFULNESS: 0.84, ANSWER_RELEVANCY: float("nan")})

    assert len(failures) == 3
    assert failures[0].startswith("faithfulness=0.84")


def test_thresholds_match_the_quality_gate() -> None:
    assert dict(RAGAS_THRESHOLDS) == {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
    }
    assert langfuse_scores({FAITHFULNESS: 0.9}) == {"ragas.faithfulness": 0.9}


# ── rag-marked wiring tests (ragas installed, mock transport, no network) ────


def _canned_judge_reply(request_text: str) -> dict[str, Any]:
    """A valid instance of whichever ragas response model the request asks for."""
    if "NLIStatementOutput" in request_text:
        return {
            "statements": [{"statement": "Click Forgot Password.", "reason": "ok", "verdict": 1}]
        }
    if "StatementGeneratorOutput" in request_text:
        return {"statements": ["Click Forgot Password."]}
    if "AnswerRelevanceOutput" in request_text:
        return {"question": _QUESTION, "noncommittal": 0}
    if "ContextPrecisionOutput" in request_text:
        return {"reason": "useful", "verdict": 1}
    raise AssertionError(f"unexpected judge request: {request_text[:300]}")


def _records() -> list[EvalRecord]:
    return [
        EvalRecord(
            user_input=_QUESTION,
            response="Click Forgot Password.",
            retrieved_contexts=(_CONTEXT,),
            reference="Use Forgot Password.",
            outcome="answer",
            generated=True,
        )
    ]


@pytest.mark.rag
async def test_anthropic_judge_scores_through_ragas_without_rejected_params(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ragas")
    httpx2 = pytest.importorskip("httpx2")
    import anthropic

    from faq_core.eval.ragas_eval import build_judge, score_records

    bodies: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:  # noqa: ANN401
        body = json.loads(request.content)
        bodies.append(body)
        tool = body["tools"][0]
        reply = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": tool["name"],
                    "input": _canned_judge_reply(json.dumps(tool)),
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return httpx2.Response(200, json=reply)

    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    judge = build_judge(
        _config(tmp_path, LLM_PROVIDER="anthropic"), _DigestEmbedder(), http_client=http_client
    )

    scores = await score_records(_records(), judge)

    assert scores == pytest.approx(
        {FAITHFULNESS: 1.0, ANSWER_RELEVANCY: 1.0, CONTEXT_PRECISION: 1.0}
    )
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert bodies
    for body in bodies:
        assert "temperature" not in body and "top_p" not in body
        assert body["tool_choice"] == {"type": "auto"}
        assert "calling the provided tool" in json.dumps(body["system"])
        assert body["max_tokens"] == judge.settings.max_tokens


@pytest.mark.rag
async def test_openai_compatible_judge_scores_through_ragas(tmp_path: Path) -> None:
    pytest.importorskip("ragas")
    httpx2 = pytest.importorskip("httpx2")
    import openai

    from faq_core.eval.ragas_eval import build_judge, score_records

    urls: list[str] = []

    def handler(request: Any) -> Any:  # noqa: ANN401
        urls.append(str(request.url))
        body = json.loads(request.content)
        content = json.dumps(_canned_judge_reply(json.dumps(body["messages"])))
        reply = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx2.Response(200, json=reply)

    config = _config(tmp_path, LLM_PROVIDER="ollama", LLM_MODEL="llama3.2", LLM_API_KEY="")
    http_client = openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    judge = build_judge(config, _DigestEmbedder(), http_client=http_client)

    scores = await score_records(_records(), judge)

    assert scores == pytest.approx(
        {FAITHFULNESS: 1.0, ANSWER_RELEVANCY: 1.0, CONTEXT_PRECISION: 1.0}
    )
    assert urls and all(url.startswith("http://localhost:11434/v1/") for url in urls)
