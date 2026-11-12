"""RAGAS evaluation of generated answers: faithfulness, answer relevancy, context precision.

Pinned to ``ragas==0.4.3`` (see ``requirements-rag.txt``) and its *collections*
metric API — ``ragas.metrics.collections.{Faithfulness, AnswerRelevancy,
ContextPrecision}`` scored one sample at a time with ``await metric.ascore(...)``,
each returning a ``MetricResult`` whose ``.value`` is a float. The 0.1.x
``evaluate(Dataset, metrics=[faithfulness, ...])`` path the old test used is
legacy in 0.4 and is removed in ragas 1.0.

Flow: :func:`collect_records` asks the real :class:`~faq_core.service.SupportService`
for each sample (``answer`` for the response, ``search_kb`` for the full passage
texts the answer was grounded on); :func:`build_judge` turns the project's own
configuration into a ragas judge — the configured LLM (``LLM_PROVIDER`` preset,
same key settings as the gateway) and the retrieval index's embedder;
:func:`score_records` averages each metric; :func:`threshold_failures` compares the
means with :data:`RAGAS_THRESHOLDS`.

UNVERIFIED live: written against the ragas 0.4.3 / instructor 1.17.0 sources and
import-checked in a throwaway environment, but never run against a real LLM
(no key was available). Known caveats:

* The judge is the same model that wrote the answers (self-grading bias).
* Anthropic judge: ragas drives it through ``instructor.from_anthropic`` in tool
  mode. Current Claude models reject forced ``tool_choice`` and the
  ``temperature``/``top_p`` sampling parameters, so :func:`build_judge` sends
  ``tool_choice=auto`` with a system instruction and drops both parameters.
* ``AnswerRelevancy`` needs a *semantic* embedder; the default hashing embedder
  is refused (:class:`JudgeUnavailableError`).
* ragas sends usage analytics by default; :func:`disable_ragas_telemetry` sets
  ``RAGAS_DO_NOT_TRACK=true`` before any ragas call.
"""

from __future__ import annotations

import asyncio
import ipaddress
import math
import os
import statistics
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

from faq_core.llm.presets import WireProtocol, api_key_for, resolve_llm_settings
from faq_core.network_guard import (
    NetworkPolicyError,
    host_of,
    is_local_url,
    parse_allowed_hosts,
)
from faq_core.retrieval.embedders import HashingEmbedder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from faq_core.config import AppConfig
    from faq_core.models import KBSearchResult, SearchResult
    from faq_core.retrieval.embedders import Embedder

FAITHFULNESS: Final = "faithfulness"
ANSWER_RELEVANCY: Final = "answer_relevancy"
CONTEXT_PRECISION: Final = "context_precision"

RAGAS_THRESHOLDS: Final[Mapping[str, float]] = MappingProxyType(
    {FAITHFULNESS: 0.85, ANSWER_RELEVANCY: 0.80, CONTEXT_PRECISION: 0.75}
)
"""Minimum mean score per metric — the RAGAS quality gate."""

_ANTHROPIC_TOOL_INSTRUCTION: Final = (
    "Reply only by calling the provided tool, exactly once, with arguments that match its schema."
)
"""Replaces the forced ``tool_choice`` that current Claude models reject."""

_ANTHROPIC_DROPPED_ARGS: Final = ("temperature", "top_p")
"""ragas' default sampling arguments; current Claude models reject them with a 400."""

_RAGAS_DO_NOT_TRACK: Final = "RAGAS_DO_NOT_TRACK"
"""ragas posts usage analytics to its vendor by default unless this is ``true``."""

_NO_KEY_PLACEHOLDER: Final = "unused"
"""The OpenAI SDK needs a non-empty key; keyless local runtimes ignore it."""


class JudgeUnavailableError(RuntimeError):
    """The configuration cannot drive a RAGAS judge (the message says why; never a secret)."""


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One curated question with its reference (ground-truth) answer."""

    question: str
    reference: str


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """What the service produced for one :class:`EvalSample`, in ragas field names."""

    user_input: str
    response: str
    retrieved_contexts: tuple[str, ...]
    reference: str
    outcome: str
    generated: bool


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    """The effective judge LLM endpoint (no secrets)."""

    provider: str
    protocol: WireProtocol
    model: str
    base_url: str
    max_tokens: int
    timeout: float


@dataclass(frozen=True, slots=True)
class RagasJudge:
    """A ragas judge LLM (``InstructorBaseRagasLLM``) and embeddings (``BaseRagasEmbedding``)."""

    settings: JudgeSettings
    llm: Any  # ragas ships no stubs we can rely on in the lite env
    embeddings: Any


class AnswerSource(Protocol):
    """The two :class:`~faq_core.service.SupportService` calls the evaluation needs."""

    async def answer(
        self, query: str, top_k: int = 3, *, session_id: str | None = None
    ) -> SearchResult: ...

    async def search_kb(
        self, query: str, top_k: int = 3, *, session_id: str | None = None
    ) -> KBSearchResult: ...


async def collect_records(
    service: AnswerSource, samples: Sequence[EvalSample], *, top_k: int = 3
) -> list[EvalRecord]:
    """Answer every sample through *service* and pair it with its retrieved passages.

    ``search_kb`` never calls the LLM and retrieves with the same query and
    *top_k* as ``answer``, so its full passage texts are the contexts the
    answer was generated from (citations carry only snippets).

    Raises:
        AnswerGenerationUnavailableError: ``LLM_MODE=client``.
    """
    records: list[EvalRecord] = []
    for sample in samples:
        answered = await service.answer(sample.question, top_k=top_k)
        passages = await service.search_kb(sample.question, top_k=top_k)
        records.append(
            EvalRecord(
                user_input=sample.question,
                response=answered.answer,
                retrieved_contexts=tuple(passage.text for passage in passages.passages),
                reference=sample.reference,
                outcome=answered.outcome.value,
                generated=answered.generated,
            )
        )
    return records


def resolve_judge_settings(config: AppConfig) -> JudgeSettings:
    """Derive the judge LLM from the project's ``LLM_*`` settings.

    Raises:
        JudgeUnavailableError: ``LLM_MODE=client`` (no server-side LLM), an
            invalid provider/model, a required key that is not configured, or
            plain ``http`` to a host that is not loopback, a private IP or listed
            in ``OFFLINE_ALLOWED_HOSTS`` (decided by the endpoint's host, not by
            the preset: ``LLM_PROVIDER=ollama`` pointed at a remote host over
            ``http`` is refused).
    """
    if not config.LLM_MODE.generates_answers:
        raise JudgeUnavailableError(
            f"LLM_MODE={config.LLM_MODE.value} has no server-side LLM to judge with"
        )
    try:
        resolved = resolve_llm_settings(config)
        api_key_for(config, resolved)  # validates presence only; the value is read later
    except ValueError as err:
        raise JudgeUnavailableError(str(err)) from err
    if not resolved.base_url.lower().startswith("https://") and not _is_trusted_plain_http(
        resolved.base_url, config.OFFLINE_ALLOWED_HOSTS
    ):
        raise JudgeUnavailableError(
            f"LLM_PROVIDER={resolved.provider} judge endpoint must use https (non-local host)"
        )
    return JudgeSettings(
        provider=resolved.provider,
        protocol=resolved.protocol,
        model=resolved.model,
        base_url=resolved.base_url,
        max_tokens=resolved.max_tokens,
        timeout=config.LLM_TIMEOUT,
    )


def _is_trusted_plain_http(url: str, allowed_hosts_setting: str) -> bool:
    """Whether *url* may be called without TLS: loopback, a private IP, or an allowed host."""
    try:
        allowed = parse_allowed_hosts(allowed_hosts_setting)
    except NetworkPolicyError:
        allowed = frozenset()  # a public IP in the list is never trusted
    if is_local_url(url, allowed):
        return True
    try:
        address = ipaddress.ip_address(host_of(url).strip("[]"))
    except ValueError:
        return False  # a host name that is not localhost or allow-listed
    return address.is_private and not address.is_unspecified


def require_semantic_embedder(embedder: Embedder) -> None:
    """Refuse the hashing embedder: answer relevancy compares meanings, not tokens.

    Raises:
        JudgeUnavailableError: *embedder* is the lexical hashing embedder (the
            default, and the silent fallback when sentence-transformers is missing).
    """
    if isinstance(embedder, HashingEmbedder):
        raise JudgeUnavailableError(
            "answer_relevancy needs a semantic embedder: set EMBED_PROVIDER=local "
            "(with requirements-rag.txt installed) or EMBED_PROVIDER=openai|ollama"
        )


def build_judge(
    config: AppConfig,
    embedder: Embedder,
    *,
    http_client: Any = None,  # noqa: ANN401 — the provider SDK's own async HTTP client type
) -> RagasJudge:
    """Build the ragas judge LLM and embeddings from *config* and the index *embedder*.

    Imports ragas (and the provider SDK) lazily — call only when ragas is installed.
    *http_client* is handed to the provider SDK client unchanged (a proxy, or a
    mock transport in tests); ``None`` keeps the SDK default.

    Raises:
        JudgeUnavailableError: see :func:`resolve_judge_settings` and
            :func:`require_semantic_embedder`.
        ValueError: ragas could not wrap the client.
    """
    settings = resolve_judge_settings(config)
    require_semantic_embedder(embedder)
    disable_ragas_telemetry()
    key = api_key_for(config, resolve_llm_settings(config)).get_secret_value()
    if settings.protocol is WireProtocol.ANTHROPIC:
        llm = _anthropic_judge_llm(settings, key, http_client)
    else:
        llm = _openai_compatible_judge_llm(settings, key, http_client)
    return RagasJudge(settings=settings, llm=llm, embeddings=_ragas_embeddings(embedder))


def disable_ragas_telemetry() -> None:
    """Turn off ragas' default usage analytics (an outbound call to the vendor).

    Must run before ragas' first tracked call: it caches the setting. Forced on
    rather than defaulted, so offline mode can never phone home through it.
    """
    os.environ[_RAGAS_DO_NOT_TRACK] = "true"
    analytics = sys.modules.get("ragas._analytics")
    cached = getattr(analytics, "do_not_track", None)
    cache_clear = getattr(cached, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def _anthropic_judge_llm(settings: JudgeSettings, key: str, http_client: Any) -> Any:  # noqa: ANN401
    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory

    client = AsyncAnthropic(
        api_key=key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        http_client=http_client,
    )
    llm = llm_factory(
        settings.model,
        provider="anthropic",
        client=client,
        max_tokens=settings.max_tokens,
        tool_choice={"type": "auto"},
        system_prompt=_ANTHROPIC_TOOL_INSTRUCTION,
    )
    model_args: dict[str, Any] = llm.model_args
    for name in _ANTHROPIC_DROPPED_ARGS:
        model_args.pop(name, None)
    return llm


def _openai_compatible_judge_llm(settings: JudgeSettings, key: str, http_client: Any) -> Any:  # noqa: ANN401
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key=key or _NO_KEY_PLACEHOLDER,
        base_url=settings.base_url,
        timeout=settings.timeout,
        http_client=http_client,
    )
    return llm_factory(
        settings.model, provider="openai", client=client, max_tokens=settings.max_tokens
    )


def _ragas_embeddings(embedder: Embedder) -> Any:  # noqa: ANN401 — ragas embeddings
    """Wrap the project's async :class:`Embedder` as a ragas ``BaseRagasEmbedding``.

    Reusing the index embedder keeps every ``EMBED_PROVIDER`` (local, openai,
    ollama) and the offline network guard in force for the judge as well.
    """
    from ragas.embeddings.base import BaseRagasEmbedding

    # BaseRagasEmbedding is Any while ragas is absent (the lite mypy gate).
    class _ProjectEmbeddings(BaseRagasEmbedding):  # type: ignore[misc]
        def __init__(self, inner: Embedder) -> None:
            super().__init__()
            self._inner = inner

        def embed_text(self, text: str, **_kwargs: Any) -> list[float]:  # noqa: ANN401
            return asyncio.run(self._inner.embed_query(text))

        async def aembed_text(self, text: str, **_kwargs: Any) -> list[float]:  # noqa: ANN401
            return await self._inner.embed_query(text)

        def embed_texts(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:  # noqa: ANN401
            return asyncio.run(self._inner.embed_documents(texts))

        async def aembed_texts(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:  # noqa: ANN401
            return await self._inner.embed_documents(texts)

    return _ProjectEmbeddings(embedder)


async def score_records(records: Sequence[EvalRecord], judge: RagasJudge) -> dict[str, float]:
    """Score *records* with the three collections metrics; return each metric's mean.

    A sample whose score is missing or NaN counts as 0.0 so a judge failure can
    never raise the mean.

    Raises:
        ValueError: *records* is empty, or ragas rejects the judge components.
    """
    if not records:
        raise ValueError("no records to score")
    disable_ragas_telemetry()
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    faithfulness = Faithfulness(llm=judge.llm)
    relevancy = AnswerRelevancy(llm=judge.llm, embeddings=judge.embeddings)
    precision = ContextPrecision(llm=judge.llm)

    per_metric: dict[str, list[float]] = {name: [] for name in RAGAS_THRESHOLDS}
    for record in records:
        contexts = list(record.retrieved_contexts)
        # One sample at a time keeps the request rate low; the three metrics run together.
        results = await asyncio.gather(
            faithfulness.ascore(
                user_input=record.user_input,
                response=record.response,
                retrieved_contexts=contexts,
            ),
            relevancy.ascore(user_input=record.user_input, response=record.response),
            precision.ascore(
                user_input=record.user_input,
                reference=record.reference,
                retrieved_contexts=contexts,
            ),
        )
        for name, result in zip(
            (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION), results, strict=True
        ):
            per_metric[name].append(metric_value(result))
    return {name: statistics.fmean(values) for name, values in per_metric.items()}


def metric_value(result: object) -> float:
    """``MetricResult.value`` as a float; missing, non-numeric or NaN values become 0.0."""
    value = getattr(result, "value", None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    number = float(value)
    return 0.0 if math.isnan(number) else number


def threshold_failures(
    scores: Mapping[str, float], thresholds: Mapping[str, float] = RAGAS_THRESHOLDS
) -> list[str]:
    """Describe every metric below its threshold (a missing score fails); empty = pass."""
    failures: list[str] = []
    for name, minimum in thresholds.items():
        score = scores.get(name)
        if score is None or math.isnan(score) or score < minimum:
            failures.append(f"{name}={score} < {minimum}")
    return failures


def langfuse_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Prefix metric names for :func:`observability.tracing.push_evaluation_scores`."""
    return {f"ragas.{name}": value for name, value in scores.items()}
