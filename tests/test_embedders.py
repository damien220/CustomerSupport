"""Embedders and rerankers: the lite default, factories, and the HTTP adapters.

The HTTP adapters are exercised against ``httpx.MockTransport`` — the network is
the system boundary; request building and response parsing are real.
"""

from __future__ import annotations

import json
import math

import httpx
import pytest
from pydantic import SecretStr

from faq_core.config import AppConfig
from faq_core.retrieval import embedders, rerankers
from faq_core.retrieval.embedders import (
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)
from faq_core.retrieval.http_json import RemoteServiceError, require_tls
from faq_core.retrieval.rerankers import (
    CohereReranker,
    NoOpReranker,
    RerankerError,
    create_reranker,
)


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _config(**overrides: object) -> AppConfig:
    return AppConfig(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


# ── hashing embedder ─────────────────────────────────────────────────────────


async def test_hashing_embedder_is_deterministic_and_unit_length() -> None:
    embedder = HashingEmbedder(64)

    first = await embedder.embed_query("Where is my parcel?")
    [second] = await embedder.embed_documents(["Where is my parcel?"])

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


async def test_hashing_embedder_ranks_lexically_related_text_higher() -> None:
    embedder = HashingEmbedder()
    query = await embedder.embed_query("refund for returned products")
    related, unrelated = await embedder.embed_documents(
        ["How do I return a product and get a refund?", "What are your business hours?"]
    )

    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_hashing_embedder_handles_stopword_only_text() -> None:
    vector = HashingEmbedder(32).embed_sync("how do I")

    assert vector[0] == 1.0 and sum(abs(v) for v in vector) == 1.0


def test_hashing_embedder_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbedder(8)


# ── embedder factory ─────────────────────────────────────────────────────────


def test_default_provider_is_the_hashing_embedder() -> None:
    embedder = create_embedder(_config())

    assert isinstance(embedder, HashingEmbedder)
    assert (embedder.name, embedder.dimension) == ("hashing-v1", 384)


def test_local_provider_falls_back_to_hashing_without_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(embedders.importlib.util, "find_spec", lambda _name: None)

    embedder = create_embedder(_config(EMBED_PROVIDER="local"))

    assert isinstance(embedder, HashingEmbedder)
    assert "falling back to the hashing embedder" in caplog.text


def test_unknown_embed_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="EMBED_PROVIDER"):
        create_embedder(_config(EMBED_PROVIDER="nope"))


def test_api_embedder_requires_tls_for_remote_hosts() -> None:
    with pytest.raises(ValueError, match="https"):
        create_embedder(_config(EMBED_PROVIDER="openai", EMBED_BASE_URL="http://example.com/v1"))


def test_require_tls_allows_plain_http_only_locally() -> None:
    require_tls("http://localhost:11434/v1", setting="X")
    require_tls("https://api.example.com", setting="X")
    with pytest.raises(ValueError):
        require_tls("ftp://example.com", setting="X")


# ── OpenAI-compatible embedder ───────────────────────────────────────────────


def _embedding_transport(dimension: int, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        texts = json.loads(request.content)["input"]
        data = [
            {"index": i, "embedding": [float(i + 1)] * dimension}
            for i in reversed(range(len(texts)))  # out of order on purpose
        ]
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


async def test_api_embedder_posts_inputs_and_orders_results_by_index() -> None:
    seen: list[httpx.Request] = []
    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.example.com/v1/",
        model="text-embedding-3-small",
        dimension=16,
        api_key=SecretStr("test-embed-key"),
        transport=_embedding_transport(16, seen),
    )

    vectors = await embedder.embed_documents(["a", "b"])

    assert [v[0] for v in vectors] == [1.0, 2.0]
    assert str(seen[0].url) == "https://api.example.com/v1/embeddings"
    assert seen[0].headers["authorization"].endswith("test-embed-key")
    assert json.loads(seen[0].content) == {"model": "text-embedding-3-small", "input": ["a", "b"]}


async def test_api_embedder_rejects_wrong_dimension() -> None:
    embedder = OpenAICompatibleEmbedder(
        base_url="http://localhost:9/v1",
        model="m",
        dimension=32,
        transport=_embedding_transport(16, []),
    )

    with pytest.raises(EmbeddingError, match="EMBED_DIM=32"):
        await embedder.embed_query("x")


async def test_api_errors_carry_status_but_never_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faq_core.retrieval.http_json._BACKOFF_SECONDS", 0.0)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": "busy"})

    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.example.com/v1",
        model="m",
        dimension=16,
        api_key=SecretStr("super-secret-value"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RemoteServiceError) as excinfo:
        await embedder.embed_query("x")

    assert "HTTP 503" in str(excinfo.value)
    assert "super-secret-value" not in str(excinfo.value)
    assert len(calls) == 3  # first try + two retries


# ── rerankers ────────────────────────────────────────────────────────────────


async def test_noop_reranker_keeps_order() -> None:
    results = await NoOpReranker().rerank("q", ["a", "b", "c"], top_n=2)

    assert [(r.index, r.score) for r in results] == [(0, None), (1, None)]


async def test_cohere_reranker_maps_results() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 7, "relevance_score": 0.1},  # out of range: ignored
                ]
            },
        )

    reranker = CohereReranker(
        api_key=SecretStr("test-rerank-key"), transport=httpx.MockTransport(handler)
    )

    results = await reranker.rerank("refunds", ["a", "b", "c"], top_n=2)

    assert [(r.index, r.score) for r in results] == [(2, 0.9), (0, 0.2)]
    assert seen[0]["top_n"] == 2 and seen[0]["model"] == "rerank-v3.5"


async def test_cohere_reranker_wraps_bad_bodies() -> None:
    reranker = CohereReranker(
        api_key=SecretStr("k" * 10),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"oops": []})),
    )

    with pytest.raises(RerankerError):
        await reranker.rerank("q", ["a"], top_n=1)


def test_reranker_factory() -> None:
    assert isinstance(create_reranker(_config()), NoOpReranker)
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        create_reranker(_config(RERANKER="cohere"))
    with pytest.raises(ValueError, match="RERANKER"):
        create_reranker(_config(RERANKER="mystery"))


def test_bge_reranker_falls_back_without_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rerankers.importlib.util, "find_spec", lambda _name: None)

    assert isinstance(create_reranker(_config(RERANKER="bge")), NoOpReranker)


def test_multilingual_alias_resolves_to_a_documented_model() -> None:
    from faq_core.retrieval.embedders import EMBED_MODEL_PRESETS, resolve_embed_model

    assert resolve_embed_model("multilingual") == EMBED_MODEL_PRESETS["multilingual"]
    assert resolve_embed_model(" Multilingual ") == "BAAI/bge-m3"
    assert resolve_embed_model("intfloat/e5-base") == "intfloat/e5-base"


def test_multilingual_preset_falls_back_to_hashing_without_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    from faq_core.config import AppConfig
    from faq_core.retrieval.embedders import HashingEmbedder, create_embedder

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    config = AppConfig(_env_file=None, EMBED_PROVIDER="local", EMBED_MODEL="multilingual")  # type: ignore[call-arg]

    assert isinstance(create_embedder(config), HashingEmbedder)
