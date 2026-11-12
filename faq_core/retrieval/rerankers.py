"""Optional second-stage rerankers (``RERANKER=none|cohere|bge``).

The default, :class:`NoOpReranker`, keeps the fused RRF order. The two real
adapters are opt-in and import nothing heavy until they are constructed:

* :class:`CohereReranker` — Cohere's hosted ``/v2/rerank`` API (``rerank-v3.5``
  by default); needs ``RERANK_API_KEY``. Plain HTTP via ``httpx``.
* :class:`BGEReranker` — the self-hostable ``BAAI/bge-reranker-v2-m3``
  cross-encoder via ``sentence-transformers`` (``requirements-rag.txt``).

Both return relevance scores in ``0..1``, which the retriever then uses as the
passage confidence. A reranker failure at query time never fails the search:
the retriever logs it and keeps the fused order.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from faq_core.retrieval.http_json import RemoteServiceError, post_json, require_tls
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx
    from pydantic import SecretStr

    from faq_core.config import AppConfig

_log = get_logger(__name__)

DEFAULT_COHERE_MODEL: Final = "rerank-v3.5"
DEFAULT_COHERE_URL: Final = "https://api.cohere.com/v2/rerank"
DEFAULT_BGE_MODEL: Final = "BAAI/bge-reranker-v2-m3"


class RerankerError(RemoteServiceError):
    """A reranker call failed or returned an unusable body."""


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Position of a candidate in the input list and its relevance (``None`` = unscored)."""

    index: int
    score: float | None


class Reranker(Protocol):
    """Reorders candidate passages for a query; returns at most *top_n* results."""

    @property
    def name(self) -> str: ...

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankResult]: ...


class NoOpReranker:
    """Keeps the incoming (fused) order; the default."""

    @property
    def name(self) -> str:
        return "none"

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        return [RerankResult(index=i, score=None) for i in range(min(top_n, len(documents)))]


class CohereReranker:
    """Cohere Rerank API adapter (``POST /v2/rerank``)."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = DEFAULT_COHERE_MODEL,
        url: str = DEFAULT_COHERE_URL,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key.get_secret_value()
        if not key:
            raise ValueError("RERANKER=cohere requires RERANK_API_KEY")
        require_tls(url, setting="Cohere rerank URL")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._model = model
        self._url = url
        self._timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        try:
            body = await post_json(
                self._url,
                {
                    "model": self._model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": min(top_n, len(documents)),
                },
                service="Cohere rerank",
                headers=self._headers,
                request_timeout=self._timeout,
                transport=self._transport,
            )
        except RemoteServiceError as err:
            raise RerankerError(str(err)) from err
        try:
            results: list[dict[str, Any]] = cast("dict[str, Any]", body)["results"]
            return [
                RerankResult(index=int(item["index"]), score=float(item["relevance_score"]))
                for item in results
                if 0 <= int(item["index"]) < len(documents)
            ]
        except (KeyError, TypeError, ValueError) as err:
            raise RerankerError("Cohere rerank returned an unexpected body") from err


class BGEReranker:
    """Local BGE Reranker v2 cross-encoder (``sentence-transformers``, imported lazily)."""

    def __init__(self, model_name: str = DEFAULT_BGE_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model: Any = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return f"bge:{self._model_name}"

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        raw = await asyncio.to_thread(self._model.predict, [(query, d) for d in documents])
        # Recent sentence-transformers already apply a sigmoid for single-label
        # cross-encoders; older ones return logits. Sigmoid is monotonic, so only
        # the reported score (not the order) depends on this.
        scores = [
            s if 0.0 <= s <= 1.0 else _sigmoid(s)
            for s in (float(v) for v in cast("list[float]", list(raw)))
        ]
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [RerankResult(index=i, score=scores[i]) for i in order[:top_n]]


def _sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def create_reranker(config: AppConfig) -> Reranker:
    """Build the reranker selected by ``RERANKER`` (``none`` by default).

    Raises:
        ValueError: unknown ``RERANKER`` value, or ``cohere`` without an API key.
    """
    choice = config.RERANKER.strip().lower()
    if choice in ("", "none", "off"):
        return NoOpReranker()
    if choice == "cohere":
        return CohereReranker(
            api_key=config.RERANK_API_KEY, model=config.RERANK_MODEL or DEFAULT_COHERE_MODEL
        )
    if choice in ("bge", "local"):
        if importlib.util.find_spec("sentence_transformers") is None:
            _log.warning(
                "RERANKER=bge needs sentence-transformers (requirements-rag.txt); "
                "reranking disabled",
                extra={"reranker": choice},
            )
            return NoOpReranker()
        return BGEReranker(config.RERANK_MODEL or DEFAULT_BGE_MODEL)
    raise ValueError(f"Unknown RERANKER: {choice!r}")
