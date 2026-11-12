"""Text embedders for the dense side of hybrid retrieval.

Choice of default (``EMBED_PROVIDER=hashing``): lite installs must stay
torch-free, so the default is :class:`HashingEmbedder` — signed feature
hashing of word unigrams, word bigrams and character trigrams into a fixed
L2-normalised vector. It is deterministic across processes (BLAKE2b, not the
salted builtin ``hash``), needs no model download and makes tests reproducible.
It only captures *lexical* similarity (shared words and word fragments, which
tolerates plurals and typos), so with it the dense side mostly reinforces BM25.
For semantic recall configure a real model:

* ``local`` / ``sentence-transformers`` — :class:`SentenceTransformerEmbedder`
  (``EMBED_MODEL``, e.g. ``all-MiniLM-L6-v2`` or a BGE/GTE/E5 model); needs
  ``requirements-rag.txt``. ``EMBED_MODEL=multilingual`` selects the preset in
  :data:`EMBED_MODEL_PRESETS` (``BAAI/bge-m3``: 100+ languages, 1024-dim dense
  vectors, no query/passage prefix needed — checked against its model card
  2026-10-30), so a question in one language retrieves passages written in
  another. When the package is missing the factory logs a
  warning and falls back to hashing instead of failing startup.
* ``openai`` / ``ollama`` — :class:`OpenAICompatibleEmbedder` against any
  OpenAI-compatible ``/embeddings`` endpoint.

Every embedder exposes a ``name`` and ``dimension``; together they form the
vector-index signature, so switching embedders triggers a full re-embed at the
next startup instead of mixing incompatible vectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from faq_core.retrieval.http_json import RemoteServiceError, post_json, require_tls
from faq_core.retrieval.ranking import STOPWORDS, normalize_text
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx
    from pydantic import SecretStr

    from faq_core.config import AppConfig

_log = get_logger(__name__)

MIN_DIMENSION: Final = 16
MAX_DIMENSION: Final = 4096
_UNIGRAM_WEIGHT: Final = 1.0
_BIGRAM_WEIGHT: Final = 0.5
_TRIGRAM_WEIGHT: Final = 0.3
_BATCH_SIZE: Final = 64

EMBED_MODEL_PRESETS: Final[dict[str, str]] = {
    "multilingual": "BAAI/bge-m3",
}
"""``EMBED_MODEL`` aliases for ``EMBED_PROVIDER=local`` (sentence-transformers)."""


def resolve_embed_model(name: str) -> str:
    """Expand an :data:`EMBED_MODEL_PRESETS` alias; other names pass through."""
    return EMBED_MODEL_PRESETS.get(name.strip().lower(), name.strip())


class EmbeddingError(RemoteServiceError):
    """An embedder returned unusable output (wrong shape or dimension)."""


class Embedder(Protocol):
    """Turns text into fixed-size vectors; documents and queries may differ."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic, dependency-free lexical embedder (signed feature hashing).

    Args:
        dimension: Vector size, ``16..4096``. 384 keeps vectors small while
            leaving few collisions for a support-sized vocabulary.
    """

    def __init__(self, dimension: int = 384) -> None:
        if not MIN_DIMENSION <= dimension <= MAX_DIMENSION:
            raise ValueError(f"dimension must be {MIN_DIMENSION}..{MAX_DIMENSION}")
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "hashing-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # Pure-Python hashing is CPU work; keep large batches off the event loop.
        return await asyncio.to_thread(lambda: [self.embed_sync(text) for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return self.embed_sync(text)

    def embed_sync(self, text: str) -> list[float]:
        """Embed one text synchronously (cheap: microseconds per short text)."""
        vector = [0.0] * self._dimension
        for feature, weight in _features(text).items():
            digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
            sign = 1.0 if digest >> 63 else -1.0
            vector[digest % self._dimension] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # No usable features (empty or stopwords only): a fixed unit vector
            # keeps cosine distance defined instead of NaN.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


def _features(text: str) -> dict[str, float]:
    words = [w for w in normalize_text(text).split() if w not in STOPWORDS]
    counts: Counter[str] = Counter()
    weights: dict[str, float] = {}
    for word in words:
        counts["w:" + word] += 1
        weights["w:" + word] = _UNIGRAM_WEIGHT
        padded = f"<{word}>"
        for start in range(len(padded) - 2):
            gram = "c:" + padded[start : start + 3]
            counts[gram] += 1
            weights[gram] = _TRIGRAM_WEIGHT
    for left, right in zip(words, words[1:], strict=False):
        key = f"b:{left} {right}"
        counts[key] += 1
        weights[key] = _BIGRAM_WEIGHT
    # Sublinear term frequency: repeating a word should not dominate the vector.
    return {key: weights[key] * (1.0 + math.log(count)) for key, count in counts.items()}


class SentenceTransformerEmbedder:
    """Local transformer embeddings via ``sentence-transformers`` (imported lazily).

    Vectors are L2-normalised so cosine distance in sqlite-vec is meaningful.
    Encoding runs in a worker thread (CPU/GPU-bound).
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model: Any = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension() or 0)
        if self._dimension <= 0:
            raise EmbeddingError(f"model {model_name!r} does not report an embedding dimension")

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text]))[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=_BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True
        )
        return cast("list[list[float]]", vectors.tolist())


class OpenAICompatibleEmbedder:
    """Embeddings from any OpenAI-compatible ``POST {base_url}/embeddings`` API.

    The returned dimension must equal *dimension* (``EMBED_DIM``), because the
    vector table is created before the first call.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: SecretStr | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        require_tls(base_url, setting="EMBED_BASE_URL")
        if not MIN_DIMENSION <= dimension <= MAX_DIMENSION:
            raise ValueError(f"EMBED_DIM must be {MIN_DIMENSION}..{MAX_DIMENSION}")
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._transport = transport
        key = api_key.get_secret_value() if api_key is not None else ""
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    @property
    def name(self) -> str:
        return f"openai-compatible:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            vectors.extend(await self._embed(list(texts[start : start + _BATCH_SIZE])))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        body = await post_json(
            self._url,
            {"model": self._model, "input": texts},
            service="embedding API",
            headers=self._headers,
            request_timeout=self._timeout,
            transport=self._transport,
        )
        try:
            data: list[dict[str, Any]] = cast("dict[str, Any]", body)["data"]
            items = sorted(data, key=lambda item: int(item["index"]))
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            raise EmbeddingError("embedding API returned an unexpected body") from err
        if len(vectors) != len(texts) or any(len(v) != self._dimension for v in vectors):
            raise EmbeddingError(
                f"embedding API returned vectors that do not match EMBED_DIM={self._dimension}"
            )
        return vectors


def create_embedder(config: AppConfig) -> Embedder:
    """Build the embedder selected by ``EMBED_PROVIDER`` (see module docstring).

    Raises:
        ValueError: unknown provider, or an invalid ``EMBED_DIM``/base URL.
    """
    provider = config.EMBED_PROVIDER.strip().lower()
    if provider in ("", "hashing"):
        return HashingEmbedder(config.EMBED_DIM)
    if provider in ("local", "sentence-transformers", "sentence_transformers"):
        if importlib.util.find_spec("sentence_transformers") is None:
            _log.warning(
                "sentence-transformers is not installed; falling back to the hashing embedder "
                "(install requirements-rag.txt for semantic embeddings)",
                extra={"embed_provider": provider, "embed_model": config.EMBED_MODEL},
            )
            return HashingEmbedder(config.EMBED_DIM)
        return SentenceTransformerEmbedder(resolve_embed_model(config.EMBED_MODEL))
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            base_url=config.EMBED_BASE_URL,
            model=config.EMBED_MODEL,
            dimension=config.EMBED_DIM,
            api_key=config.EMBED_API_KEY,
        )
    if provider == "ollama":
        return OpenAICompatibleEmbedder(
            base_url=config.OLLAMA_BASE_URL, model=config.EMBED_MODEL, dimension=config.EMBED_DIM
        )
    raise ValueError(f"Unknown EMBED_PROVIDER: {provider!r}")
