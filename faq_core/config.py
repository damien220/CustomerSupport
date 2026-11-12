"""Application configuration — the only place ``.env`` is read."""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from faq_core.models import LLMMode

# Settings that LLM_MODE=offline fills in when the operator did not set them:
# a local LLM runtime, the local embedder and the local BGE reranker (the last
# two fall back to hashing / no reranking when sentence-transformers is absent).
_OFFLINE_DEFAULTS: Final[dict[str, str]] = {
    "LLM_PROVIDER": "ollama",
    "EMBED_PROVIDER": "local",
    "RERANKER": "bge",
}


class AppConfig(BaseSettings):
    # ── LLM access mode (plan §2) ─────────────────────────────────────────────
    # api (default): this server calls the LLM provider below.
    # client: no LLM here — the calling agent (e.g. a Claude/ChatGPT subscriber's
    #   chat app over MCP) writes replies from search_kb passages; no key needed.
    # offline: local LLM/embedder/reranker only; startup refuses any non-local
    #   endpoint (see faq_core/network_guard.py).
    LLM_MODE: LLMMode = LLMMode.API

    # ── LLM provider (used in api/offline mode) ───────────────────────────────
    # A preset from faq_core/llm/presets.py: anthropic (default) | openai | gemini
    # | groq | openrouter | mistral | cohere | together | ollama | vllm | llamacpp
    # | lmstudio | custom. Blank LLM_MODEL / LLM_BASE_URL / LLM_MAX_TOKENS use
    # the preset's values; set them to override.
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    LLM_MAX_TOKENS: int = 0  # 0 = preset default
    LLM_TIMEOUT: float = 60.0  # seconds per request (each retry gets its own)
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")  # LLM_PROVIDER=anthropic
    LLM_API_KEY: SecretStr = SecretStr("")  # every other hosted provider (optional for local)
    # Anthropic only: let the API re-run a request a model declines on a
    # fallback model (server-side fallbacks beta; supported models only).
    ANTHROPIC_REFUSAL_FALLBACK: bool = True
    # Local runtime base URLs — still honoured for their presets (LLM_BASE_URL
    # wins). Override when the server is not on localhost.
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"

    # ── Offline network guard (LLM_MODE=offline) ──────────────────────────────
    # Comma-separated hosts besides loopback that offline mode may reach, e.g.
    # "host.docker.internal,gpu-box.lan,10.0.0.5". IP literals must be private.
    OFFLINE_ALLOWED_HOSTS: str = ""

    # ── Embedding (dense side of hybrid retrieval) ────────────────────────────
    # hashing (default) — deterministic feature-hashing embedder, torch-free;
    #   lexical similarity only (see faq_core/retrieval/embedders.py).
    # local | sentence-transformers — EMBED_MODEL via sentence-transformers
    #   (requirements-rag.txt); falls back to hashing with a warning if absent.
    # openai — any OpenAI-compatible /embeddings API at EMBED_BASE_URL.
    # ollama — Ollama's OpenAI-compatible /embeddings at OLLAMA_BASE_URL.
    EMBED_PROVIDER: str = "hashing"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    EMBED_API_KEY: SecretStr = SecretStr("")
    EMBED_BASE_URL: str = "https://api.openai.com/v1"
    EMBED_DIM: int = 384  # hashing / API embedders; sentence-transformers reports its own

    # ── Vector Store ──────────────────────────────────────────────────────────
    # sqlite-vec (default; same SQLite file) | none (BM25 only). The removed
    # chroma/inmemory backends and the not-yet-built qdrant one fall back to
    # sqlite-vec with a warning.
    VECTOR_STORE: str = "sqlite-vec"
    QDRANT_URL: str = "http://localhost:6333"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_RRF_K: int = 60  # Reciprocal Rank Fusion constant (Cormack et al., 2009)
    RETRIEVAL_CANDIDATES: int = 30  # per-retriever candidate pool before fusion
    RETRIEVAL_DEDUPE_THRESHOLD: float = 0.85  # word-shingle Jaccard ≥ this = duplicate
    CHUNK_MAX_CHARS: int = 1200
    # Optional reranker: none (default) | cohere (API) | bge (local BGE Reranker v2)
    RERANKER: str = "none"
    RERANK_MODEL: str = ""  # blank = adapter default (rerank-v3.5 / BAAI/bge-reranker-v2-m3)
    RERANK_API_KEY: SecretStr = SecretStr("")

    # ── Ingestion (python app.py --mode ingest --source ...) ─────────────────
    INGEST_MAX_BYTES: int = 5_000_000  # per file / per fetched URL
    INGEST_URL_TIMEOUT: float = 15.0  # seconds, per request

    # ── Database ──────────────────────────────────────────────────────────────
    DB_PATH: str = "./data/faq.db"

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Blank (default) = in-process MemoryDataStore, lost on restart. Set to a
    # SQLAlchemy URL (e.g. "sqlite+aiosqlite:///./data/scheduler.db") for a
    # persistent re-index queue that survives restarts.
    SCHEDULER_DB_URL: str = ""

    # ── Cache ─────────────────────────────────────────────────────────────────
    # In-process answer cache (faq_core/cache.py). Per process: each worker and
    # each surface keeps its own copy.
    CACHE_TTL: int = 300
    CACHE_MAX_ENTRIES: int = 1024

    # ── Confidence (faq_core/confidence.py) ───────────────────────────────────
    # score >= ANSWER → answer; >= CAVEAT → answer_with_caveat; below → escalate.
    # Tuned on data/retrieval_benchmark.json with the hashing embedder; the score
    # is a heuristic, not a calibrated probability. The similarity floor/ceiling
    # map dense cosine onto 0..1 — re-check them for a semantic embedder.
    CONFIDENCE_ANSWER_THRESHOLD: float = 0.55
    CONFIDENCE_CAVEAT_THRESHOLD: float = 0.30
    CONFIDENCE_SIMILARITY_FLOOR: float = 0.05
    CONFIDENCE_SIMILARITY_CEILING: float = 0.35
    # Optional LLM self-check of each generated answer (api/offline only; one
    # extra LLM call per answer; never runs in client mode). Off by default.
    CONFIDENCE_SELF_CHECK: bool = False

    # ── Escalation triggers (faq_core/triage.py) ──────────────────────────────
    # Comma-separated groups: low_confidence, repeated_rephrase, frustration,
    # pii, sensitive_topic — or "all" (default) / "none".
    ESCALATION_TRIGGERS: str = "all"
    # Comma-separated phrases; blank = built-in English defaults.
    ESCALATION_FRUSTRATION_TERMS: str = ""
    ESCALATION_LEGAL_TERMS: str = ""
    ESCALATION_MEDICAL_TERMS: str = ""
    ESCALATION_REPHRASE_SIMILARITY: float = 0.5
    ESCALATION_REPHRASE_MIN_REPEATS: int = 2
    ESCALATION_REPHRASE_WINDOW: int = 5

    # ── Escalation hand-off sink (faq_core/handoff.py) ────────────────────────
    # log (default): the packet stays in the escalations table + a log line.
    # webhook: signed JSON POST to ESCALATION_WEBHOOK_URL (config only — never
    #   taken from tool/REST input). HMAC-SHA256 with ESCALATION_WEBHOOK_SECRET.
    ESCALATION_SINK: str = "log"
    ESCALATION_WEBHOOK_URL: str = ""
    ESCALATION_WEBHOOK_SECRET: SecretStr = SecretStr("")
    ESCALATION_WEBHOOK_TIMEOUT: float = 5.0
    ESCALATION_WEBHOOK_MAX_RETRIES: int = 2  # 0..5 extra attempts (429/5xx/transport)

    # ── Privacy (faq_core/pii.py) ─────────────────────────────────────────────
    # on (default): customer text is PII-redacted before logging, storage,
    # tracing and the LLM. off: stored as typed (detection still feeds the
    # pii escalation trigger). The knowledge base itself is never redacted.
    PII_REDACTION: bool = True

    # ── Knowledge gaps (faq_core/gaps.py) ─────────────────────────────────────
    GAP_CLUSTER_THRESHOLD: float = 0.35  # embedding cosine to join a cluster (hashing scale)
    GAP_SCAN_LIMIT: int = 1000  # newest gap-log rows considered per listing

    # ── Resolution metrics (faq_core/metrics.py) ──────────────────────────────
    # Max rows read per activity table (query_log, escalations, feedback) for one
    # metrics computation; beyond it only the newest rows count and the result
    # says *_scan_truncated=true. Results are cached per (window) for
    # METRICS_CACHE_TTL seconds (0 disables the cache).
    METRICS_ROW_LIMIT: int = 20_000
    METRICS_CACHE_TTL: float = 60.0
    # Operator-only MCP tools (get_resolution_metrics). They expose aggregated
    # customer questions (redacted), so keep this off on any MCP server that a
    # customer-facing agent uses; the REST metrics routes are API-key gated.
    MCP_OPERATOR_TOOLS: bool = False

    # ── Observability ─────────────────────────────────────────────────────────
    LANGFUSE_SECRET_KEY: SecretStr = SecretStr("")
    LANGFUSE_PUBLIC_KEY: SecretStr = SecretStr("")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── Alerting ──────────────────────────────────────────────────────────────
    # Off by default. When enabled, LLM circuit-breaker transitions are logged
    # (CRITICAL when the breaker opens) — see observability/alerting.py.
    ALERTING_ENABLED: bool = False

    # ── Server ────────────────────────────────────────────────────────────────
    # BIND_HOST is the interface for --mode api and --mode mcp --transport http
    # (the CLI's --host wins). Loopback by default; a container sets 0.0.0.0.
    BIND_HOST: str = "127.0.0.1"
    PORT: int = 8000  # REST + WebSocket (--mode api)
    MCP_HTTP_PORT: int = 8001  # remote MCP (--mode mcp --transport http)
    GRADIO_PORT: int = 7860

    # ── Remote access security (faq_core/security/) ───────────────────────────
    # Applies to REST/WebSocket and MCP over HTTP; MCP stdio is never
    # authenticated (the client launched the process). Fails closed: a mode
    # without its settings refuses to start.
    # token  — static bearer keys from API_KEYS (single tenant)
    # oauth  — OAuth 2.1 resource server: JWT access tokens from OAUTH_ISSUER,
    #          verified against OAUTH_JWKS_URL, audience = OAUTH_RESOURCE_URL
    # both   — either of the above
    # none   — no authentication; refused unless bound to loopback
    AUTH_MODE: str = "token"
    # Comma-separated "name:token" pairs; tokens >= 32 characters. The name
    # identifies the key in logs and rate limits; the token is never logged.
    API_KEYS: SecretStr = SecretStr("")
    OAUTH_ISSUER: str = ""  # must equal the tokens' "iss" claim exactly
    OAUTH_RESOURCE_URL: str = ""  # canonical URI of this server, e.g. https://host/mcp
    OAUTH_JWKS_URL: str = ""  # the issuer's JWKS (https unless local)
    OAUTH_ALGORITHMS: str = "RS256,ES256"  # asymmetric only; HS*/none are refused
    OAUTH_REQUIRED_SCOPES: str = ""  # comma-separated; blank = none required
    OAUTH_CLOCK_SKEW: int = 60  # seconds of leeway for exp/nbf/iat
    OAUTH_JWKS_CACHE_TTL: int = 3600  # seconds a fetched key set is trusted
    OAUTH_JWKS_TIMEOUT: float = 5.0  # seconds per JWKS fetch
    # Browser Origins allowed to call the HTTP surfaces (comma-separated
    # scheme://host[:port]). Requests without an Origin header (non-browser
    # clients) are allowed; any other Origin gets 403.
    ALLOWED_ORIGINS: str = ""
    # Host header values accepted (host or host:port, comma-separated). On a
    # loopback bind localhost / 127.0.0.1 / [::1] are always accepted and every
    # other Host gets 421 (DNS-rebinding protection); on other binds the check
    # applies only when this is set — set it to your public host name.
    ALLOWED_HOSTS: str = ""
    # Sliding-window rate limits: per authenticated key, and per client IP for
    # unauthenticated traffic (open routes and failed sign-ins). 0 disables.
    RATE_LIMIT_REQUESTS: int = 120
    RATE_LIMIT_ANON_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: float = 60.0  # seconds

    # ── Background ────────────────────────────────────────────────────────────
    REINDEX_CONCURRENCY: int = 2
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        secrets_dir="/run/secrets",
        extra="ignore",
    )

    @field_validator("LLM_MODE", mode="before")
    @classmethod
    def _normalise_mode(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("PII_REDACTION", mode="before")
    @classmethod
    def _on_off(cls, value: object) -> object:
        """Accept ``on``/``off`` as well as the usual boolean spellings."""
        if isinstance(value, str) and value.strip().lower() in ("on", "off"):
            return value.strip().lower() == "on"
        return value

    @model_validator(mode="after")
    def _apply_offline_defaults(self) -> Self:
        if self.LLM_MODE is LLMMode.OFFLINE:
            for name, value in _OFFLINE_DEFAULTS.items():
                if name not in self.model_fields_set:
                    setattr(self, name, value)
        return self

    def secret_values(self) -> list[str]:
        """Return every non-empty ``SecretStr`` value, for log redaction only."""
        values: list[str] = []
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr) and value.get_secret_value():
                values.append(value.get_secret_value())
        # API_KEYS holds several tokens; each must be redacted on its own.
        for pair in self.API_KEYS.get_secret_value().split(","):
            _, _, token = pair.partition(":")
            if token.strip():
                values.append(token.strip())
        return values


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Singleton accessor — reads .env exactly once."""
    return AppConfig()
