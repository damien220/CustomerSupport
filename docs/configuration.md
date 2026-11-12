# Configuration reference

Every setting is a field on `AppConfig` in `faq_core/config.py`, the only place `.env` is read.
Back to the [README](../README.md) or the [how-to](../howto.md).

- Source: environment variables and a `.env` file in the **current working directory**
  (`env_file=".env"`). Names are case-insensitive; unknown names are ignored. Pydantic-settings also
  reads files from `/run/secrets` (Docker/Kubernetes secrets) named after the variable.
- Environment variables win over `.env`.
- Secrets are `SecretStr` fields. Keep them only in `.env` (git-ignored) or the environment. This
  page lists names and defaults only; never paste real values into docs or issues.
- Invalid values fail at startup. Booleans accept the usual spellings; `PII_REDACTION` also accepts
  `on` and `off`.
- `.env.example` is the annotated template. Where it differs from the code default, the table says
  so.

## LLM access mode

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODE` | `api` | `api`: this server calls an LLM provider. `client`: no LLM here, the calling agent writes the reply from `search_kb` passages. `offline`: local LLM and embedder, startup refuses non-local endpoints. Case-insensitive |

In `offline` mode, `LLM_PROVIDER=ollama`, `EMBED_PROVIDER=local` and `RERANKER=bge` are applied
**only if you did not set them** (in `.env` or the environment). `.env.example` sets
`EMBED_PROVIDER=hashing` and `RERANKER=none` explicitly, which suppresses those defaults.

## LLM provider (`api` and `offline`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Preset from `faq_core/llm/presets.py`: `anthropic`, `openai`, `gemini`, `groq`, `openrouter`, `mistral`, `cohere`, `together`, `ollama`, `vllm`, `llamacpp`, `lmstudio`, `custom` |
| `LLM_MODEL` | blank | Blank uses the preset's default model. Required for local runtimes (`ollama`, `vllm`, `llamacpp`, `lmstudio`) and `custom` |
| `LLM_BASE_URL` | blank | Overrides the preset endpoint. Required for `custom` |
| `LLM_MAX_TOKENS` | `0` | `0` uses the preset value |
| `LLM_TIMEOUT` | `60.0` | Seconds per request (each retry gets its own) |
| `ANTHROPIC_API_KEY` | blank (secret) | Used when `LLM_PROVIDER=anthropic`. Hosted providers fail at startup without a key |
| `LLM_API_KEY` | blank (secret) | Every other hosted provider; optional for local runtimes (sent as a bearer token when set) |
| `ANTHROPIC_REFUSAL_FALLBACK` | `true` | Anthropic only: re-run a declined request on a fallback model, for models that support it |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Endpoint for the `ollama` preset and the `ollama` embedder (`LLM_BASE_URL` wins for the LLM) |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | `vllm` preset |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | `lmstudio` preset |

Preset defaults (model, URL) are in `faq_core/llm/presets.py`; some are unverified (vLLM default
port, default models for mistral, cohere and together). Not verified: any live LLM call.

## Offline network guard

| Variable | Default | Meaning |
|---|---|---|
| `OFFLINE_ALLOWED_HOSTS` | blank | Comma-separated hosts besides loopback that offline mode may reach (for example `host.docker.internal,gpu-box.lan,10.0.0.5`). IP literals must be private. Also covers OAuth issuer/JWKS URLs and the HTTP bind address. The guard judges hosts by name, without DNS |

## Embedding and vector store

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_PROVIDER` | `hashing` | `hashing` (torch-free, lexical only), `local` / `sentence-transformers` (needs `requirements-rag.txt`, falls back to hashing with a warning), `openai` (any OpenAI-compatible `/embeddings`), `ollama` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Model name. `multilingual` selects `BAAI/bge-m3` (1024-dim; never run here) |
| `EMBED_API_KEY` | blank (secret) | For `EMBED_PROVIDER=openai` |
| `EMBED_BASE_URL` | `https://api.openai.com/v1` | For `EMBED_PROVIDER=openai` |
| `EMBED_DIM` | `384` | For hashing and API embedders; local models report their own |
| `VECTOR_STORE` | `sqlite-vec` | `sqlite-vec` or `none` (BM25 only). Removed backends and `qdrant` fall back to `sqlite-vec` with a warning |
| `QDRANT_URL` | `http://localhost:6333` | Reserved; the Qdrant backend is not built |

Changing the embedder name or dimension triggers a full re-embed at the next start.

## Retrieval

| Variable | Default | Meaning |
|---|---|---|
| `RETRIEVAL_RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates per retriever before fusion |
| `RETRIEVAL_DEDUPE_THRESHOLD` | `0.85` | Word-shingle Jaccard at or above this counts as a duplicate |
| `CHUNK_MAX_CHARS` | `1200` | Maximum chunk size |
| `RERANKER` | `none` | `none`, `cohere` (API), `bge` (local BGE Reranker v2; falls back to none with a warning without sentence-transformers) |
| `RERANK_MODEL` | blank | Blank uses the adapter default (`rerank-v3.5` / `BAAI/bge-reranker-v2-m3`) |
| `RERANK_API_KEY` | blank (secret) | For `RERANKER=cohere` |

## Ingestion

| Variable | Default | Meaning |
|---|---|---|
| `INGEST_MAX_BYTES` | `5000000` | Per file and per fetched URL |
| `INGEST_URL_TIMEOUT` | `15.0` | Seconds per request |

## Storage, scheduler, cache

| Variable | Default | Meaning |
|---|---|---|
| `DB_PATH` | `./data/faq.db` | SQLite database (relative to the working directory) |
| `SCHEDULER_DB_URL` | blank | Blank: in-memory re-index queue (lost on restart). Set a SQLAlchemy URL such as `sqlite+aiosqlite:///./data/scheduler.db` for a persistent queue |
| `CACHE_TTL` | `300` | Seconds identical queries reuse an answer (in process) |
| `CACHE_MAX_ENTRIES` | `1024` | Answer cache size |
| `REINDEX_CONCURRENCY` | `2` | Concurrent background re-index jobs |

## Confidence

| Variable | Default | Meaning |
|---|---|---|
| `CONFIDENCE_ANSWER_THRESHOLD` | `0.55` | Score at or above: outcome `answer` |
| `CONFIDENCE_CAVEAT_THRESHOLD` | `0.30` | Score at or above: `answer_with_caveat`; below: `escalate` |
| `CONFIDENCE_SIMILARITY_FLOOR` | `0.05` | Dense cosine mapped to 0 |
| `CONFIDENCE_SIMILARITY_CEILING` | `0.35` | Dense cosine mapped to 1 |
| `CONFIDENCE_SELF_CHECK` | `false` | One extra LLM call per generated answer that can only lower the outcome. Never runs in `client` mode |

The score is a heuristic and `calibrated` is always `false`. The values were tuned with the hashing
embedder; re-check them after switching `EMBED_PROVIDER`.

## Escalation triggers and hand-off

| Variable | Default | Meaning |
|---|---|---|
| `ESCALATION_TRIGGERS` | `all` | `all`, `none`, or a comma list of `low_confidence`, `repeated_rephrase`, `frustration`, `pii`, `sensitive_topic` |
| `ESCALATION_FRUSTRATION_TERMS` | blank | Comma-separated phrases; blank uses built-in English lists |
| `ESCALATION_LEGAL_TERMS` | blank | Same |
| `ESCALATION_MEDICAL_TERMS` | blank | Same |
| `ESCALATION_REPHRASE_SIMILARITY` | `0.5` | Similarity for two queries to count as a rephrase |
| `ESCALATION_REPHRASE_MIN_REPEATS` | `2` | Escalate when at least this many of the session's recent queries are similar |
| `ESCALATION_REPHRASE_WINDOW` | `5` | How many recent session queries are compared |
| `ESCALATION_SINK` | `log` | `log` (packet stays in the `escalations` table) or `webhook` |
| `ESCALATION_WEBHOOK_URL` | blank | Required for `webhook`. Non-local hosts need TLS. In `offline` mode it must be local |
| `ESCALATION_WEBHOOK_SECRET` | blank (secret) | HMAC-SHA256 key for `X-Support-Signature` |
| `ESCALATION_WEBHOOK_TIMEOUT` | `5.0` | Seconds per attempt |
| `ESCALATION_WEBHOOK_MAX_RETRIES` | `2` | Extra attempts (0 to 5) on transport errors, 429 and 5xx |

## Privacy and knowledge gaps

| Variable | Default | Meaning |
|---|---|---|
| `PII_REDACTION` | `true` | `on`/`true`: customer text is redacted before logging, storage, traces and the LLM. `off`/`false`: stored as typed; detection still feeds the PII trigger. The KB is never redacted |
| `GAP_CLUSTER_THRESHOLD` | `0.35` | Embedding cosine to join a gap cluster (hashing scale) |
| `GAP_SCAN_LIMIT` | `1000` | Newest gap-log rows considered per listing; the report flags truncation |

## Resolution metrics

| Variable | Default | Meaning |
|---|---|---|
| `METRICS_ROW_LIMIT` | `20000` | Max rows read per table (`query_log`, `escalations`, `feedback`) for one metrics computation; beyond it only the newest rows count and `*_scan_truncated` is `true`. Must be >= 1 |
| `METRICS_CACHE_TTL` | `60` | Seconds a computed metrics result is reused per window (`0` disables the cache). Must be >= 0 |
| `MCP_OPERATOR_TOOLS` | `false` | Registers the operator-only MCP tool `get_resolution_metrics`, which returns aggregated (redacted) customer questions. Keep it off on any MCP server a customer-facing agent uses; the REST metrics routes are API-key gated and unaffected |

## Observability

| Variable | Default | Meaning |
|---|---|---|
| `LANGFUSE_SECRET_KEY` | blank (secret) | Blank disables tracing |
| `LANGFUSE_PUBLIC_KEY` | blank (secret) | |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | In `offline` mode a non-local host is refused |
| `ALERTING_ENABLED` | `false` | Log LLM circuit-breaker transitions (CRITICAL when it opens) |
| `LOG_LEVEL` | `INFO` | Python logging level. The logger redacts configured secrets, JWTs and query-string tokens |

## Server

| Variable | Default | Meaning |
|---|---|---|
| `BIND_HOST` | `127.0.0.1` | Interface for `--mode api` and `--mode mcp --transport http` (`--host` wins). Containers set `0.0.0.0` |
| `PORT` | `8000` | REST + WebSocket (`--port` wins) |
| `MCP_HTTP_PORT` | `8001` | Remote MCP (`--port` wins) |
| `GRADIO_PORT` | `7860` | Gradio UI. The UI binds `0.0.0.0` |

## Remote access security

Applies to REST, WebSocket and MCP over HTTP. stdio MCP is never authenticated. A mode without its
settings makes the server exit with status 2. Behavior: [remote-mcp.md](remote-mcp.md).

| Variable | Default | Meaning |
|---|---|---|
| `AUTH_MODE` | `token` | `token` (static keys), `oauth` (JWT access tokens), `both`, `none` (refused unless bound to loopback with no public hosts) |
| `API_KEYS` | blank (secret) | Comma-separated `name:token`. Tokens at least 32 characters, no duplicates. Name appears in logs and keys the rate limit |
| `OAUTH_ISSUER` | blank | Must equal the tokens' `iss` exactly |
| `OAUTH_RESOURCE_URL` | blank | This server's canonical URL, for example `https://host/mcp`; tokens need it in `aud` |
| `OAUTH_JWKS_URL` | blank | Issuer's JWKS (https unless local) |
| `OAUTH_ALGORITHMS` | `RS256,ES256` | Asymmetric only; `HS*` and `none` are refused |
| `OAUTH_REQUIRED_SCOPES` | blank | Comma-separated; blank requires none |
| `OAUTH_CLOCK_SKEW` | `60` | Seconds of leeway for `exp`/`nbf`/`iat` |
| `OAUTH_JWKS_CACHE_TTL` | `3600` | Seconds a fetched key set is trusted |
| `OAUTH_JWKS_TIMEOUT` | `5.0` | Seconds per JWKS fetch |
| `ALLOWED_ORIGINS` | blank | Comma-separated `scheme://host[:port]`, no wildcards. Requests without `Origin` pass; any other Origin gets 403 |
| `ALLOWED_HOSTS` | blank | Accepted `Host` values. On loopback, `localhost`/`127.0.0.1`/`[::1]` are always accepted and others get 421. On other binds the check runs only when this is set |
| `RATE_LIMIT_REQUESTS` | `120` | Per authenticated key per window; `0` disables |
| `RATE_LIMIT_ANON_REQUESTS` | `30` | Per client IP for open routes and failed sign-ins; `0` disables |
| `RATE_LIMIT_WINDOW` | `60.0` | Window in seconds |

## Side effects set at startup (not settings)

- `LLM_MODE=offline` sets `HF_HUB_OFFLINE` so models load from the local Hugging Face cache only.
- `uvicorn` runs with `log_config=None`, so its access logs go through the redacting logger.
