# Operations

Running the Customer Support Assistant safely: security checklist, limits, backups, monitoring,
tuning, re-indexing, upgrades and known limits. Back to the [README](../README.md). Settings are in
[configuration.md](configuration.md).

Nothing here has been run in a production deployment. Items marked **Not verified** have no live
test behind them.

## Security checklist

Before exposing an HTTP surface (`--mode api`, `--mode mcp --transport http`):

- [ ] Auth is on. Use `AUTH_MODE=token` with strong keys, or `oauth` with a real authorization
      server. Never run `AUTH_MODE=none` behind a proxy or tunnel (the server refuses it when
      `ALLOWED_HOSTS` or `OAUTH_RESOURCE_URL` name a public host, but it cannot detect a tunnel that
      rewrites `Host` to `localhost`).
- [ ] Keys are at least 32 URL-safe characters, generated with
      `python -c "import secrets; print(secrets.token_urlsafe(32))"`, one per client
      (`API_KEYS=name:token,...`). Rotate by editing `API_KEYS` and restarting.
- [ ] TLS is terminated by a reverse proxy or tunnel in front of a loopback bind. The server itself
      serves plain HTTP.
- [ ] `ALLOWED_HOSTS` is set to your public host name. Without it, on a non-loopback bind the Host
      header is not checked (a warning is logged at start).
- [ ] `ALLOWED_ORIGINS` lists only the browser origins you trust, if any. Requests with no `Origin`
      header (agents, curl) pass; any other Origin gets 403.
- [ ] `.env` is not in version control or in a Docker image (`.dockerignore` excludes it) and is
      readable only by the service account.
- [ ] `PII_REDACTION=on` (default). Customer text is redacted before the LLM, traces, logs, the query
      log, feedback and escalation packets. The KB is never redacted: do not put personal data in it.
- [ ] If `ESCALATION_SINK=webhook`: the receiver is on TLS (non-local hosts require it), verifies the
      HMAC signature and the timestamp, and uses `Idempotency-Key` to drop duplicates. See
      [howto.md section 6](../howto.md#6-escalation-webhook-and-hmac-verification).
- [ ] The Gradio UI (`--mode ui`) has no authentication and binds `0.0.0.0`. Do not expose it. In
      compose it is opt-in (profile `ui`) and published on `127.0.0.1:7860` only.
- [ ] `MCP_OPERATOR_TOOLS=false` (default) on every MCP server a customer-facing agent uses. When
      true, `get_resolution_metrics` returns other customers' redacted questions to any MCP caller.
- [ ] `/docs` and `/openapi.json` are protected like every other route; do not add routes outside
      `SecurityMiddleware`.

What the middleware does, in order: Host check (421), Origin check (403), authentication (401/403/
503), per-key rate limit (429). A token in the URL gets 400. The `Authorization` header is removed
before any route, tool or SDK code runs, so tokens are never forwarded upstream. Invalid-token
responses are always identical; the reason goes to the server log only. Full detail:
[remote-mcp.md](remote-mcp.md).

Ingest is CLI-only so an agent cannot make the server fetch arbitrary URLs. In `offline` mode the
network guard also refuses non-local ingest URLs.

## Rate limits

| Setting | Default | Applies to |
|---|---|---|
| `RATE_LIMIT_REQUESTS` | 120 | per authenticated key, per window |
| `RATE_LIMIT_ANON_REQUESTS` | 30 | per client IP: open routes (`/health`, metadata) and failed sign-ins |
| `RATE_LIMIT_WINDOW` | 60 s | the window for both |

`0` disables a limit. Over the limit the response is 429 with `Retry-After`. Each WebSocket message
counts against the key's limit; over the limit the socket closes with code 1013.

Limits are in memory and per process: with several workers each keeps its own counts. The per-IP
limit uses the address uvicorn derives from `X-Forwarded-For`, which uvicorn trusts only from
`127.0.0.1`. A proxy that passes a client-supplied header through unchanged would let clients rotate
their per-IP key; per-key limits are unaffected.

## Backups

All state is in the SQLite file `DB_PATH` (default `./data/faq.db`): FAQ entries, documents, chunks,
the FTS5 index, the dense vectors, the query log, feedback and escalations. The scheduler has its own
file only if `SCHEDULER_DB_URL` is set; it holds queued re-index jobs and is safe to lose.

The repository opens file databases in WAL mode (`PRAGMA journal_mode=WAL`) with
`PRAGMA busy_timeout=5000`, so the API, MCP and CLI processes can share the file: readers do not
block the writer, and a writer waits up to 5 s for a lock instead of failing at once. WAL keeps
recent writes in `faq.db-wal` (with `faq.db-shm`) until a checkpoint, so copying `faq.db` alone with
`cp` can miss committed data or produce a torn copy. Always use SQLite's online backup, which
includes the WAL contents:

```bash
sqlite3 data/faq.db ".backup 'backup/faq-$(date +%F).db'"
# or:  sqlite3 data/faq.db "VACUUM INTO 'backup/faq-$(date +%F).db'"
```

Restore by stopping every process that uses the database, deleting any `faq.db-wal` / `faq.db-shm`
next to it, and replacing the file. WAL needs the database on a local filesystem (not NFS or another
network share). The index can be rebuilt from the tables at
any start (`sync_index`), so a restore that drops `chunk_vec` or `chunks_fts` rows self-heals; the
`faq_entries`, `documents`, `query_log`, `feedback` and `escalations` tables are the data to protect.

Retention: nothing prunes `query_log`, `feedback` or `escalations`. They grow without bound. Query
text is stored redacted, but treat the database as sensitive. Plan your own retention (for example a
periodic `DELETE` of old `query_log` rows) and keep the metrics window in mind: deleting rows removes
them from future reports. **Not verified:** a restore drill, and backups while the server is under
load.

## Monitoring

- **Health:** `GET /health` (no auth) returns `status`, `kb_id`, `llm_mode`, `answer_generation`.
  It only proves the process is up, not that the LLM or the index work. The Docker health check
  probes `http://localhost:8000/health` (REST image only).
- **Logs:** structured, through a redacting logger that masks configured secrets, JWTs and
  query-string tokens. uvicorn access logs go through it too. Set `LOG_LEVEL`. Look for warnings at
  start: hashing fallback, `sqlite-vec` not loaded (BM25 only), rate-limited requests, refused
  requests, `ALLOWED_HOSTS` not set.
- **Langfuse:** set `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` to trace retrieval
  and LLM calls (`@observe` wrappers). Leave the keys blank to disable. The weekly report
  (`python app.py --mode report` or `GET /reports/weekly-gaps`) pushes aggregate scores named
  `support.total_queries`, `support.escalation_rate`, `support.escalations_created`,
  `support.recontact_rate_48h`, `support.knowledge_gaps`, `support.avg_confidence` and
  `support.negative_feedback_rate`; scores whose denominator is zero are skipped, and no query text
  is sent. `GET /metrics/resolution` and the MCP tool push nothing (polling them does not create
  traces), so schedule the report. In `offline` mode a non-local Langfuse host is refused. **Not
  verified** against a real Langfuse project.
- **LLM circuit breaker:** after repeated provider failures calls fail fast with `llm_unavailable`
  (502). `ALERTING_ENABLED=true` logs the transitions (CRITICAL when it opens).
- **Weekly review:** run `python app.py --mode report` (or `GET /reports/weekly-gaps`) each week,
  read the suggested topics, add FAQ entries or ingest documents, and watch the escalation rate and
  the re-contact proxy move. The proxy's limits are in
  [howto.md section 7](../howto.md#7-resolution-metrics-and-the-weekly-gap-report).
- **Retrieval quality:** `python -m faq_core.retrieval.benchmark` seeds a throw-away database with
  `data/sample_faq.json` and scores `data/retrieval_benchmark.json` (recall@k and MRR for `hybrid`,
  `bm25`, `dense`). Options: `--cases FILE`, `--k 1 3 5`, `--modes hybrid bm25 dense`, and `--env`
  to use the embedder and reranker from `.env`. A test enforces recall and MRR floors.
  Answer-quality evaluation is covered in [evaluation.md](evaluation.md).

## Tuning the confidence thresholds

The score is a heuristic: a weighted mean of query-term coverage (0.4), dense similarity (0.3),
BM25/dense agreement (0.2), margin over the runner-up (0.1), plus a reranker score (0.3) when present.
It is **not** a probability (`calibrated: false`).

| Setting | Effect of raising it |
|---|---|
| `CONFIDENCE_ANSWER_THRESHOLD` (0.55) | fewer plain answers, more caveats |
| `CONFIDENCE_CAVEAT_THRESHOLD` (0.30) | fewer caveated answers, more escalations |
| `CONFIDENCE_SIMILARITY_FLOOR` / `_CEILING` (0.05 / 0.35) | how dense cosine maps onto 0-1; semantic embedders produce a different range |

Procedure:

1. Collect labeled questions, including ones the KB cannot answer (the `out_of_kb` key in
   `data/retrieval_benchmark.json` is the existing example).
2. Look at `confidence.score` and `outcome` for each through `search_kb`.
3. Move the thresholds so in-KB questions land at `answer` and out-of-KB ones at `escalate`.
4. Re-run `pytest tests/test_confidence.py` and the retrieval benchmark.
5. After switching `EMBED_PROVIDER` (for example to `local`), repeat from step 1; the shipped values
   were tuned only against the 24-query benchmark with the hashing embedder.

Known behavior with the hashing embedder: out-of-KB questions that share words with the KB (for
example "change my username") get a caveat instead of an escalation. Not verified: thresholds with a
semantic embedder, and the optional LLM self-check live (`CONFIDENCE_SELF_CHECK`, one extra LLM call
per answer, can only lower the outcome).

Escalation triggers have their own settings (`ESCALATION_TRIGGERS`, `ESCALATION_*_TERMS`,
`ESCALATION_REPHRASE_*`). The default term lists are English only.

## Re-index behavior

- **At every start** the server seeds `data/sample_faq.json` if the database is empty, then always
  runs `sync_index()`: sources with no or stale chunks are re-chunked, index rows of sources that no
  longer exist are dropped, missing FTS and vector rows are repaired. So the index is rebuilt from
  the database after every restart.
- **`add_faq_entry` / `POST /faq`** saves the entry at once, clears the answer cache, and emits
  `faq.added`. A background job indexes it; when done, `faq.indexed` clears the cache again. With the
  default in-memory queue, a job lost in a restart is repaired by the next start's `sync_index()`.
  Set `SCHEDULER_DB_URL` for a persistent queue. `REINDEX_CONCURRENCY` bounds parallel jobs.
- **Ingest** indexes directly (no `faq.added` events). Documents are keyed by `source_uri`; unchanged
  ones are skipped, changed ones are updated and re-indexed. Existing FAQ questions are skipped.
- **Embedder change** (name or dimension): the dense vectors are dropped and re-embedded at the next
  start. With a large KB and a heavy model this takes time; the server is not ready until it finishes.
- The answer cache is per process and cleared on add, ingest, seed and sync. Content changed by
  another process (for example the ingest CLI against a database a server has open) shows up in
  that server only after `CACHE_TTL` expires or a restart. **Not verified:** concurrent writers on
  one SQLite file.

## Upgrades and migrations

- Schema changes are versioned SQL in `faq_core/migrations.py` (currently versions 1 to 6). They run
  automatically at start, each in its own transaction with its bookkeeping row; a failed migration
  rolls back and the start fails. Released migrations are never edited, new ones are appended.
- Back up the database before upgrading (see Backups). There are no down-migrations; roll back by
  restoring the backup and the previous code.
- The `chunk_vec` table is created at runtime, not by a migration.
- Migration 6 adds `(kb_id, created_at)` indexes on `feedback` and `escalations` and a unique
  `(kb_id, query_id)` index on `feedback`. Before creating it, it **deletes duplicate ratings**,
  keeping only the latest one per answer (by `created_at`, then insertion order). Back up first if
  you need the older ratings.
- Dependency pins to respect: `mcp>=1.10,<2` (2.x removed `FastMCP`), `pyjwt[crypto]>=2.10,<3`,
  `apscheduler==4.0.0a6` (3.x API is incompatible), `sqlalchemy[asyncio]`, `sqlite-vec>=0.1.6,<0.2`.
- Tool and route field names are a stable contract (additive changes only). Breaking changes so far
  are listed in `CLAUDE.md`: Phase 1 removed REST `/chat`; Phase 5 made auth mandatory on every REST
  route except `/health` and made `--mode api` bind `127.0.0.1` by default.

## Known limits

- No live LLM call has ever been made to any provider. The `ANTHROPIC_API_KEY` in the checked-in
  `.env` is a placeholder; preset defaults for mistral, cohere, together and the vLLM port are
  unconfirmed.
- Confidence is uncalibrated and tuned on a 24-query benchmark with the hashing embedder. Dense KNN
  has no similarity floor.
- Not run: sentence-transformers, bge-m3, the BGE reranker, a live Cohere reranker, Gradio, RAGAS,
  the Docker build and compose services, CI (no git repository), Claude or ChatGPT connecting, a real
  authorization server, a tunnel, a real Langfuse project.
- No RFC 7662 token introspection: only JWT access tokens work in OAuth mode.
- Rate limits, the JWKS cache and the answer cache are per process. No per-tool scopes. No ChatGPT
  mixed auth.
- The per-IP limit depends on the proxy appending `X-Forwarded-For`. On a non-loopback bind the Host
  header is checked only when `ALLOWED_HOSTS` is set.
- Browsers cannot connect to `/ws/chat` (no `Authorization` header on a browser WebSocket).
- Helpdesk adapters (Zendesk, Intercom, email) do not exist; the sinks are `log` and `webhook`.
- Single knowledge base (`kb_id` column kept for later). No retention policy for the query log.
- Trigger term lists are English only; language is not detected, prompts ask the model to reply in
  the customer's language.
- The 48h re-contact figure is a proxy based on `session_id`, not a measured resolution rate.
- `faq_core/service.py` is about 960 lines (guideline ~700); escalation, gap and metrics methods are
  candidates to move into their own modules (deferred; see the Resolution section of
  `docs/code-review.md`).
- Metrics reads are capped at `METRICS_ROW_LIMIT` rows per table (newest kept); a capped window is
  flagged (`*_scan_truncated`) rather than computed exactly.
