# API reference

MCP tools (`server.py`) and REST/WebSocket routes (`api/routes.py`), with the result models from
`faq_core/models.py`. Field names are a stable contract: fields are added, never renamed or removed.
Back to the [README](../README.md). Examples: [howto.md](../howto.md).

All results are JSON. Timestamps are ISO-8601 UTC. Ids are UUID strings. Rates are fractions 0-1.

## Common objects

**Citation** (`citations[]`): `id` (chunk UUID), `source_type` (`faq` or `document`), `source_id`,
`title`, `heading` (`""` for FAQ entries), `snippet`, `source_uri` (`""` for FAQ entries),
`updated_at`, `score` (0.0-1.0).

**Passage** (`passages[]`, `search_kb` only): `rank` (1 = best), `citation_id` (cite this),
`source_type`, `source_id`, `title`, `heading`, `text` (the whole passage), `source_uri`,
`updated_at`, `score`.

**FAQEntry**: `id`, `question`, `answer`, `category`, `kb_id`, `created_at`.

**ConfidenceSignal**: `score` (0.0-1.0, 0.0 when nothing was found), `level` (`high` | `medium` |
`low`), `basis` (`retrieval_signals` or `retrieval_signals+llm_self_check`), `calibrated` (always
`false`), `signals` (component scores that were available: for example term coverage, similarity,
agreement, margin, rerank).

**Outcome fields** on every search result:

| Field | Meaning |
|---|---|
| `outcome` | `answer`, `answer_with_caveat`, `escalate` |
| `escalate` | `true` exactly when `outcome` is `escalate` |
| `reasons` | Reason codes: `low_confidence`, `moderate_confidence`, `repeated_rephrase`, `frustration`, `pii_detected`, `legal_topic`, `medical_topic`, `self_check_failed` |
| `caveat` | Text to show with a medium-confidence answer, otherwise `""` |
| `pii_detected` | Kinds of personal data found (and redacted) in the query |
| `session_id` | Echo of the input |

**Error envelope** (MCP tool errors and REST service errors): `{"error": {"code": "...",
"message": "...", ...details}}`.

| Code | REST status | Meaning |
|---|---|---|
| `invalid_request` | 422 | Bad input (`fields` lists the problems for argument validation) |
| `faq_not_found` | 404 | Unknown FAQ id (`faq_id`) |
| `query_not_found` | 404 | `create_escalation` or `POST /feedback` with a `query_id` that was never logged |
| `escalation_not_found` | 404 | Unknown escalation id |
| `answer_generation_unavailable` | 501 | `answer_question` / `POST /answer` in `LLM_MODE=client` (`use_instead: search_kb`) |
| `llm_unavailable` | 502 | Provider failed, refused, or the circuit breaker is open |
| `unknown_tool` | n/a | MCP: tool name does not exist |

REST request bodies that fail pydantic validation (wrong types, `top_k` outside 1-20, unknown
fields) use FastAPI's default 422 body `{"detail": [...]}`. Request models forbid extra fields.

## MCP tools

Over stdio (unauthenticated) and Streamable HTTP (`/mcp`, authenticated). Tool names and results
are the same on both. Failed calls return `isError: true` with the error envelope as text. Every tool
returns a pydantic model, so MCP publishes an `outputSchema` and each result carries
`structuredContent` plus the same JSON as text.

| Tool | Modes | Arguments | Result |
|---|---|---|---|
| `search_kb` | all | `query` (str), `top_k` (int, 1-20, default 3), `session_id` (str, optional) | KBSearchResult |
| `answer_question` | `api`, `offline` only | same | SearchResult |
| `search_faq` | all | same | SearchResult (`generated: false`, empty `answer` in `client` mode) |
| `add_faq_entry` | all | `question`, `answer`, `category` (non-blank strings) | FAQEntry |
| `list_faq_categories` | all | none | `{kb_id, categories[]}` |
| `get_faq_entry` | all | `faq_id` | FAQEntry |
| `create_escalation` | all | `query_id`, `session_id`, `customer_goal` (<= 1000), `reason` (<= 500), `transcript` (<= 50 turns `{role, text}`; role `customer`, `assistant`, `agent`, `system`; text <= 4000) | EscalationResult |
| `get_escalation` | all | `escalation_id` | EscalationResult |
| `list_knowledge_gaps` | all | `limit` (1-100, default 20), `min_count` (>= 1, default 1), `since` (ISO-8601, UTC if no offset; default 30 days ago) | KnowledgeGapList |
| `get_resolution_metrics` | all, **only with `MCP_OPERATOR_TOOLS=true`** | `window_days` (1-365, default 7) | ResolutionMetrics |

`answer_question` is not registered in `client` mode (an agent never sees a tool it cannot use);
calling it anyway returns `answer_generation_unavailable`. The server also publishes `instructions`
text to the client, which differs between generating modes and `client` mode.

`get_resolution_metrics` is an operator tool: its `top_gaps` hold other customers' (redacted)
questions, so it is registered only when `MCP_OPERATOR_TOOLS=true` (default `false`). Keep it off
on any server a customer-facing agent talks to; calling it when it is not registered returns
`unknown_tool`. `list_knowledge_gaps` is unchanged (always registered).

`create_escalation` needs at least one of `query_id`, `session_id`, `customer_goal`. Where the
packet goes (sink, webhook URL) is server configuration, never a tool argument.

### KBSearchResult (`search_kb`)

`query_id`, `query` (redacted), `kb_id`, `llm_mode`, `passages[]`, `citations[]`, `confidence`
(ConfidenceSignal), `has_passages`, `handling_instructions[]` (rules the answering model must
follow, in priority order), plus the outcome fields. Never calls an LLM.

### SearchResult (`answer_question`, `search_faq`, `POST /search`, `POST /answer`)

`query_id`, `query`, `answer`, `confidence` (float, same as `confidence_detail.score`), `sources[]`
(FAQEntry items among the hits; document hits appear only in `citations`), `cached`, `citations[]`,
`generated`, `confidence_detail` (ConfidenceSignal), plus the outcome fields. When `escalate` is
true, `generated` is `false` and `answer` is the fixed hand-off message (empty in `client` mode).

### EscalationResult

`escalation_id`, `kb_id`, `status` (`open`, `sent`, `resolved`; `sent` once a forwarding sink
accepted it), `query_id`, `session_id`, `reasons[]`, `packet` (redacted, schema
`support.escalation/v1`), `delivery` (`{sink, delivered, attempts, error, delivered_at}`;
`error` never holds secrets), `created_at`.

Packet fields: `schema`, `escalation_id`, `kb_id`, `created_at`, `customer_goal`, `reasons`,
`reason_detail`, `query_id`, `session_id`, `pii_detected`, `sources[]`, `attempts[]`
(`query_id`, `query`, `outcome`, `confidence`, `reasons`, `at`), `transcript[]`.

### KnowledgeGapList

`kb_id`, `since`, `scanned` (gap-log rows considered), `gaps[]`. Each gap: `gap_id` (stable hash of
the representative query), `representative_query`, `count`, `example_queries[]` (up to 5, redacted),
`outcomes` (counts by outcome), `reasons` (counts by reason), `avg_confidence`, `first_seen`,
`last_seen`. Ordered by `count` then `last_seen`.

### ResolutionMetrics

Over the half-open window `[since, until)`:

| Field | Meaning |
|---|---|
| `kb_id`, `window_days`, `since`, `until` | scope |
| `total_queries` | logged queries in the window |
| `outcomes` | `{answer, answer_with_caveat, escalate}` counts |
| `escalation_rate` | `escalate` outcomes / `total_queries`, or `null` |
| `escalations_created`, `escalations_by_status` | hand-off records created in the window |
| `recontact` | 48h re-contact **proxy**: `horizon_hours` (48), `definition` (text), `sessions`, `eligible_sessions` (sessions with an answer in the window whose 48h period has ended or that already asked again), `recontacted_sessions`, `rate` = recontacted / eligible (or `null`), `observation_incomplete_sessions` (48h period still open, not yet re-contacted; not in the rate), `queries_without_session` |
| `avg_confidence`, `confidence_samples`, `confidence_calibrated` | mean stored score (heuristic; `confidence_calibrated` is `false`) |
| `feedback_count`, `negative_feedback`, `negative_feedback_rate` | one (the latest) rating per answer; negative means rating <= 2 or not helpful |
| `knowledge_gaps`, `gap_queries_scanned`, `gap_scan_truncated`, `top_gaps[]` | gap clusters; truncated when more than `GAP_SCAN_LIMIT` rows existed |
| `query_scan_truncated`, `escalation_scan_truncated`, `feedback_scan_truncated` | `true` when that table held more than `METRICS_ROW_LIMIT` rows for the window; only the newest rows were counted (added after the Phase 6 review; default `false`) |

Results are cached per window for `METRICS_CACHE_TTL` seconds (default 60). `GET
/metrics/resolution` and the MCP tool have no side effects; the weekly report (REST or CLI) is what
pushes Langfuse scores and emits `metrics.reported`.

What the re-contact number is and is not: [howto.md section 7](../howto.md#7-resolution-metrics-and-the-weekly-gap-report).

### WeeklyGapReport

`kb_id`, `generated_at`, `window_days`, `since`, `until`, `metrics` (ResolutionMetrics), `gaps[]`,
`suggested_topics[]` (`gap_id`, `topic`, `queries`, `action` one of `add_faq_entry`,
`expand_existing_content`, `revise_existing_answer`, `example_queries`, `top_reasons`),
`policy_only_gaps`, `markdown`.

## REST and WebSocket

`python app.py --mode api` serves these on `BIND_HOST:PORT` (default `127.0.0.1:8000`).

**Auth:** every route requires `Authorization: Bearer <token>` (static key or OAuth JWT, per
`AUTH_MODE`) **except** `GET /health` (and, in OAuth modes, the `/.well-known/oauth-protected-resource*`
metadata). That includes `/docs` and `/openapi.json`. Responses before the route runs: 401 (missing
or invalid credentials, always the same body for an invalid token), 403 (bad Origin or missing
scope), 421 (bad Host), 400 (token in the URL), 429 with `Retry-After`, 503 (OAuth key set
unreachable). See [remote-mcp.md](remote-mcp.md).

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| GET | `/health` | none | | `{status: "ok", kb_id, llm_mode, answer_generation}` |
| POST | `/search` | yes | `{query, top_k?=3, session_id?}` | SearchResult |
| POST | `/kb/search` | yes | same | KBSearchResult |
| POST | `/answer` | yes | same | SearchResult; 501 `answer_generation_unavailable` in `client` mode |
| GET | `/faq/{faq_id}` | yes | | FAQEntry; 404 `faq_not_found` |
| POST | `/faq` | yes | `{question, answer, category}` | 201 FAQEntry |
| GET | `/categories` | yes | | `{kb_id, categories[]}` |
| POST | `/feedback` | yes | `{query_id (UUID), rating (1-5), comment?="", helpful?=true}` | 201 `{status: "recorded", feedback_id}`; 404 `query_not_found` |
| POST | `/escalations` | yes | `{query_id?, session_id?, customer_goal?, reason?, transcript?}` | 201 EscalationResult |
| GET | `/escalations/{escalation_id}` | yes | | EscalationResult; 404 |
| GET | `/knowledge-gaps` | yes | query: `limit`, `min_count`, `since` | KnowledgeGapList |
| GET | `/metrics/resolution` | yes | query: `window_days` | ResolutionMetrics |
| GET | `/reports/weekly-gaps` | yes | query: `window_days`, `limit` | WeeklyGapReport |
| WS | `/ws/chat` | handshake header | messages `{query, top_k?=3, session_id?}` | one SearchResult or error envelope per message |

Notes:

- `query` must be non-blank; `top_k` is 1-20.
- `POST /feedback` needs a `query_id` that was logged in this knowledge base; an unknown one returns
  404 `query_not_found` (the same envelope as `POST /escalations`) and stores nothing. Each answer
  keeps one rating: posting again for the same `query_id` replaces the earlier rating, comment and
  timestamp (the receipt's `feedback_id` is the new row's id). A rating of 2 or less, or
  `helpful: false`, adds the query to the knowledge-gap log.
- Query-parameter ranges for the GET routes are validated by the service and return 422
  `invalid_request`, not FastAPI's `detail` body.
- `/ws/chat` calls `search` (not `search_kb`). A bad message gets an error envelope and the socket
  stays open. The handshake needs the `Authorization` header, so browsers cannot connect directly.
  Each message counts toward the key's rate limit; over the limit the socket closes with code 1013.
  A refused handshake is answered 403 (close 1008).
- Service errors use the envelope table above.

### Not available over any network surface

Ingest (`python app.py --mode ingest`) is command-line only. There is no route or tool that fetches
a URL or reads a file on behalf of a caller. The weekly report is available both as
`GET /reports/weekly-gaps` and as `python app.py --mode report`; there is no MCP tool for the full
report (use `get_resolution_metrics`, if `MCP_OPERATOR_TOOLS=true`, and `list_knowledge_gaps`).
