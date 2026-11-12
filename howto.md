# How to use the Customer Support Assistant

Task-oriented guide. Every command and setting below was checked against the code
(`app.py`, `server.py`, `api/routes.py`, `faq_core/config.py`). Things that have **never been run
live** are marked **Not verified**. See [Known limits](#known-limits) at the end.

Related docs: [README](README.md), [configuration](docs/configuration.md),
[API reference](docs/api-reference.md), [architecture](docs/architecture.md),
[operations](docs/operations.md), [remote MCP](docs/remote-mcp.md).

Contents: [1 Install](#1-install) - [2 Configure a mode](#2-configure-an-llm-mode) -
[3 Run a surface](#3-run-a-surface) - [4 Load content](#4-load-content) -
[5 How an agent calls the tools](#5-how-an-agent-calls-the-tools) -
[6 Escalation webhook](#6-escalation-webhook-and-hmac-verification) -
[7 Metrics and gap report](#7-resolution-metrics-and-the-weekly-gap-report) -
[8 Docker](#8-docker) - [9 Troubleshooting](#9-troubleshooting)

## 1. Install

Requires Python 3.13.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # lite core, no torch
cp .env.example .env                                       # then edit .env; never commit it
```

The lite core runs the MCP, REST and ingest/report modes with the torch-free `hashing` embedder
(`--mode ui` needs Gradio, see below). `requirements-rag.txt` (sentence-transformers, gradio, ragas, datasets) is the
optional heavy stack:

```bash
pip install -r requirements-rag.txt   # only for EMBED_PROVIDER=local, RERANKER=bge, --mode ui
```

Without it, `EMBED_PROVIDER=local` falls back to `hashing` and `RERANKER=bge` to no reranking, each
with a warning in the log. `--mode ui` needs Gradio, which is in the heavy stack only.

Smoke check (no network, no LLM key needed):

```bash
python -c "import app, server"
python -m faq_core.retrieval.benchmark      # retrieval recall@k / MRR on data/sample_faq.json
```

The first real start seeds `data/sample_faq.json` into an empty database (`data/faq.db`) and
indexes it.

## 2. Configure an LLM mode

`.env` is read by `faq_core/config.py` from the **current working directory**. Start the process
from the repo root (or use absolute paths, see the Claude Desktop example). Full variable list:
[docs/configuration.md](docs/configuration.md).

| Mode | Who writes the reply | Needs |
|---|---|---|
| `api` (default) | This server, via an LLM provider | a provider key |
| `client` | The calling agent's own model (for example a Claude or ChatGPT chat app) | no LLM key |
| `offline` | A local model (Ollama, vLLM, llama.cpp, LM Studio) | a local runtime; no remote endpoints |

### `api` mode

```dotenv
LLM_MODE=api
LLM_PROVIDER=anthropic          # anthropic | openai | gemini | groq | openrouter | mistral | cohere | together | custom
ANTHROPIC_API_KEY=<your key>    # anthropic only
# LLM_API_KEY=<your key>        # every other hosted provider
# LLM_MODEL=                    # blank = the preset's default model
```

Hosted providers fail at startup without a key. `custom` needs `LLM_BASE_URL` and `LLM_MODEL` (any
OpenAI-compatible server). **Not verified:** no live LLM call has ever been made to any provider;
the default models for the mistral / cohere / together presets are unconfirmed.

### `client` mode

```dotenv
LLM_MODE=client
```

No LLM settings or keys. `answer_question` is not registered (MCP) and `POST /answer` returns 501.
`search_kb` is the primary tool; `search_faq` returns `generated: false` with an empty `answer`.
To serve a Claude or ChatGPT chat app you also need the HTTP MCP transport behind public HTTPS and
authentication: see [section 3.2](#32-remote-mcp-streamable-http) and
[docs/remote-mcp.md](docs/remote-mcp.md).

### `offline` mode

```dotenv
LLM_MODE=offline
LLM_PROVIDER=ollama             # ollama | vllm | llamacpp | lmstudio
LLM_MODEL=llama3.2              # required: the model your runtime has loaded
# OFFLINE_ALLOWED_HOSTS=gpu-box.lan,10.0.0.5     # extra private hosts, if the runtime is not on localhost
```

Offline defaults (`LLM_PROVIDER=ollama`, `EMBED_PROVIDER=local`, `RERANKER=bge`) apply only when
you have not set those variables. `.env.example` sets `EMBED_PROVIDER=hashing` and `RERANKER=none`
explicitly, which overrides those defaults, so comment those two lines out if you want the local
embedder and reranker. At startup the network guard refuses any non-local endpoint (LLM, embedder,
reranker, Langfuse, escalation webhook, OAuth issuer/JWKS, ingest URLs) and any non-loopback bind:
it prints a JSON error to stderr and exits with status 2. **Not verified:** a real local LLM
runtime, sentence-transformers and bge-m3 have not been run here.

### Authentication for the HTTP surfaces

`--mode api` and `--mode mcp --transport http` **refuse to start (exit 2)** without working
authentication. The default `AUTH_MODE=token` needs at least one key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```dotenv
AUTH_MODE=token
API_KEYS=crm:<generated-token>,widget:<another-generated-token>
```

Tokens must be at least 32 characters. Clients send `Authorization: Bearer <token>`. For OAuth
(`AUTH_MODE=oauth`, needed for ChatGPT) see [docs/remote-mcp.md](docs/remote-mcp.md). stdio MCP and
the Gradio UI do not use these keys.

## 3. Run a surface

| Surface | Command | Default address |
|---|---|---|
| MCP over stdio | `python app.py --mode mcp` | the client launches it |
| MCP over Streamable HTTP | `python app.py --mode mcp --transport http` | `127.0.0.1:8001/mcp` |
| REST + WebSocket | `python app.py --mode api` | `127.0.0.1:8000` |
| Gradio dev UI | `python app.py --mode ui` | `0.0.0.0:7860` |

`--host` and `--port` override `BIND_HOST`, `PORT` and `MCP_HTTP_PORT` for the two HTTP surfaces
only.

### 3.1 MCP over stdio (Claude Desktop, Claude Code)

Stdio is unauthenticated by design: the client launched the process. The server reads `.env` and
`DB_PATH` relative to its working directory, so make the client start it in the repo root.

**Claude Desktop** (`claude_desktop_config.json`): a shell wrapper pins the working directory.

```json
{
  "mcpServers": {
    "customer-support": {
      "command": "/bin/sh",
      "args": [
        "-c",
        "cd /absolute/path/to/CustomerSupport && exec .venv/bin/python app.py --mode mcp"
      ]
    }
  }
}
```

**Claude Code:**

```bash
claude mcp add customer-support -- /bin/sh -c \
  'cd /absolute/path/to/CustomerSupport && exec .venv/bin/python app.py --mode mcp'
```

**Not verified:** these two registrations were written from the clients' documented formats and
were not run against Claude Desktop or Claude Code here. The server itself was exercised over real
stdio in earlier phases.

Tools you will see: `search_kb`, `search_faq`, `add_faq_entry`, `list_faq_categories`,
`get_faq_entry`, `create_escalation`, `get_escalation`, `list_knowledge_gaps`, plus
`answer_question` in `api` and `offline` mode only. The operator tool `get_resolution_metrics` is
registered only with `MCP_OPERATOR_TOOLS=true` (default `false`): it returns aggregated customer
questions, so keep it off on any server a customer-facing agent uses.

### 3.2 Remote MCP (Streamable HTTP)

```bash
python app.py --mode mcp --transport http      # http://127.0.0.1:8001/mcp
```

Same tools as stdio; same authentication as REST. To let Claude or ChatGPT connect you need a
public HTTPS URL, `ALLOWED_HOSTS` set to your public host name, and (for ChatGPT) OAuth. Everything
about tunnels, Claude custom connectors, ChatGPT, OAuth settings and the security checks is in
[docs/remote-mcp.md](docs/remote-mcp.md). **Not verified:** Claude or ChatGPT actually connecting,
a real authorization server, a real tunnel.

### 3.3 REST and WebSocket

```bash
python app.py --mode api
export API_KEY=<one of the tokens in API_KEYS>
```

Every route except `GET /health` needs `Authorization: Bearer <token>`, including `/docs` and
`/openapi.json`.

```bash
# liveness (no auth)
curl -s http://127.0.0.1:8000/health

# retrieval + generated answer (in client mode: generated=false, empty answer)
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "top_k": 3, "session_id": "chat-42"}'

# passages only, every LLM mode
curl -s -X POST http://127.0.0.1:8000/kb/search \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'

# add an FAQ entry
curl -s -X POST http://127.0.0.1:8000/faq \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Do you ship abroad?", "answer": "Yes, to most countries.", "category": "Shipping"}'

# rate an answer (query_id comes from a search result; rating 1-5). An unknown query_id
# returns 404 query_not_found; rating the same query_id again replaces the earlier rating.
curl -s -X POST http://127.0.0.1:8000/feedback \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"query_id": "<query_id>", "rating": 4, "helpful": true, "comment": ""}'

# escalate
curl -s -X POST http://127.0.0.1:8000/escalations \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"query_id": "<query_id>", "reason": "customer asked for a human"}'
```

All routes and fields: [docs/api-reference.md](docs/api-reference.md).

WebSocket `/ws/chat`: the handshake needs the `Authorization` header (browsers cannot set it, so
use a backend). Each message is `{"query": "...", "top_k": 3, "session_id": "optional"}`; each reply
is one search result or an error envelope.

```bash
websocat -H "Authorization: Bearer $API_KEY" ws://127.0.0.1:8000/ws/chat
{"query": "How do I reset my password?"}
```

**Not verified:** the `websocat` invocation was not run here (REST and WebSocket were exercised
under uvicorn in Phase 5 with other clients).

### 3.4 Gradio dev UI

```bash
pip install -r requirements-rag.txt     # Gradio lives only in the heavy stack
python app.py --mode ui                 # http://localhost:7860 (GRADIO_PORT)
```

It is an adapter over the same service with no authentication and binds `0.0.0.0`; use it on a
trusted machine only. **Not verified:** the UI has been written but never run.

## 4. Load content

Ingest is CLI-only on purpose (no MCP or REST route, so an agent cannot make the server fetch
arbitrary URLs).

```bash
python app.py --mode ingest --source ./docs-folder
python app.py --mode ingest --source help.md --source faq.csv --source https://example.com/help
```

| Source | Becomes |
|---|---|
| `.md`, `.markdown` | document, heading-aware chunks (title = first `#` heading) |
| `.txt` | plain-text document |
| `.html`, `.htm` | converted to markdown |
| `.csv` | FAQ entries (header needs `question` and `answer`; `category` optional, default `General`) |
| `.json` | FAQ entries: list of `{question, answer, category}` (the `data/sample_faq.json` format) |
| folder | every supported file below it (hidden files, symlinks, other types skipped and reported) |
| `http(s)://` URL | by response content type; query strings are dropped from stored URIs and logs |

It prints one JSON report per source on stdout (`faqs_added`, `faqs_duplicate`, `documents_added`,
`documents_updated`, `documents_unchanged`, `skipped`). Re-ingesting a document with the same
source URI updates it only if its content changed; FAQ questions that already exist are skipped.
Limits: `INGEST_MAX_BYTES` (5 MB) and `INGEST_URL_TIMEOUT` (15 s). A failed source prints a JSON
error to stderr and exits 1; earlier sources stay ingested.

A running server caches answers for `CACHE_TTL` seconds (default 300) in its own process. After an
ingest from another process, restart the server or wait out the TTL. **Not verified:** ingesting
into the SQLite file while a server has it open was not tested.

Add a single FAQ entry through `POST /faq` (above) or the `add_faq_entry` MCP tool. It is saved
immediately and indexed by a background job, so it can take a moment to become searchable.

## 5. How an agent calls the tools

**Which search tool**

| Tool | Returns | Use when |
|---|---|---|
| `search_kb` | passages (full text), citations, confidence, outcome, handling instructions; never calls the LLM | You (the agent) write the reply. Required in `client` mode. Works in all modes |
| `answer_question` | grounded generated answer plus citations | `api` or `offline` mode, the server should write the reply |
| `search_faq` | like `answer_question`, plus the matched FAQ entries in `sources` | Older contract; in `client` mode `generated` is `false` and `answer` is empty |

**Flow**

1. Call `search_kb` (or `answer_question`) with the customer's question. Pass the **same
   `session_id` for every turn** of one conversation (letters, digits, `._:-`, up to 128 chars).
   It feeds the repeated-rephrase trigger, the escalation packet and the re-contact metric.
2. Read `outcome`:

   | `outcome` | Meaning | What to do |
   |---|---|---|
   | `answer` | confidence at or above `CONFIDENCE_ANSWER_THRESHOLD` (0.55) | Answer from the passages; cite `citation_id` values |
   | `answer_with_caveat` | at or above `CONFIDENCE_CAVEAT_THRESHOLD` (0.30) | Answer, and show `caveat`; offer a human if it does not help |
   | `escalate` | below that, or a trigger fired | Do **not** answer from the passages. Tell the customer a human will help and call `create_escalation` with `query_id`. `escalate` is `true` |

   `reasons` lists why (`low_confidence`, `moderate_confidence`, `frustration`, `pii_detected`,
   `legal_topic`, `medical_topic`, `repeated_rephrase`, `self_check_failed`). In
   `search_kb`, follow `handling_instructions` (in priority order).
3. For `escalate`, the generated result never contains an LLM-written answer: `answer` is a fixed
   hand-off message (empty in `client` mode) and `generated` is `false`.
4. Escalate:

   ```json
   {"query_id": "<query_id>", "session_id": "chat-42", "customer_goal": "Get a refund for order 1234",
    "reason": "no refund policy found", "transcript": [{"role": "customer", "text": "..."}]}
   ```

   At least one of `query_id`, `session_id`, `customer_goal` is required. The result has an
   `escalation_id`, `status` (`open`, or `sent` once a webhook accepted it) and `delivery`. Inspect
   later with `get_escalation`.
5. Report quality with `POST /feedback` (REST) using the `query_id`.

**Confidence is a heuristic.** `confidence.calibrated` is always `false`: the score was tuned on a
24-query benchmark with the hashing embedder and has not been tried with a semantic embedder.
Treat it as a ranking signal, not a probability.

**Privacy.** Customer text is PII-redacted (`PII_REDACTION=on`, default) before retrieval, the LLM,
traces, logs, the query log, feedback and escalation packets. `pii_detected` lists the kinds found.
The knowledge base itself is never redacted.

**Errors** from MCP tools come back with `isError: true` and text
`{"error": {"code": "...", "message": "..."}}`. Common codes: `invalid_request`, `faq_not_found`,
`query_not_found`, `escalation_not_found`, `answer_generation_unavailable`, `llm_unavailable`.

## 6. Escalation webhook and HMAC verification

Default sink is `log`: the packet stays in the `escalations` table and a log line records ids and
reason codes. To forward packets to your own system:

```dotenv
ESCALATION_SINK=webhook
ESCALATION_WEBHOOK_URL=https://helpdesk.example.com/hooks/support
ESCALATION_WEBHOOK_SECRET=<long random string>
# ESCALATION_WEBHOOK_TIMEOUT=5
# ESCALATION_WEBHOOK_MAX_RETRIES=2
```

The URL comes from configuration only; agents cannot set it. Non-local hosts must use TLS. Failed
deliveries (transport errors, 429, 5xx) are retried with backoff; the final result is stored in
`delivery`. In `offline` mode the URL must be local.

Each request is a JSON `POST` `{"event": "escalation.created", "packet": {...}}` with headers
`X-Support-Signature: sha256=<hex>`, `X-Support-Timestamp` (unix seconds), `X-Support-Event:
escalation.created` and `Idempotency-Key` (the escalation id, so you can drop retried duplicates).
The signature is HMAC-SHA256 over `"<timestamp>.<raw body>"` with the secret. Verify against the
**raw bytes** before parsing, and reject old timestamps yourself:

```python
import hashlib
import hmac
import time


def verify(
    secret: bytes, timestamp: str, raw_body: bytes, signature: str, max_age: int = 300
) -> bool:
    if abs(time.time() - int(timestamp)) > max_age:
        return False
    expected = (
        "sha256="
        + hmac.new(secret, timestamp.encode("ascii") + b"." + raw_body, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, signature)
```

The packet (schema `support.escalation/v1`) contains `escalation_id`, `kb_id`, `created_at`,
`customer_goal`, `reasons`, `reason_detail`, `query_id`, `session_id`, `pii_detected`, `sources`,
`attempts` and `transcript`, all redacted. **Not verified:** delivery to a real receiver outside the
local test receiver used in Phase 4. No helpdesk adapters exist (Zendesk, Intercom, etc. are
deferred).

## 7. Resolution metrics and the weekly gap report

Three ways to get the same numbers:

```bash
# MCP tool (only with MCP_OPERATOR_TOOLS=true): get_resolution_metrics {"window_days": 7}
curl -s "http://127.0.0.1:8000/metrics/resolution?window_days=7" -H "Authorization: Bearer $API_KEY"
curl -s "http://127.0.0.1:8000/reports/weekly-gaps?window_days=7&limit=10" -H "Authorization: Bearer $API_KEY"

python app.py --mode report                                  # Markdown on stdout
python app.py --mode report --window-days 14 --limit 20 --output gaps.md   # writes the file, prints a JSON summary
```

`window_days` is 1-365 (default 7); `limit` is 1-100 gaps (default 10). Out-of-range values return
`invalid_request` (CLI: JSON error on stderr, exit 1, checked before anything is built). A malformed
or inconsistent setting makes the CLI print a JSON `config_error` on stderr and exit 2, like the
server modes; the message names the setting, never its value. The report CLI is read-only: it
neither seeds nor re-indexes.

Results are cached per window for `METRICS_CACHE_TTL` seconds (default 60), so polling does not
recompute; the gap clustering runs in a worker thread and does not block other requests. Each
table (`query_log`, `escalations`, `feedback`) is read with at most `METRICS_ROW_LIMIT` rows
(default 20000, newest kept); when a table holds more for the window, `query_scan_truncated`,
`escalation_scan_truncated` or `feedback_scan_truncated` is `true` and the report says the window is
partial.

What you get: queries per outcome, `escalation_rate`, escalation records created (by status), the
48h re-contact proxy, average confidence, negative-feedback rate, the number of knowledge-gap
clusters and the largest ones. The report adds ranked gaps and suggested FAQ topics
(`add_faq_entry`, `expand_existing_content` or `revise_existing_answer`). Clusters escalated only
for policy reasons (PII, legal, medical, frustration) are counted but not suggested as content.
Query text comes from the redacted query log. Rates are fractions 0-1, or `null` when there is
nothing to divide by.

**The 48h re-contact number is a proxy, not a resolution rate.**

- A session counts as re-contacted when, after a query with outcome `answer` or
  `answer_with_caveat`, the **same `session_id`** asks again within 48 hours (inclusive).
- Eligible sessions are those with such a query inside the window whose 48h period has ended, or
  that already asked again. `rate` = re-contacted / eligible.
- Queries without a `session_id` cannot be followed; they are excluded and counted in
  `queries_without_session`.
- A customer who opens a new session is missed. A legitimate unrelated follow-up in the same
  session counts as a re-contact.
- `observation_incomplete_sessions` are sessions with such a query and no re-contact yet whose
  48h period had not ended when the numbers were computed. Their status may still change, so they
  are left out of `eligible_sessions` and the rate (counting them as "not re-contacted" would make a
  fresh window look better than it is).
- Feedback counts one rating per answer (`query_id`): rating it again replaces the earlier rating.
- There is no customer identity and no ticket state, so this cannot say whether a problem was
  actually fixed.

With Langfuse keys configured, each weekly report (`GET /reports/weekly-gaps` or
`--mode report`) also pushes aggregate scores (`support.escalation_rate`,
`support.recontact_rate_48h`, `support.knowledge_gaps`, and others) to the trace layer and emits
`metrics.reported`; only aggregates, never query text. `GET /metrics/resolution` and the MCP tool
are plain reads with no telemetry. **Not verified against a real Langfuse instance.**

For a weekly report, schedule the CLI yourself (cron or a CI job); the server does not schedule it:

```bash
0 8 * * 1  cd /path/to/CustomerSupport && .venv/bin/python app.py --mode report --output /var/reports/gaps-$(date +\%F).md
```

## 8. Docker

```bash
docker compose build                         # lite image (no torch)
INSTALL_RAG=1 docker compose build           # adds sentence-transformers, Gradio, RAGAS
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 INSTALL_RAG=1 docker compose build   # CUDA torch

docker compose up api                        # REST on :8000
INSTALL_RAG=1 docker compose --profile ui up --build ui   # Gradio on 127.0.0.1:7860 only
docker compose --profile remote up mcp-http  # remote MCP on :8001
docker compose run --rm mcp                  # MCP over stdio, interactive (profile "mcp")
```

Notes from `Dockerfile` and `docker-compose.yml`:

- The image runs as a non-root user, entrypoint `python app.py`, default `--mode api`. `data/` is
  mounted from `./data`. Compose runs every service as `${APP_UID:-1000}:${APP_GID:-1000}` (with
  `HOME=/tmp`) so the container can write `faq.db` into the bind-mounted `./data`; if your user is
  not 1000:1000, run `APP_UID=$(id -u) APP_GID=$(id -g) docker compose up ...`.
- Services read `.env` (`env_file`). `api` and `mcp-http` set `BIND_HOST=0.0.0.0`; they still exit 2
  without `API_KEYS` or OAuth settings.
- The image health check probes `http://localhost:8000/health`. If you set `ALLOWED_HOSTS`, include
  `localhost` or the check fails.
- `.dockerignore` excludes `.env` and `*.md`; secrets reach the container only through compose.
- `ui` is opt-in (profile `ui`) and published on `127.0.0.1:7860` only: the Gradio UI has **no
  authentication** (it is not behind the security middleware). It needs the heavy build
  (`INSTALL_RAG=1`, Gradio); the lite image has no Gradio. Its compose healthcheck is disabled.

**Not verified:** the Docker build and compose services have never been run, and CI has never run
(there is no git repository).

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Process exits with status 2, JSON `security_config_error` on stderr | Missing or unsafe auth setting: `AUTH_MODE=token` without `API_KEYS`, a key shorter than 32 characters, duplicate keys, `oauth` without `OAUTH_ISSUER`/`OAUTH_RESOURCE_URL`/`OAUTH_JWKS_URL`, HS/`none` algorithms, `AUTH_MODE=none` on a non-loopback bind. Fix `.env` |
| Exit 2, JSON `network_policy_violation` | `LLM_MODE=offline` meets a non-local endpoint (LLM, embedder, reranker, Langfuse host, webhook, OAuth URLs) or a non-loopback bind. Use local hosts or list private hosts in `OFFLINE_ALLOWED_HOSTS` |
| Python traceback at start (`ValueError`, not caught by `app.py`, so not exit 2) | Invalid LLM/retrieval/escalation settings: missing provider key, unknown `LLM_PROVIDER`, `ESCALATION_SINK=webhook` without a URL, local runtime without `LLM_MODEL`. Fix `.env` |
| `--mode api` refuses to start on a fresh checkout | The real `.env` may have no `API_KEYS`. Add one (section 2) |
| 401 on REST/MCP HTTP | No `Authorization: Bearer ...` header, or a wrong token. The body is always the same generic text; the reason is only in the server log. Tokens in the URL (`?access_token=`) are rejected with 400 |
| 403 | The request has an `Origin` header not listed in `ALLOWED_ORIGINS` (or a missing OAuth scope: `insufficient_scope`). Non-browser clients should send no `Origin` |
| 421 | `Host` header not allowed (DNS-rebinding guard). On loopback only `localhost`, `127.0.0.1`, `[::1]` pass; set `ALLOWED_HOSTS` to your public host name behind a proxy or tunnel |
| 429 with `Retry-After` | Rate limit: `RATE_LIMIT_REQUESTS` per key, `RATE_LIMIT_ANON_REQUESTS` per IP, per `RATE_LIMIT_WINDOW` seconds. Set 0 to disable |
| 503 on OAuth requests | The JWKS could not be fetched; the server fails closed. Check `OAUTH_JWKS_URL` |
| 422 from REST | Invalid body: `top_k` must be 1-20, `rating` 1-5, strings non-blank, `session_id` matching `[A-Za-z0-9._:-]{1,128}`. Request-validation errors use FastAPI's `{"detail": [...]}`; service errors use `{"error": {...}}` |
| 501 / `answer_generation_unavailable` | `LLM_MODE=client`: no LLM here. Call `search_kb` and write the reply yourself |
| `llm_unavailable` (502) | The provider failed, refused, or the circuit breaker is open. Check the key, model and network; see logs |
| `answer_question` not in the tool list | Expected in `client` mode |
| Log warning "sentence-transformers is not installed; falling back to the hashing embedder" (or "reranking disabled") | `EMBED_PROVIDER=local` / `RERANKER=bge` without sentence-transformers. Install `requirements-rag.txt`, or set `EMBED_PROVIDER=hashing`. The real `.env` currently has `EMBED_PROVIDER=local`, so it silently uses hashing |
| Log warning "dense retrieval disabled: sqlite-vec is not loaded; using BM25 only" | `sqlite-vec` extension failed to load (some Python builds disable SQLite extension loading). Retrieval continues with BM25 only. Reinstall `sqlite-vec` or use a Python build with extension support, or set `VECTOR_STORE=none` to silence it |
| New FAQ entry not found yet | Indexing is a background job. Retry shortly; a restart re-syncs the index from the database |
| Old answer after content change | Answer cache (`CACHE_TTL`, per process). Cleared on add, ingest, sync; restart to force |
| Everything re-embeds at startup | You changed the embedder name or dimension; a full re-embed is automatic |
| Out-of-KB questions get a caveat, not an escalation | Known with the hashing embedder when the question shares words with the KB. See [tuning](docs/operations.md#tuning-the-confidence-thresholds) |
| `--mode ingest` exits 1 | A source could not be read, fetched or parsed; the JSON error on stderr names it |
| `--mode ui` ImportError | Gradio is not installed; install `requirements-rag.txt` |
| `--mode ingest` / `report` error "needs ..." | `--mode ingest` requires at least one `--source`; report options (`--window-days`, `--limit`, `--output`) apply to `--mode report` only; `--host`/`--port` only to `api` and `mcp --transport http` |

## Known limits

- No live LLM call has ever been made; `ANTHROPIC_API_KEY` in the checked-in `.env` is a placeholder.
- Confidence is uncalibrated and was tuned with the hashing embedder only.
- Not run: sentence-transformers, bge-m3, BGE reranker, Cohere reranker, Gradio, Docker, CI,
  Claude or ChatGPT connecting, a real authorization server, a real Langfuse project.
- No opaque-token (RFC 7662) support; rate limits and the JWKS cache are per process.
- English-only default trigger term lists; set `ESCALATION_*_TERMS` for other languages.

Evaluation (retrieval benchmark, resolution metrics, RAGAS gate) is documented in [`docs/evaluation.md`](docs/evaluation.md).
