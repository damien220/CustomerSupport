# Evaluation

Three independent checks measure quality. Only the first two run in the lite test suite;
the third needs a live LLM and has **never been run**.

| Check | Measures | Needs | Run it |
|---|---|---|---|
| Retrieval benchmark | Does search find the right FAQ? (recall@k, MRR) | nothing (hashing embedder) | `python -m faq_core.retrieval.benchmark` |
| Resolution metrics | How is the live service doing? (outcomes, escalation rate, 48h re-contact proxy, gaps) | queries in `query_log` | `python app.py --mode report` |
| RAGAS gate | Are generated answers grounded, relevant and precise? | live LLM key + `requirements-rag.txt` + a semantic embedder | `pytest tests/test_ragas.py tests/test_ragas_eval.py -m rag -v -rs` |

## 1. Retrieval benchmark

`data/retrieval_benchmark.json` holds 24 labeled queries (each names the FAQ it should find) plus 13
out-of-KB queries used to tune the escalation thresholds, over `data/sample_faq.json`. The module seeds a throw-away SQLite database, indexes it with the real
retrieval stack and reports `hybrid`, `bm25` and `dense` modes:

```bash
python -m faq_core.retrieval.benchmark          # lite defaults, ignores .env
python -m faq_core.retrieval.benchmark --env    # benchmark the embedder/reranker from .env
```

Measured with the hashing embedder (Phase 2): hybrid recall@1/3/5 = 0.875 / 0.958 / 1.0, MRR 0.920.
`tests/test_retrieval_benchmark.py` fails if the hybrid scores fall below a floor slightly under
these numbers, so a regression in chunking, fusion or ranking is caught offline.

Run it with `--env` after switching `EMBED_PROVIDER`, then re-tune the confidence settings
(see [operations.md](operations.md)). Add your own labeled queries to the JSON file as the
knowledge base grows; a query's `relevant` list names FAQ questions or document titles.

## 2. Resolution metrics and the weekly gap report

Computed from `query_log`, `escalations` and `feedback` over a window (default 7 days).
Surfaces: MCP `get_resolution_metrics` (only with `MCP_OPERATOR_TOOLS=true`), REST `GET /metrics/resolution`,
`GET /reports/weekly-gaps`, CLI `python app.py --mode report` (see [api-reference.md](api-reference.md)
and the how-to).

What each number means:

- **Outcome counts / escalation rate**: share of queries whose outcome was `escalate`.
- **Escalation hand-offs created**: rows in `escalations`; a different number, because a human
  hand-off is opt-in via `create_escalation`.
- **48h re-contact rate (a proxy, not a resolution rate)**: of the sessions that received an
  `answer` or `answer_with_caveat`, the share that asked again in the same session within 48 hours.
  Limits: it needs `session_id` (queries without one are excluded and counted), an unrelated
  follow-up counts as a re-contact, and a customer who returns in a new session is missed.
  Sessions whose 48 hours have not passed yet (and that have not asked again) are left out of the
  rate and counted separately (`observation_incomplete_sessions`), so a fresh window is not biased
  towards "resolved".
- **Negative-feedback rate**: one rating per answer (the latest); ratings must name a logged
  `query_id`, so retries or invented ids cannot skew it.
- **Knowledge gaps**: clusters of queries that were escalated, answered with a caveat, or rated
  badly. Each suggested topic says whether to add an FAQ entry, expand content or revise an answer.
  Clusters escalated only for policy reasons (PII, legal, medical, frustration) are counted but not
  suggested as content.
- **Average confidence** is the uncalibrated retrieval score, not a probability of being right.

Rates are `null` when there is nothing to divide by. When Langfuse is configured, each weekly
report (REST or CLI) pushes the numbers as scores under `support.*` names; `GET
/metrics/resolution` and the MCP tool push nothing. Each table is read with at most
`METRICS_ROW_LIMIT` rows (`*_scan_truncated` says when a window was cut), and results are cached for
`METRICS_CACHE_TTL` seconds.

## 3. RAGAS answer-quality gate

Scores generated answers with an LLM judge. All must hold over the three sample questions:

| Metric | Threshold |
|---|---|
| `faithfulness` | ≥ 0.85 |
| `answer_relevancy` | ≥ 0.80 |
| `context_precision` | ≥ 0.75 |

**Status: UNVERIFIED live.** The wiring was import-checked and tested against a mock HTTP
transport in a throwaway environment, but the gate has never run against a real LLM, so the
thresholds have never been compared with real scores. Treat the first live run as calibration, not
as a pass/fail verdict.

Pinned to `ragas==0.4.3` and its *collections* metric API (exact pins and reasons in
`requirements-rag.txt`; the old `evaluate(Dataset, ...)` API is legacy in 0.4 and removed in 1.0).

### How it works

1. `build_runtime` + `serve` start the production service on a temporary database seeded from
   `data/sample_faq.json`.
2. For each sample, `SupportService.answer` produces the response and `search_kb` the passages it
   was grounded on.
3. The judge LLM is the project's configured LLM (`LLM_PROVIDER` preset and key settings). Claude
   models go through the Anthropic SDK; other providers, including local runtimes, through an
   OpenAI-compatible client. Plain `http` is refused unless the host is local.
4. Answer relevancy uses the retrieval index's own embedder; the lexical hashing embedder is
   refused, because it cannot compare meaning.
5. Scores are averaged per metric, compared with the thresholds, and pushed to Langfuse when it is
   configured. RAGAS's own usage telemetry is switched off (`RAGAS_DO_NOT_TRACK=true`).

### Running it live

```bash
source .venv/bin/activate
pip install -r requirements-rag.txt        # large: pulls sentence-transformers / torch
# .env:
#   LLM_MODE=api   LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=...   (or another provider + LLM_API_KEY)
#   EMBED_PROVIDER=local                    (a semantic embedder; openai / ollama also work)
pytest tests/test_ragas.py tests/test_ragas_eval.py -m rag -v -rs
```

One run makes real, billed calls: 3 answers plus about 24 judge requests. In CI it is the manual
`workflow_dispatch` job and needs the `ANTHROPIC_API_KEY` repository secret; the job installs CPU
torch and the RAG stack and sets `LLM_MODE=api`, `LLM_PROVIDER=anthropic`, `EMBED_PROVIDER=local`
and `RAGAS_REQUIRE_LIVE=1`.

Locally the test **skips with a reason** (it does not fail) when the configuration cannot judge:
ragas not installed, `LLM_MODE=client`, no key or model, or the hashing embedder. With
`RAGAS_REQUIRE_LIVE=1` every one of those skips becomes a failure, so the CI gate cannot pass
without scoring anything.

The judge refuses plain `http` unless the endpoint's host is loopback, a private IP or listed in
`OFFLINE_ALLOWED_HOSTS` (decided by the host, not by the provider preset).

### Caveats

- The judge is usually the same model family that wrote the answers, so expect self-grading bias.
- Current Claude models reject forced `tool_choice` and `temperature`/`top_p`; the judge sends
  `tool_choice: auto` plus a system instruction and drops both parameters.
- Not checked: the full `requirements-rag.txt` resolving together with sentence-transformers and
  gradio; `instructor` against real Claude responses with thinking blocks.
- Three samples is a smoke test, not a benchmark. Add samples to `_SAMPLES` in
  `tests/test_ragas.py` as the knowledge base grows.
