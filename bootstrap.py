"""Composition root: builds the runtime once and drives its lifecycle.

Every entry point (MCP stdio, REST/WS, Gradio, ingest and report CLIs) obtains its
:class:`~faq_core.service.SupportService` from here, so services are wired in
exactly one place. The answer engine (hybrid retrieval + LLM) is imported
lazily inside :func:`build_runtime`, after Langfuse is initialised, which also
keeps ``import bootstrap`` (and therefore ``import app, server``) cheap.

Startup (:func:`serve`) seeds an empty knowledge base and then **always**
reconciles the retrieval index with the database
(:meth:`~faq_core.service.SupportService.sync_index`), so entries added before a
restart — including ones whose background re-index job was lost — are
searchable again.

``LLM_MODE`` (plan §2) is applied here: :func:`build_runtime` first runs the
offline network guard (:mod:`faq_core.network_guard`; a no-op outside
``offline``), then builds the LLM gateway — a ``NullGateway`` in ``client``
mode — before anything else is constructed.

The HTTP surfaces (REST/WS, MCP over HTTP) also get their security policy here
(:func:`build_security`: token verifier, Origin/Host allowlists, rate limits).
It fails closed — missing auth settings stop the start — and MCP stdio never
builds one.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from background.scheduler import create_scheduler
from faq_core.config import get_config
from faq_core.events import EventBus
from faq_core.handoff import create_escalation_sink
from faq_core.models import FAQDraft
from faq_core.network_guard import enforce_network_policy, offline_url_policy
from faq_core.pii import PIIRedactor
from faq_core.repository import FAQRepository
from faq_core.retrieval.ingest import HttpxUrlFetcher, SourceLoader
from faq_core.service import SupportPolicy, SupportService, validate_metrics_query
from observability.log import configure_logging, get_logger
from observability.tracing import flush_tracing, init_langfuse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from apscheduler import AsyncScheduler

    from faq_core.config import AppConfig
    from faq_core.llm.gateway import LLMGateway
    from faq_core.models import IngestReport, WeeklyGapReport
    from faq_core.ports import AnswerEngine
    from faq_core.retrieval.embedders import Embedder
    from faq_core.security.jwks import JWKSFetcher
    from faq_core.security.policy import SecurityPolicy

_log = get_logger(__name__)

DEFAULT_SEED_PATH: Final = Path(__file__).resolve().parent / "data" / "sample_faq.json"


@dataclass(slots=True)
class Runtime:
    """Every long-lived object of one running process."""

    config: AppConfig
    service: SupportService
    engine: AnswerEngine
    repo: FAQRepository
    events: EventBus
    scheduler: AsyncScheduler
    loader: SourceLoader
    llm: LLMGateway | None = None


def assemble_runtime(
    config: AppConfig,
    engine: AnswerEngine,
    repo: FAQRepository,
    *,
    llm: LLMGateway | None = None,
    embedder: Embedder | None = None,
) -> Runtime:
    """Wire the service, event bus, re-index scheduler and ingest loader.

    *llm* is the gateway inside *engine*, kept so :func:`serve` can close it;
    *embedder* is the index's embedder, reused to cluster knowledge gaps
    (hashing when ``None``). The support-workflow policy, PII redactor and
    escalation sink are built here from *config*.

    Raises:
        ValueError: invalid confidence / escalation / sink settings.
    """
    events = EventBus()
    service = SupportService(
        engine=engine,
        store=repo,
        events=events,
        cache_ttl=config.CACHE_TTL,
        cache_max_entries=config.CACHE_MAX_ENTRIES,
        llm_mode=config.LLM_MODE,
        policy=SupportPolicy.from_config(config),
        redactor=PIIRedactor(enabled=config.PII_REDACTION),
        sink=create_escalation_sink(config),
        gap_embedder=embedder,
    )
    scheduler = create_scheduler(config, engine, repo, events)
    loader = SourceLoader(
        fetcher=HttpxUrlFetcher(
            timeout=config.INGEST_URL_TIMEOUT,
            max_bytes=config.INGEST_MAX_BYTES,
            url_policy=offline_url_policy(config),
        ),
        max_bytes=config.INGEST_MAX_BYTES,
    )
    return Runtime(
        config=config,
        service=service,
        engine=engine,
        repo=repo,
        events=events,
        scheduler=scheduler,
        loader=loader,
        llm=llm,
    )


def build_runtime(config: AppConfig, *, llm: LLMGateway | None = None) -> Runtime:
    """Build the production runtime (blocking: may load models, migrates the DB).

    Order matters: the offline network guard runs before anything can open a
    connection; Langfuse is initialised *before* the traced LLM/retrieval
    modules are imported, because ``@observe`` decides at decoration time
    whether to trace; the gateway is built before the database is opened so a
    configuration error (unknown provider, missing key) leaks nothing.
    *llm* replaces the configured gateway (tests, smoke runs).

    Raises:
        NetworkPolicyError: ``LLM_MODE=offline`` with a non-local endpoint
            (including the escalation webhook URL).
        ValueError: invalid LLM / retrieval / escalation settings.
    """
    enforce_network_policy(config)
    # Validate workflow + sink settings before anything opens a connection.
    SupportPolicy.from_config(config)
    create_escalation_sink(config)
    init_langfuse(config)
    from faq_core.llm.factory import create_llm_gateway
    from faq_core.retrieval.engine import create_answer_engine

    gateway = llm or create_llm_gateway(config)
    repo = FAQRepository(config)
    engine = create_answer_engine(config, gateway, repo)
    _log.info(
        "llm gateway ready",
        extra={"llm_mode": config.LLM_MODE.value, "llm_gateway": gateway.name},
    )
    return assemble_runtime(config, engine, repo, llm=gateway, embedder=engine.embedder)


def build_security(
    config: AppConfig, *, bind_host: str, jwks_fetcher: JWKSFetcher | None = None
) -> SecurityPolicy:
    """Build the security policy of an HTTP surface bound to *bind_host*.

    Runs the offline hosting guard first. *jwks_fetcher* replaces the HTTP
    key-set fetcher (tests). The caller's app closes the policy on shutdown.

    Raises:
        SecurityConfigError: missing / unsafe auth settings (fail closed), or
            ``AUTH_MODE=none`` on a non-loopback bind.
        NetworkPolicyError: ``LLM_MODE=offline`` with remote hosting settings.
    """
    # Imported here: the JWT stack (PyJWT + cryptography) is only needed by the
    # HTTP surfaces, so MCP stdio and `import app, server` stay cheap.
    from faq_core.security.policy import build_security as build_security_policy

    return build_security_policy(config, bind_host=bind_host, jwks_fetcher=jwks_fetcher)


async def _close_clients(runtime: Runtime) -> None:
    """Close the LLM gateway's HTTP connections, then the database."""
    try:
        if runtime.llm is not None:
            await runtime.llm.aclose()
    finally:
        await runtime.repo.close()


def load_seed_drafts(path: Path = DEFAULT_SEED_PATH) -> list[FAQDraft]:
    """Read seed FAQ entries from a JSON list of ``{question, answer, category}``."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FAQDraft.model_validate(item) for item in raw]


async def prepare_knowledge_base(
    runtime: Runtime, *, seed_path: Path | None = DEFAULT_SEED_PATH
) -> None:
    """Seed an empty knowledge base from *seed_path*, then sync the retrieval index.

    The sync runs on every start, not only after seeding: it rebuilds or
    verifies the index from the database contents.
    """
    if seed_path is not None:
        drafts = await asyncio.to_thread(load_seed_drafts, seed_path)
        await runtime.service.seed_if_empty(drafts)
    await runtime.service.sync_index()


@asynccontextmanager
async def serve(
    runtime: Runtime, *, seed_path: Path | None = DEFAULT_SEED_PATH
) -> AsyncIterator[SupportService]:
    """Run *runtime*: prepare the knowledge base, run the scheduler; close LLM + DB on exit."""
    try:
        await prepare_knowledge_base(runtime, seed_path=seed_path)
        async with runtime.scheduler:
            await runtime.scheduler.start_in_background()
            try:
                yield runtime.service
            finally:
                await runtime.events.drain()
    finally:
        await _close_clients(runtime)


@asynccontextmanager
async def production_service() -> AsyncIterator[SupportService]:
    """:data:`~faq_core.service.ServiceLifespan` for real deployments."""
    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    # Model loading and DB migration block; keep them off the event loop.
    runtime = await asyncio.to_thread(build_runtime, config)
    async with serve(runtime) as service:
        _log.info(
            "support service ready",
            extra={"kb_id": service.kb_id, "llm_mode": service.llm_mode.value},
        )
        yield service


async def ingest_sources(runtime: Runtime, sources: Sequence[str]) -> list[IngestReport]:
    """Load, store and index each of *sources* into *runtime*'s knowledge base.

    Syncs the index first (so the DB and index agree before new content lands)
    and closes the database afterwards. Does not seed the sample FAQ: a
    knowledge base that is being ingested into is the operator's own.

    Raises:
        IngestError: a source could not be loaded (earlier ones stay ingested).
    """
    try:
        await runtime.service.sync_index()
        reports: list[IngestReport] = []
        for source in sources:
            batch = await runtime.loader.load(source)
            reports.append(await runtime.service.ingest(batch))
        return reports
    finally:
        await runtime.events.drain()
        await _close_clients(runtime)


async def run_ingest(sources: Sequence[str]) -> list[IngestReport]:
    """Production entry point for ``python app.py --mode ingest``."""
    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    runtime = await asyncio.to_thread(build_runtime, config)
    return await ingest_sources(runtime, sources)


async def report_gaps(runtime: Runtime, window_days: int, limit: int) -> WeeklyGapReport:
    """Build the weekly knowledge-gap report from *runtime*'s database, then close it.

    Read-only: no seeding, no index sync (gap clustering only needs the
    embedder). Buffered trace scores are flushed before the process exits.

    Raises:
        InvalidRequestError: ``window_days`` / ``limit`` out of range.
    """
    try:
        return await runtime.service.weekly_gap_report(window_days, limit)
    finally:
        await runtime.events.drain()
        await _close_clients(runtime)
        await asyncio.to_thread(flush_tracing)


async def run_report(window_days: int, limit: int) -> WeeklyGapReport:
    """Production entry point for ``python app.py --mode report``.

    The arguments are validated before the runtime is built, so a bad flag
    neither migrates the database nor loads an embedding model.

    Raises:
        InvalidRequestError: ``window_days`` / ``limit`` out of range.
        ValueError: invalid settings (including pydantic ``ValidationError``).
        NetworkPolicyError: ``LLM_MODE=offline`` with a non-local endpoint.
    """
    validate_metrics_query(window_days, limit)
    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    runtime = await asyncio.to_thread(build_runtime, config)
    return await report_gaps(runtime, window_days, limit)
