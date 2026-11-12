"""``SupportService`` — the single business layer behind every surface.

MCP tools (``server.py``), REST/WebSocket routes (``api/``) and the Gradio UI
(``ui/``) are thin adapters: they translate transport input into calls on this
class and serialise its pydantic results. Nothing here imports a web, MCP or UI
framework.

Inputs arrive as plain values and are validated here, so every surface gets the
same rules and the same :class:`~faq_core.errors.InvalidRequestError` on bad
input. Results are the stable contract models from :mod:`faq_core.models`.

Support workflow (Phase 4), per query:

1. **Redact** the customer's text (:mod:`faq_core.pii`). Everything after this
   — retrieval, the LLM, traces, logs, the query log, events — sees only the
   redacted text; the knowledge base itself is never redacted.
2. **Triggers** on the text and the session (:mod:`faq_core.triage`):
   frustration, PII, legal/medical topics, repeated rephrasing.
3. **Retrieve** and **score confidence** (:mod:`faq_core.confidence`).
4. **Generate** (api/offline) only when the outcome is not ``escalate``;
   ``answer_with_caveat`` asks the model for a caveat. The optional LLM
   self-check can still turn the outcome into ``escalate``.
5. **Log** the query (redacted) to ``query_log`` — rows whose outcome is not
   ``answer`` form the knowledge-gap log — and emit ``query.answered`` /
   ``query.escalated`` / ``gap.logged``.

Escalations are created on request (:meth:`SupportService.create_escalation`),
packaged by :mod:`faq_core.handoff` and delivered through the configured sink.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from faq_core.cache import AsyncStampedeCache, MemoryTTLCache
from faq_core.confidence import ConfidenceScorer, ConfidenceThresholds, Outcome
from faq_core.errors import (
    AnswerGenerationUnavailableError,
    EscalationNotFoundError,
    FAQNotFoundError,
    InvalidRequestError,
    QueryNotFoundError,
)
from faq_core.events import (
    FAQ_INDEXED,
    build_escalation_created_event,
    build_faq_added_event,
    build_feedback_received_event,
    build_gap_logged_event,
    build_query_answered_event,
    build_query_escalated_event,
)
from faq_core.gaps import DEFAULT_CLUSTER_THRESHOLD, cluster_gaps
from faq_core.handoff import LogSink, build_packet
from faq_core.metrics import build_weekly_gap_report, collect_resolution_metrics, publish_metrics
from faq_core.models import (
    DEFAULT_KB_ID,
    DEFAULT_METRICS_ROW_LIMIT,
    DEFAULT_METRICS_WINDOW_DAYS,
    DEFAULT_REPORT_GAPS,
    CategoryList,
    ConfidenceSignal,
    DocumentChange,
    Escalation,
    EscalationRequest,
    EscalationResult,
    EscalationStatus,
    FAQDraft,
    FAQEntry,
    FeedbackEntry,
    FeedbackReceipt,
    FeedbackRequest,
    GapQuery,
    IngestReport,
    KBSearchResult,
    KnowledgeGapList,
    LLMMode,
    MetricsQuery,
    QueryLogRecord,
    QueryRequest,
    SearchResult,
    SourceRef,
    TranscriptTurn,
    utc_now,
)
from faq_core.pii import PLACEHOLDERS, PIIRedactor, strip_placeholders
from faq_core.retrieval.embedders import HashingEmbedder
from faq_core.retrieval.ranking import normalize_text
from faq_core.triage import (
    DEFAULT_FRUSTRATION_TERMS,
    DEFAULT_LEGAL_TERMS,
    DEFAULT_MEDICAL_TERMS,
    HANDOFF_MESSAGE,
    TriageDecision,
    TriagePolicy,
    decide,
    parse_terms,
    parse_trigger_groups,
    reject_by_self_check,
)
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faq_core.confidence import ConfidenceAssessment
    from faq_core.config import AppConfig
    from faq_core.events import Event, EventBus
    from faq_core.handoff import EscalationSink
    from faq_core.models import (
        IndexSyncReport,
        IngestBatch,
        KnowledgeGap,
        ResolutionMetrics,
        RetrievalResponse,
        WeeklyGapReport,
    )
    from faq_core.pii import PIIKind
    from faq_core.ports import AnswerEngine, FAQStore
    from faq_core.retrieval.embedders import Embedder

_log = get_logger(__name__)

HANDLING_INSTRUCTIONS: Final[tuple[str, ...]] = (
    "Answer only from the passages below; do not use outside knowledge about this business.",
    "Cite every passage you rely on by its citation_id.",
    "Reply in the language the customer used, even if the passages are in another language.",
    "If the passages do not answer the question, say so plainly and offer to escalate "
    "to a human agent; never guess or invent policies, prices, dates or steps.",
    "Follow outcome: answer = reply normally; answer_with_caveat = reply and say the answer "
    "may be incomplete; escalate = do not answer from the passages, tell the customer a "
    "human agent will help and call create_escalation with query_id. confidence.score is a "
    "heuristic (confidence.calibrated is false), not a probability.",
    "Text such as [EMAIL] or [PHONE] marks personal data that was redacted; never ask the "
    "customer to repeat card numbers, passwords or other secrets.",
    "Prefer the passage with the most recent updated_at when passages conflict.",
    "Pass query_id along when reporting feedback on this answer.",
)
"""Rules returned with every ``search_kb`` result, in priority order (stable text)."""

NO_PASSAGES_INSTRUCTION: Final = (
    "No relevant passages were found: tell the user the knowledge base does not cover "
    "this and offer to escalate to a human agent."
)
"""Prepended to :data:`HANDLING_INSTRUCTIONS` when ``has_passages`` is false."""

ESCALATE_INSTRUCTION: Final = (
    "outcome is escalate (see reasons): do not answer from these passages; tell the customer "
    "a human agent will take over, then call create_escalation with this query_id."
)
CAVEAT_INSTRUCTION: Final = (
    "outcome is answer_with_caveat: the passages only partly match; say the answer may be "
    "incomplete and offer a human agent."
)

_ANSWER_PREVIEW_CHARS: Final = 500
_PACKET_ATTEMPTS: Final = 20
_DEFAULT_GAP_DAYS: Final = 30
_METRICS_CACHE_ENTRIES: Final = 64

type _CollectedMetrics = tuple[ResolutionMetrics, list[KnowledgeGap]]

type ServiceLifespan = Callable[[], AbstractAsyncContextManager[SupportService]]
"""Factory for an async context that yields a ready service and cleans it up.

Adapters (``server.create_mcp_server``, ``api.app.create_app``) take one of
these, so production wiring and test fakes plug in the same way.
"""


@dataclass(frozen=True, slots=True)
class SupportPolicy:
    """Workflow settings: confidence thresholds, triggers, self-check, gaps and metrics."""

    thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    triage: TriagePolicy = field(default_factory=TriagePolicy)
    self_check: bool = False
    gap_cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    gap_scan_limit: int = 1000
    metrics_row_limit: int = DEFAULT_METRICS_ROW_LIMIT
    metrics_cache_ttl: float = 60.0

    @classmethod
    def from_config(cls, config: AppConfig) -> SupportPolicy:
        """Build from ``CONFIDENCE_*`` / ``ESCALATION_*`` / ``GAP_*`` / ``METRICS_*`` settings.

        ``CONFIDENCE_SELF_CHECK`` only takes effect when the server generates
        answers (never in ``LLM_MODE=client``).

        Raises:
            ValueError: inconsistent thresholds or an unknown trigger group.
        """
        if config.ESCALATION_REPHRASE_MIN_REPEATS < 1 or config.ESCALATION_REPHRASE_WINDOW < 1:
            raise ValueError("ESCALATION_REPHRASE_MIN_REPEATS and _WINDOW must be >= 1")
        if config.GAP_SCAN_LIMIT < 1:
            raise ValueError("GAP_SCAN_LIMIT must be >= 1")
        if config.METRICS_ROW_LIMIT < 1:
            raise ValueError("METRICS_ROW_LIMIT must be >= 1")
        if config.METRICS_CACHE_TTL < 0:
            raise ValueError("METRICS_CACHE_TTL must be >= 0")
        return cls(
            thresholds=ConfidenceThresholds(
                answer=config.CONFIDENCE_ANSWER_THRESHOLD,
                caveat=config.CONFIDENCE_CAVEAT_THRESHOLD,
                similarity_floor=config.CONFIDENCE_SIMILARITY_FLOOR,
                similarity_ceiling=config.CONFIDENCE_SIMILARITY_CEILING,
            ),
            triage=TriagePolicy(
                enabled=parse_trigger_groups(config.ESCALATION_TRIGGERS),
                frustration_terms=parse_terms(
                    config.ESCALATION_FRUSTRATION_TERMS, DEFAULT_FRUSTRATION_TERMS
                ),
                legal_terms=parse_terms(config.ESCALATION_LEGAL_TERMS, DEFAULT_LEGAL_TERMS),
                medical_terms=parse_terms(config.ESCALATION_MEDICAL_TERMS, DEFAULT_MEDICAL_TERMS),
                rephrase_similarity=config.ESCALATION_REPHRASE_SIMILARITY,
                rephrase_min_repeats=config.ESCALATION_REPHRASE_MIN_REPEATS,
                rephrase_window=config.ESCALATION_REPHRASE_WINDOW,
            ),
            self_check=config.CONFIDENCE_SELF_CHECK and config.LLM_MODE.generates_answers,
            gap_cluster_threshold=config.GAP_CLUSTER_THRESHOLD,
            gap_scan_limit=config.GAP_SCAN_LIMIT,
            metrics_row_limit=config.METRICS_ROW_LIMIT,
            metrics_cache_ttl=config.METRICS_CACHE_TTL,
        )


@dataclass(frozen=True, slots=True)
class _Prepared:
    """A validated query after redaction and text/session triggers."""

    request: QueryRequest
    redacted: str
    pii_kinds: tuple[PIIKind, ...]
    triggers: tuple[str, ...]

    @property
    def retrieval_query(self) -> str:
        # "[EMAIL]" would otherwise match passages about e-mail.
        return strip_placeholders(self.redacted)


@dataclass(frozen=True, slots=True)
class _Computed:
    """The cacheable part of a query: retrieval, confidence and the LLM answer."""

    retrieval: RetrievalResponse
    assessment: ConfidenceAssessment
    answer: str = ""
    generated: bool = False
    self_check: bool | None = None


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Everything one query produced, before it is shaped into a result model."""

    query_id: uuid.UUID
    prepared: _Prepared
    computed: _Computed
    decision: TriageDecision
    answer: str
    generated: bool
    cached: bool

    @property
    def confidence(self) -> ConfidenceSignal:
        assessment = self.computed.assessment
        return ConfidenceSignal(
            score=assessment.score,
            level=assessment.level,
            basis=(
                "retrieval_signals"
                if self.computed.self_check is None
                else "retrieval_signals+llm_self_check"
            ),
            signals=assessment.components,
        )


class SupportService:
    """Search/answer, triage, escalation, knowledge gaps, FAQ management and feedback.

    Args:
        engine: Answer engine (retrieval + generation), e.g.
            ``faq_core.retrieval.engine.RetrievalAnswerEngine``.
        store: Persistence, e.g. ``FAQRepository``.
        events: Bus that receives ``faq.added`` / ``query.answered`` /
            ``query.escalated`` / ``gap.logged`` / ``escalation.created`` /
            ``feedback.received``; the scheduler re-indexes on ``faq.added`` and
            publishes ``faq.indexed``, which clears the answer cache.
        cache_ttl: Seconds a computed answer is reused for an identical query.
        cache_max_entries: Bound on cached answers (LRU eviction beyond it).
        kb_id: Knowledge base this service instance serves.
        llm_mode: The configured ``LLM_MODE`` (reported to callers). Whether
            answers are generated is decided by ``engine.generates_answers``.
        policy: Confidence/trigger/gap settings (defaults: built-in values).
        redactor: PII redaction (default: on).
        sink: Escalation delivery (default: :class:`~faq_core.handoff.LogSink`).
        gap_embedder: Embedder used to cluster knowledge gaps (default:
            hashing; production passes the index's embedder).
    """

    def __init__(
        self,
        *,
        engine: AnswerEngine,
        store: FAQStore,
        events: EventBus,
        cache_ttl: float = 300,
        cache_max_entries: int = 1024,
        kb_id: str = DEFAULT_KB_ID,
        llm_mode: LLMMode = LLMMode.API,
        policy: SupportPolicy | None = None,
        redactor: PIIRedactor | None = None,
        sink: EscalationSink | None = None,
        gap_embedder: Embedder | None = None,
    ) -> None:
        self._engine = engine
        self._store = store
        self._events = events
        self._kb_id = kb_id
        self._llm_mode = llm_mode
        self._policy = policy or SupportPolicy()
        self._scorer = ConfidenceScorer(self._policy.thresholds)
        self._redactor = redactor or PIIRedactor()
        self._sink: EscalationSink = sink or LogSink()
        self._gap_embedder: Embedder = gap_embedder or HashingEmbedder()
        self._answer_cache: AsyncStampedeCache[_Computed] = AsyncStampedeCache(
            MemoryTTLCache(cache_max_entries), ttl=cache_ttl
        )
        # Metrics are recomputed at most once per window and TTL: polling a
        # dashboard must not re-cluster the gap log on every request.
        self._metrics_cache: AsyncStampedeCache[_CollectedMetrics] = AsyncStampedeCache(
            MemoryTTLCache(_METRICS_CACHE_ENTRIES), ttl=self._policy.metrics_cache_ttl
        )
        events.subscribe(FAQ_INDEXED, self._on_index_changed)

    @property
    def kb_id(self) -> str:
        return self._kb_id

    @property
    def llm_mode(self) -> LLMMode:
        return self._llm_mode

    @property
    def generates_answers(self) -> bool:
        """``False`` in ``LLM_MODE=client``: callers write replies from passages."""
        return self._engine.generates_answers

    # ── search / answer ──────────────────────────────────────────────────────

    async def search(
        self, query: str, top_k: int = 3, *, session_id: str | None = None
    ) -> SearchResult:
        """Answer *query* from the knowledge base using up to *top_k* sources.

        Identical ``(query, top_k)`` calls within the cache TTL reuse the
        computed answer (``cached=True``) but always get a fresh ``query_id``.
        ``citations`` lists every retrieved passage (FAQ or document) with its
        snippet and freshness date; ``sources`` the FAQ entries among them.

        When the server does not generate answers (``LLM_MODE=client``) the
        result has ``generated=False`` and ``answer=""``. When the outcome is
        ``escalate`` no LLM answer is generated (see :class:`SearchResult`).

        Raises:
            InvalidRequestError: blank query, ``top_k`` outside ``1..20`` or a
                malformed ``session_id``.
        """
        request = _validate(QueryRequest, query=query, top_k=top_k, session_id=session_id)
        outcome = await self._run(
            request, surface="search_faq", generate=self._engine.generates_answers
        )
        return self._search_result(outcome)

    async def answer(
        self, query: str, top_k: int = 3, *, session_id: str | None = None
    ) -> SearchResult:
        """Generate a grounded answer (``answer_question``); same result as :meth:`search`.

        Raises:
            AnswerGenerationUnavailableError: ``LLM_MODE=client`` — use
                :meth:`search_kb` and let the caller write the reply.
            InvalidRequestError: invalid input (see :meth:`search`).
        """
        if not self._engine.generates_answers:
            raise AnswerGenerationUnavailableError(self._llm_mode.value)
        request = _validate(QueryRequest, query=query, top_k=top_k, session_id=session_id)
        outcome = await self._run(request, surface="answer_question", generate=True)
        return self._search_result(outcome)

    async def search_kb(
        self, query: str, top_k: int = 3, *, session_id: str | None = None
    ) -> KBSearchResult:
        """Return retrieved passages, citations, confidence, outcome and handling instructions.

        Never calls an LLM (not even the self-check), so it works in every
        ``LLM_MODE``; the caller's own model writes the reply (passages-only
        contract, plan §2).

        Raises:
            InvalidRequestError: invalid input (see :meth:`search`).
        """
        request = _validate(QueryRequest, query=query, top_k=top_k, session_id=session_id)
        outcome = await self._run(request, surface="search_kb", generate=False)
        retrieval = outcome.computed.retrieval
        has_passages = bool(retrieval.passages)
        instructions = list(HANDLING_INSTRUCTIONS)
        if outcome.decision.escalate:
            instructions.insert(0, ESCALATE_INSTRUCTION)
        elif outcome.decision.outcome is Outcome.ANSWER_WITH_CAVEAT:
            instructions.insert(0, CAVEAT_INSTRUCTION)
        if not has_passages:
            instructions.insert(0, NO_PASSAGES_INSTRUCTION)
        return KBSearchResult(
            query_id=outcome.query_id,
            query=request.query,
            kb_id=self._kb_id,
            llm_mode=self._llm_mode,
            passages=retrieval.passages,
            citations=retrieval.citations,
            confidence=outcome.confidence,
            has_passages=has_passages,
            handling_instructions=instructions,
            outcome=outcome.decision.outcome,
            escalate=outcome.decision.escalate,
            reasons=list(outcome.decision.reasons),
            caveat=outcome.decision.caveat,
            pii_detected=[kind.value for kind in outcome.prepared.pii_kinds],
            session_id=request.session_id,
        )

    def _search_result(self, outcome: _Outcome) -> SearchResult:
        retrieval = outcome.computed.retrieval
        confidence = outcome.confidence
        return SearchResult(
            query_id=outcome.query_id,
            query=outcome.prepared.request.query,
            answer=outcome.answer,
            confidence=confidence.score,
            sources=retrieval.sources,
            cached=outcome.cached,
            citations=retrieval.citations,
            generated=outcome.generated,
            outcome=outcome.decision.outcome,
            escalate=outcome.decision.escalate,
            reasons=list(outcome.decision.reasons),
            caveat=outcome.decision.caveat,
            confidence_detail=confidence,
            pii_detected=[kind.value for kind in outcome.prepared.pii_kinds],
            session_id=outcome.prepared.request.session_id,
        )

    async def _prepare(self, request: QueryRequest) -> _Prepared:
        redaction = self._redactor.redact(request.query)
        policy = self._policy.triage
        triggers = policy.text_triggers(redaction.text, redaction.kinds)
        if request.session_id is not None:
            previous = await self._store.list_session_queries(
                self._kb_id, request.session_id, policy.rephrase_window
            )
            if policy.is_repeated_rephrase(redaction.text, [row.query for row in previous]):
                triggers.append("repeated_rephrase")
        return _Prepared(request, redaction.text, redaction.kinds, tuple(triggers))

    async def _run(self, request: QueryRequest, *, surface: str, generate: bool) -> _Outcome:
        started = time.perf_counter()
        prepared = await self._prepare(request)
        # A trigger already decided "escalate": do not spend an LLM call.
        want_llm = generate and not prepared.triggers
        key = AsyncStampedeCache.make_key(
            "search", self._kb_id, prepared.redacted, request.top_k, want_llm
        )
        computed_now = False

        async def _compute() -> _Computed:
            nonlocal computed_now
            computed_now = True
            return await self._compute(prepared, want_llm=want_llm)

        computed = await self._answer_cache.get_or_compute(key, _compute)
        decision = decide(computed.assessment, prepared.triggers, self._policy.triage)
        if computed.self_check is False:
            decision = reject_by_self_check(decision)
        if decision.escalate:
            answer, generated = (HANDOFF_MESSAGE if generate else ""), False
        else:
            answer, generated = computed.answer, computed.generated
        outcome = _Outcome(
            query_id=uuid.uuid4(),
            prepared=prepared,
            computed=computed,
            decision=decision,
            answer=answer,
            generated=generated,
            cached=not computed_now,
        )
        await self._record(outcome, surface=surface, started=started)
        return outcome

    async def _compute(self, prepared: _Prepared, *, want_llm: bool) -> _Computed:
        retrieval = await self._engine.retrieve(prepared.retrieval_query, prepared.request.top_k)
        assessment = self._scorer.assess(retrieval.evidence)
        if not want_llm or assessment.outcome is Outcome.ESCALATE:
            return _Computed(retrieval, assessment)
        caveat = assessment.outcome is Outcome.ANSWER_WITH_CAVEAT
        answer = await self._engine.generate(prepared.redacted, retrieval, caveat=caveat)
        self_check: bool | None = None
        if self._policy.self_check:
            self_check = await self._engine.verify(prepared.redacted, retrieval, answer)
        return _Computed(retrieval, assessment, answer, generated=True, self_check=self_check)

    async def _record(self, outcome: _Outcome, *, surface: str, started: float) -> None:
        """Write the query log row (redacted) and emit the query events."""
        retrieval = outcome.computed.retrieval
        decision = outcome.decision
        query_id = str(outcome.query_id)
        answer_preview = self._redactor.redact(outcome.answer).text[:_ANSWER_PREVIEW_CHARS]
        record = QueryLogRecord(
            id=outcome.query_id,
            kb_id=self._kb_id,
            query=outcome.prepared.redacted or "[REDACTED]",
            surface=surface,
            top_k=outcome.prepared.request.top_k,
            confidence=outcome.computed.assessment.score,
            source_ids=list(dict.fromkeys(c.source_id for c in retrieval.citations)),
            answered=not decision.escalate,
            latency_ms=int((time.perf_counter() - started) * 1000),
            session_id=outcome.prepared.request.session_id,
            outcome=decision.outcome,
            reasons=list(decision.reasons),
            citations=[
                SourceRef(
                    citation_id=c.id,
                    source_type=c.source_type,
                    source_id=c.source_id,
                    title=c.title,
                    source_uri=c.source_uri,
                    score=c.score,
                )
                for c in retrieval.citations
            ],
            answer_preview=answer_preview,
        )
        await self._store.log_query(record)
        self._events.emit(
            build_query_answered_event(
                query_id,
                record.query,
                answer_preview,
                cached=outcome.cached,
                outcome=decision.outcome.value,
                confidence=record.confidence or 0.0,
                source=surface,
            )
        )
        if decision.escalate:
            self._events.emit(
                build_query_escalated_event(
                    query_id,
                    ",".join(decision.reasons),
                    source=surface,
                    session_id=record.session_id,
                )
            )
        if decision.outcome is not Outcome.ANSWER:
            self._events.emit(
                build_gap_logged_event(query_id, record.query, decision.outcome.value, self._kb_id)
            )

    # ── escalation hand-off ──────────────────────────────────────────────────

    async def create_escalation(
        self,
        *,
        query_id: str | uuid.UUID | None = None,
        session_id: str | None = None,
        customer_goal: str = "",
        reason: str = "",
        transcript: Sequence[TranscriptTurn | dict[str, object]] = (),
    ) -> EscalationResult:
        """Hand the conversation to a human: build the packet, store it, deliver it.

        The packet (``support.escalation/v1``) carries the customer goal, the
        sources cited so far, the session's attempts with outcome and
        confidence, the reason codes and a redacted transcript. Where it is
        delivered is server configuration (``ESCALATION_SINK``), not input.

        Raises:
            InvalidRequestError: none of query_id/session_id/customer_goal, or
                malformed input.
            QueryNotFoundError: *query_id* was never logged.
        """
        request = _validate(
            EscalationRequest,
            query_id=query_id,
            session_id=session_id,
            customer_goal=customer_goal,
            reason=reason,
            transcript=list(transcript),
        )
        anchor: QueryLogRecord | None = None
        if request.query_id is not None:
            anchor = await self._store.get_query(str(request.query_id))
            if anchor is None or anchor.kb_id != self._kb_id:
                raise QueryNotFoundError(str(request.query_id))
        session = request.session_id or (anchor.session_id if anchor else None)
        attempts: list[QueryLogRecord] = []
        if session is not None:
            attempts = await self._store.list_session_queries(
                self._kb_id, session, _PACKET_ATTEMPTS
            )
        if anchor is not None and all(a.id != anchor.id for a in attempts):
            attempts.append(anchor)

        pii: set[str] = set()

        def _redact(text: str) -> str:
            result = self._redactor.redact(text)
            pii.update(kind.value for kind in result.kinds)
            return result.text.strip()

        goal = _redact(request.customer_goal) or (attempts[0].query if attempts else "")
        reason_detail = _redact(request.reason)
        turns = [
            TranscriptTurn(role=turn.role, text=_redact(turn.text)) for turn in request.transcript
        ]
        for attempt in attempts:  # stored queries carry placeholders, not the data
            pii.update(kind.value for kind, mark in PLACEHOLDERS.items() if mark in attempt.query)
        latest = anchor or (attempts[-1] if attempts else None)
        reasons = list(dict.fromkeys(["requested", *(latest.reasons if latest else [])]))

        escalation_id = uuid.uuid4()
        created_at = utc_now()
        packet = build_packet(
            escalation_id=str(escalation_id),
            kb_id=self._kb_id,
            created_at=created_at.isoformat(),
            customer_goal=goal,
            reasons=reasons,
            reason_detail=reason_detail,
            attempts=attempts,
            transcript=turns,
            query_id=str(anchor.id) if anchor else None,
            session_id=session,
            pii_detected=sorted(pii),
        )
        escalation = Escalation(
            id=escalation_id,
            kb_id=self._kb_id,
            query_id=anchor.id if anchor else None,
            reason=",".join(reasons),
            packet=packet,
            created_at=created_at,
            session_id=session,
        )
        await self._store.create_escalation(escalation)
        delivery = await self._sink.deliver(escalation)
        status = (
            EscalationStatus.SENT
            if self._sink.forwards and delivery.delivered
            else EscalationStatus.OPEN
        )
        await self._store.update_escalation_delivery(str(escalation_id), status, delivery)
        self._events.emit(
            build_escalation_created_event(
                str(escalation_id),
                escalation.reason,
                sink=delivery.sink,
                delivered=delivery.delivered,
            )
        )
        return _escalation_result(
            escalation.model_copy(update={"status": status, "delivery": delivery})
        )

    async def get_escalation(self, escalation_id: str) -> EscalationResult:
        """Return one escalation with its packet and delivery state.

        Raises:
            InvalidRequestError: *escalation_id* is not a UUID.
            EscalationNotFoundError: no such escalation in this knowledge base.
        """
        try:
            key = str(uuid.UUID(str(escalation_id)))
        except ValueError as err:
            raise InvalidRequestError(
                "invalid request",
                fields=[{"field": "escalation_id", "problem": "must be a UUID"}],
            ) from err
        escalation = await self._store.get_escalation(key)
        if escalation is None or escalation.kb_id != self._kb_id:
            raise EscalationNotFoundError(key)
        return _escalation_result(escalation)

    # ── knowledge gaps ───────────────────────────────────────────────────────

    async def list_knowledge_gaps(
        self, *, limit: int = 20, min_count: int = 1, since: object = None
    ) -> KnowledgeGapList:
        """Cluster unanswered / low-confidence / escalated queries into a ranked backlog.

        Raises:
            InvalidRequestError: ``limit`` outside 1..100, ``min_count`` < 1
                or an unparseable ``since``.
        """
        request = _validate(GapQuery, limit=limit, min_count=min_count, since=since)
        since_at = request.since or utc_now() - timedelta(days=_DEFAULT_GAP_DAYS)
        rows = await self._store.list_gap_queries(
            self._kb_id, since_at, self._policy.gap_scan_limit
        )
        clusters = await cluster_gaps(
            rows, self._gap_embedder, threshold=self._policy.gap_cluster_threshold
        )
        gaps = [gap for gap in clusters if gap.count >= request.min_count][: request.limit]
        return KnowledgeGapList(kb_id=self._kb_id, since=since_at, scanned=len(rows), gaps=gaps)

    # ── resolution metrics and the weekly gap report (Phase 6) ───────────────

    async def resolution_metrics(
        self, window_days: int = DEFAULT_METRICS_WINDOW_DAYS
    ) -> ResolutionMetrics:
        """Outcomes, escalation rate, 48h re-contact proxy, gaps and feedback over a window.

        See :mod:`faq_core.metrics`. A read without side effects: it pushes no
        trace scores and emits no event. Results are cached per window for
        ``METRICS_CACHE_TTL`` seconds, so repeated calls inside that time
        return the same object.

        Raises:
            InvalidRequestError: ``window_days`` outside ``1..365``.
        """
        request = _validate(MetricsQuery, window_days=window_days)
        return (await self._collect_metrics(request.window_days))[0]

    async def weekly_gap_report(
        self, window_days: int = DEFAULT_METRICS_WINDOW_DAYS, limit: int = DEFAULT_REPORT_GAPS
    ) -> WeeklyGapReport:
        """Metrics, up to *limit* ranked gaps, suggested FAQ topics and a Markdown rendering.

        The report path (``GET /reports/weekly-gaps`` and ``--mode report``)
        is the one that pushes the metrics to the trace layer and emits
        ``metrics.reported``.

        Raises:
            InvalidRequestError: ``window_days`` outside ``1..365`` / ``limit`` outside ``1..100``.
        """
        request = _validate(MetricsQuery, window_days=window_days, limit=limit)
        metrics, gaps = await self._collect_metrics(request.window_days)
        publish_metrics(metrics, self._events, source="weekly_gap_report")
        return build_weekly_gap_report(metrics, gaps, generated_at=metrics.until, limit=limit)

    async def _collect_metrics(self, window_days: int) -> _CollectedMetrics:
        """Compute (or reuse, within the TTL) the window's metrics and ranked gaps.

        The key is ``(kb_id, window_days)``: the report's ``limit`` only slices
        the cached gaps, so it does not need its own entry.
        """

        async def _compute() -> _CollectedMetrics:
            return await collect_resolution_metrics(
                self._store,
                self._gap_embedder,
                kb_id=self._kb_id,
                window_days=window_days,
                until=utc_now(),
                cluster_threshold=self._policy.gap_cluster_threshold,
                gap_scan_limit=self._policy.gap_scan_limit,
                row_limit=self._policy.metrics_row_limit,
            )

        if self._policy.metrics_cache_ttl <= 0:
            return await _compute()
        key = AsyncStampedeCache.make_key("metrics", self._kb_id, window_days)
        return await self._metrics_cache.get_or_compute(key, _compute)

    # ── FAQ entries, feedback, index ─────────────────────────────────────────

    async def add_entry(self, question: str, answer: str, category: str) -> FAQEntry:
        """Persist a new FAQ entry and announce it for background re-indexing.

        Clears the answer cache now and again once the entry is indexed
        (``faq.indexed``), so no cached answer predates it.

        Raises:
            InvalidRequestError: a blank question, answer or category.
        """
        draft = _validate(FAQDraft, question=question, answer=answer, category=category)
        entry = await self._store.create(FAQEntry(**draft.model_dump(), kb_id=self._kb_id))
        self.clear_answer_cache()
        self._events.emit(build_faq_added_event(str(entry.id), entry.category, entry.kb_id))
        return entry

    async def get_entry(self, faq_id: str) -> FAQEntry:
        """Return one FAQ entry.

        Raises:
            FAQNotFoundError: no entry with this id.
        """
        entry = await self._store.get(faq_id)
        if entry is None:
            raise FAQNotFoundError(faq_id)
        return entry

    async def list_categories(self) -> CategoryList:
        """Return the sorted, de-duplicated categories of this knowledge base."""
        categories = await self._store.list_categories(self._kb_id)
        return CategoryList(kb_id=self._kb_id, categories=categories)

    async def record_feedback(
        self,
        query_id: str | uuid.UUID,
        rating: int,
        comment: str = "",
        *,
        helpful: bool = True,
    ) -> FeedbackReceipt:
        """Store a rating for the answer identified by *query_id* (comment redacted).

        *query_id* must be a query logged in this knowledge base. Each answer
        keeps one rating: rating it again replaces the earlier rating (the
        latest wins), so retries cannot skew the negative-feedback rate. A
        rating of 2 or less, or ``helpful=False``, puts the query into the
        knowledge-gap log.

        Raises:
            InvalidRequestError: malformed ``query_id`` or ``rating`` outside ``1..5``.
            QueryNotFoundError: *query_id* was never logged in this knowledge base.
        """
        request = _validate(
            FeedbackRequest, query_id=query_id, rating=rating, comment=comment, helpful=helpful
        )
        anchor = await self._store.get_query(str(request.query_id))
        if anchor is None or anchor.kb_id != self._kb_id:
            raise QueryNotFoundError(str(request.query_id))
        feedback = FeedbackEntry(
            **request.model_dump(exclude={"comment"}),
            comment=self._redactor.redact(request.comment).text,
            kb_id=self._kb_id,
        )
        await self._store.save_feedback(feedback)
        self._events.emit(
            build_feedback_received_event(str(feedback.id), str(feedback.query_id), feedback.rating)
        )
        return FeedbackReceipt(feedback_id=feedback.id)

    def clear_answer_cache(self) -> None:
        """Forget every cached answer and metrics result (the knowledge base changed)."""
        self._answer_cache.clear()
        self._metrics_cache.clear()

    async def _on_index_changed(self, _event: Event) -> None:
        self.clear_answer_cache()

    async def seed_if_empty(self, drafts: Sequence[FAQDraft]) -> int:
        """Load *drafts* into an empty knowledge base; return how many were added.

        Indexes them in one batch directly (no per-entry ``faq.added`` events).
        Does nothing when the knowledge base already has entries.
        """
        if not drafts or await self._store.count(self._kb_id) > 0:
            return 0
        entries = [FAQEntry(**draft.model_dump(), kb_id=self._kb_id) for draft in drafts]
        await self._store.create_many(entries)
        await self._engine.seed_from_entries(entries)
        self.clear_answer_cache()
        _log.info("seeded knowledge base", extra={"kb_id": self._kb_id, "entries": len(entries)})
        return len(entries)

    async def sync_index(self) -> IndexSyncReport:
        """Rebuild/verify the retrieval index from the database (run at every startup).

        Indexes entries and documents that have no (or stale) chunks — e.g.
        entries whose background re-index job was lost in a restart — and drops
        index rows of sources that no longer exist.
        """
        entries = await self._store.list_all(self._kb_id)
        documents = await self._store.list_documents(self._kb_id)
        report = await self._engine.sync_index(entries, documents)
        self.clear_answer_cache()
        return report

    async def ingest(self, batch: IngestBatch) -> IngestReport:
        """Store and index everything loaded from one ingest source.

        FAQ drafts whose normalised question already exists in the knowledge
        base (or earlier in the batch) are counted as ``faqs_duplicate`` and
        skipped. Documents are keyed by ``source_uri``: unchanged ones are left
        alone, changed ones are updated and re-indexed. New entries are indexed
        directly (no ``faq.added`` events), like seeding. The answer cache is
        cleared afterwards.
        """
        known = {normalize_text(e.question) for e in await self._store.list_all(self._kb_id)}
        new_entries: list[FAQEntry] = []
        for draft in batch.faqs:
            key = normalize_text(draft.question)
            if key in known:
                continue
            known.add(key)
            new_entries.append(FAQEntry(**draft.model_dump(), kb_id=self._kb_id))
        if new_entries:
            await self._store.create_many(new_entries)
            await self._engine.seed_from_entries(new_entries)

        changes = {change: 0 for change in DocumentChange}
        for document_draft in batch.documents:
            document, change = await self._store.upsert_document(document_draft, self._kb_id)
            changes[change] += 1
            if change is not DocumentChange.UNCHANGED:
                await self._engine.index_document(document)
        self.clear_answer_cache()

        report = IngestReport(
            source=batch.source,
            kb_id=self._kb_id,
            faqs_added=len(new_entries),
            faqs_duplicate=len(batch.faqs) - len(new_entries),
            documents_added=changes[DocumentChange.ADDED],
            documents_updated=changes[DocumentChange.UPDATED],
            documents_unchanged=changes[DocumentChange.UNCHANGED],
            skipped=batch.skipped,
        )
        _log.info(
            "ingested source",
            extra=report.model_dump(exclude={"skipped"}) | {"skipped": len(report.skipped)},
        )
        return report


def _escalation_result(escalation: Escalation) -> EscalationResult:
    reasons = escalation.packet.get("reasons")
    return EscalationResult(
        escalation_id=escalation.id,
        kb_id=escalation.kb_id,
        status=escalation.status,
        query_id=escalation.query_id,
        session_id=escalation.session_id,
        reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [escalation.reason],
        packet=escalation.packet,
        delivery=escalation.delivery,
        created_at=escalation.created_at,
    )


def validate_metrics_query(
    window_days: int = DEFAULT_METRICS_WINDOW_DAYS, limit: int = DEFAULT_REPORT_GAPS
) -> MetricsQuery:
    """Validate metrics / report input the way the service does (for early CLI checks).

    Raises:
        InvalidRequestError: ``window_days`` outside ``1..365`` / ``limit`` outside ``1..100``.
    """
    return _validate(MetricsQuery, window_days=window_days, limit=limit)


def _validate[M: BaseModel](model: type[M], **data: object) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as err:
        fields = [
            {"field": ".".join(str(part) for part in error["loc"]), "problem": error["msg"]}
            for error in err.errors()
        ]
        raise InvalidRequestError("invalid request", fields=fields) from err
