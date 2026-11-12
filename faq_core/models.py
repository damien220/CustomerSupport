"""Pydantic v2 domain models shared by every surface.

These replace the former ``inout_models`` base classes with plain pydantic and
keep their validation semantics: ``extra="forbid"``, non-blank strings
(whitespace-only rejected, content preserved as entered), UUID ids that accept
strings, and timezone-aware timestamps normalised to UTC.

The models whose names end in ``Result``/``List``/``Receipt`` (plus
:class:`FAQEntry`) are the **public tool contract** returned by the MCP tools and
REST routes. Their field names are stable: add fields, never rename or remove.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from faq_core.confidence import ConfidenceLevel, HitEvidence, Outcome

DEFAULT_KB_ID: Final = "default"
"""Knowledge-base id used while the deployment is single-tenant (plan decision #4)."""

MAX_TOP_K: Final = 20
"""Upper bound on retrieved sources per query, to bound LLM context size."""


class LLMMode(StrEnum):
    """Who writes the reply (``LLM_MODE``; plan §2).

    ``api``: this server calls a hosted or local LLM provider. ``client``: the
    calling agent's own model writes the reply from retrieved passages (no LLM
    on this server). ``offline``: a local LLM, and every configured endpoint
    must be local (startup network guard).
    """

    API = "api"
    CLIENT = "client"
    OFFLINE = "offline"

    @property
    def generates_answers(self) -> bool:
        """Whether this server generates answers itself in this mode."""
        return self is not LLMMode.CLIENT


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace-only")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


NonEmptyStr = Annotated[str, AfterValidator(_reject_blank)]
UtcTimestamp = Annotated[datetime, AfterValidator(_require_utc)]
SessionId = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:\-]+$")
]
"""Caller-chosen conversation id (letters, digits, ``._:-``; at most 128 chars)."""

MAX_TRANSCRIPT_TURNS: Final = 50
MAX_TURN_CHARS: Final = 4000


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


class _Model(BaseModel):
    # Frozen: cached answers share their source objects between callers.
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── FAQ entries ──────────────────────────────────────────────────────────────


class FAQDraft(_Model):
    """Input for creating an FAQ entry (no server-assigned fields)."""

    question: NonEmptyStr
    answer: NonEmptyStr
    category: NonEmptyStr


class FAQEntry(_Model):
    """A stored FAQ entry. Also the ``add_faq_entry``/``get_faq_entry`` tool result.

    Fields: ``id`` (UUID string), ``question``, ``answer``, ``category``,
    ``kb_id``, ``created_at`` (ISO-8601 UTC).
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question: NonEmptyStr
    answer: NonEmptyStr
    category: NonEmptyStr
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    created_at: UtcTimestamp = Field(default_factory=utc_now)


# ── Search ───────────────────────────────────────────────────────────────────


class QueryRequest(_Model):
    """Input for a knowledge-base search.

    ``session_id`` (optional, added in Phase 4) groups the queries of one
    conversation: it feeds the repeated-rephrase trigger and the escalation
    packet's ``attempts``.
    """

    query: NonEmptyStr
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)
    session_id: SessionId | None = None


class SourceType(StrEnum):
    """What a retrieved passage was cut from."""

    FAQ = "faq"
    DOCUMENT = "document"


class Citation(_Model):
    """One retrieved passage backing an answer (an item of ``citations[]``).

    Fields: ``id`` (chunk UUID), ``source_type`` (``"faq"`` | ``"document"``),
    ``source_id`` (the FAQ entry or document UUID), ``title`` (FAQ question or
    document title), ``heading`` (section path inside a document, ``""`` for
    FAQ entries), ``snippet`` (short excerpt around the query terms),
    ``source_uri`` (where an ingested document came from, ``""`` for FAQ
    entries), ``updated_at`` (freshness date of the source, ISO-8601 UTC) and
    ``score`` (retrieval confidence of this passage, 0.0-1.0).
    """

    id: uuid.UUID
    source_type: SourceType
    source_id: uuid.UUID
    title: str
    heading: str = ""
    snippet: str
    source_uri: str = ""
    updated_at: UtcTimestamp
    score: float = 0.0


class AnswerResponse(_Model):
    """Raw output of the answer engine for one query.

    Internal: surfaces return :class:`SearchResult`, which adds per-call fields.
    """

    query: str
    answer: str
    sources: list[FAQEntry]
    confidence: float = 0.0
    citations: list[Citation] = Field(default_factory=list)


class SearchResult(_Model):
    """``search_faq`` / ``answer_question`` tool result.

    Fields: ``query_id`` (UUID to reference in feedback), ``query``, ``answer``,
    ``confidence`` (retrieval confidence of the best passage, 0.0 when none),
    ``sources`` (the :class:`FAQEntry` items behind FAQ-derived passages),
    ``cached`` (answer served from the TTL cache), ``citations`` (every
    retrieved passage — FAQ or ingested document — as :class:`Citation`, best
    first; added in Phase 2) and ``generated`` (``False`` when this server does
    not generate answers — ``LLM_MODE=client`` — so ``answer`` is ``""`` and the
    caller writes the reply from ``citations``; added in Phase 3).

    Added in Phase 4: ``confidence`` is now the Phase 4 score (same as
    ``confidence_detail.score``); ``outcome`` (``answer`` |
    ``answer_with_caveat`` | ``escalate``), ``escalate`` (``outcome ==
    "escalate"``), ``reasons`` (stable reason codes, see
    ``faq_core.triage``), ``caveat`` (text to show with a medium-confidence
    answer, ``""`` otherwise), ``confidence_detail``
    (:class:`ConfidenceSignal`), ``pii_detected`` (kinds of personal data
    found — and redacted — in the query) and ``session_id`` (echoed). When
    ``escalate`` is true no LLM answer is generated: ``answer`` holds a fixed
    hand-off message (``""`` in ``LLM_MODE=client``) and ``generated`` is
    ``False``.
    """

    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query: str
    answer: str
    confidence: float
    sources: list[FAQEntry]
    cached: bool = False
    citations: list[Citation] = Field(default_factory=list)
    generated: bool = True
    outcome: Outcome = Outcome.ANSWER
    escalate: bool = False
    reasons: list[str] = Field(default_factory=list)
    caveat: str = ""
    confidence_detail: ConfidenceSignal | None = None
    pii_detected: list[str] = Field(default_factory=list)
    session_id: str | None = None


class Passage(_Model):
    """One retrieved passage with its full text (an item of ``passages[]``).

    Fields: ``rank`` (1 = best), ``citation_id`` (the matching
    :class:`Citation` ``id`` — cite it), ``source_type``, ``source_id``,
    ``title``, ``heading``, ``text`` (the whole passage), ``source_uri``,
    ``updated_at`` and ``score`` (retrieval score, 0.0-1.0).
    """

    rank: int = Field(ge=1)
    citation_id: uuid.UUID
    source_type: SourceType
    source_id: uuid.UUID
    title: str
    heading: str = ""
    text: str
    source_uri: str = ""
    updated_at: UtcTimestamp
    score: float = 0.0


class RetrievalResponse(_Model):
    """Raw retrieval-only output of the answer engine (no generation).

    Internal: surfaces return :class:`KBSearchResult`.
    """

    query: str
    passages: list[Passage]
    citations: list[Citation]
    sources: list[FAQEntry]
    confidence: float = 0.0
    evidence: HitEvidence | None = None


class ConfidenceSignal(_Model):
    """How much to trust the retrieved passages (Phase 4).

    ``score`` (0.0-1.0) is a weighted mean of retrieval evidence for the top
    passage — query-term coverage, dense similarity, BM25/dense agreement,
    margin over the runner-up, reranker score — see ``faq_core.confidence``;
    0.0 when nothing was found. ``level`` buckets it with the configured
    thresholds (``high`` / ``medium`` / ``low``). ``basis`` names the method
    (``retrieval_signals``, or ``retrieval_signals+llm_self_check`` when the
    optional LLM self-check ran). ``calibrated`` is always ``False``: the score
    is a heuristic tuned on a small benchmark, not a probability. ``signals``
    holds the components that were available, each 0.0-1.0.
    """

    score: float
    level: ConfidenceLevel = ConfidenceLevel.LOW
    basis: Literal["retrieval_signals", "retrieval_signals+llm_self_check"] = "retrieval_signals"
    calibrated: bool = False
    signals: dict[str, float] = Field(default_factory=dict)


class KBSearchResult(_Model):
    """``search_kb`` tool result: passages only, for the caller to answer from.

    Fields: ``query_id`` (UUID to reference in feedback), ``query``, ``kb_id``,
    ``llm_mode`` (``api`` | ``client`` | ``offline``), ``passages`` (best first,
    full text), ``citations`` (same passages as :class:`Citation` objects),
    ``confidence`` (:class:`ConfidenceSignal`), ``has_passages`` (``False`` when
    nothing relevant was retrieved) and ``handling_instructions`` (rules the
    answering model must follow, in priority order). Added in Phase 4:
    ``outcome``, ``escalate``, ``reasons``, ``caveat``, ``pii_detected`` and
    ``session_id`` — same meaning as on :class:`SearchResult`.
    """

    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query: str
    kb_id: str = DEFAULT_KB_ID
    llm_mode: LLMMode
    passages: list[Passage]
    citations: list[Citation]
    confidence: ConfidenceSignal
    has_passages: bool
    handling_instructions: list[str]
    outcome: Outcome = Outcome.ANSWER
    escalate: bool = False
    reasons: list[str] = Field(default_factory=list)
    caveat: str = ""
    pii_detected: list[str] = Field(default_factory=list)
    session_id: str | None = None


class CategoryList(_Model):
    """``list_faq_categories`` tool result: sorted, de-duplicated categories."""

    kb_id: str = DEFAULT_KB_ID
    categories: list[str]


# ── Feedback ─────────────────────────────────────────────────────────────────


class FeedbackRequest(_Model):
    """Input for rating an answer."""

    query_id: uuid.UUID
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    helpful: bool = True


class FeedbackEntry(_Model):
    """A stored feedback record."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query_id: uuid.UUID
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    helpful: bool = True
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    created_at: UtcTimestamp = Field(default_factory=utc_now)


class FeedbackReceipt(_Model):
    """Result of recording feedback."""

    status: str = "recorded"
    feedback_id: uuid.UUID


# ── Documents, chunks and ingestion ──────────────────────────────────────────


class DocumentFormat(StrEnum):
    """Markup of a stored document body (decides how it is chunked)."""

    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentDraft(_Model):
    """An ingested document before it is stored (markdown file, text file, web page)."""

    title: NonEmptyStr
    body: NonEmptyStr
    format: DocumentFormat = DocumentFormat.MARKDOWN
    source_uri: NonEmptyStr
    updated_at: UtcTimestamp = Field(default_factory=utc_now)


class Document(_Model):
    """A stored knowledge-base document; re-ingesting its ``source_uri`` updates it."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    title: NonEmptyStr
    body: NonEmptyStr
    format: DocumentFormat = DocumentFormat.MARKDOWN
    source_uri: NonEmptyStr
    content_hash: str = ""
    updated_at: UtcTimestamp = Field(default_factory=utc_now)
    created_at: UtcTimestamp = Field(default_factory=utc_now)


class DocumentChange(StrEnum):
    """Outcome of storing a :class:`DocumentDraft`."""

    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class Chunk(_Model):
    """A retrievable slice of an FAQ entry or ingested document.

    Exactly one of ``faq_id`` / ``document_id`` names the source. ``title`` is
    the FAQ question or document title, ``heading`` the section path inside a
    document. ``content_hash`` fingerprints the normalised content (dedupe);
    ``source_hash`` fingerprints the source version it was cut from, so a
    changed source (or chunker) is detected and re-chunked. ``updated_at`` is
    the source's freshness date.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    faq_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    ordinal: int = Field(default=0, ge=0)
    title: str = ""
    heading: str = ""
    content: NonEmptyStr
    source_uri: str = ""
    content_hash: str = ""
    source_hash: str = ""
    updated_at: UtcTimestamp = Field(default_factory=utc_now)
    created_at: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _one_source(self) -> Chunk:
        if self.faq_id is not None and self.document_id is not None:
            raise ValueError("a chunk belongs to an FAQ entry or a document, not both")
        return self

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT if self.document_id is not None else SourceType.FAQ

    @property
    def source_id(self) -> uuid.UUID | None:
        return self.document_id if self.document_id is not None else self.faq_id


class SkippedSource(_Model):
    """A file or URL that ingestion could not use, and why."""

    source: str
    reason: str


class IngestBatch(_Model):
    """Everything loaded from one ingest source, ready for :meth:`SupportService.ingest`.

    CSV rows and FAQ JSON lists become ``faqs``; markdown, text and HTML become
    ``documents``.
    """

    source: str
    faqs: list[FAQDraft] = Field(default_factory=list)
    documents: list[DocumentDraft] = Field(default_factory=list)
    skipped: list[SkippedSource] = Field(default_factory=list)


class IngestReport(_Model):
    """What one ingest run changed in the knowledge base."""

    source: str
    kb_id: str = DEFAULT_KB_ID
    faqs_added: int = 0
    faqs_duplicate: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_unchanged: int = 0
    skipped: list[SkippedSource] = Field(default_factory=list)


class IndexSyncReport(_Model):
    """Result of reconciling the retrieval index with the database at startup.

    ``sources_indexed`` FAQ entries/documents were (re-)chunked because they had
    no chunks or stale ones; ``sources_removed`` sources no longer exist;
    ``fts_repaired`` keyword-index rows were added or dropped;
    ``vectors_embedded`` / ``vectors_removed`` dense vectors were (re)computed or
    dropped. ``dense_enabled`` is ``False`` when only BM25 is available.
    """

    kb_id: str = DEFAULT_KB_ID
    sources_indexed: int = 0
    sources_removed: int = 0
    fts_repaired: int = 0
    vectors_embedded: int = 0
    vectors_removed: int = 0
    chunks: int = 0
    dense_enabled: bool = False


# ── Query log, escalations, knowledge gaps (Phase 4) ─────────────────────────


class SourceRef(_Model):
    """A compact reference to a cited passage, kept in the query log and packets."""

    citation_id: uuid.UUID
    source_type: SourceType
    source_id: uuid.UUID
    title: str
    source_uri: str = ""
    score: float = 0.0


class QueryLogRecord(_Model):
    """One answered (or unanswered) query, for gap analysis and resolution metrics.

    ``id`` is the ``query_id`` returned to the caller. ``query`` and
    ``answer_preview`` are stored **redacted**. ``outcome`` / ``reasons`` are
    the triage result; rows whose outcome is not ``answer`` form the
    knowledge-gap log.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    query: NonEmptyStr
    surface: NonEmptyStr
    top_k: int = Field(ge=1)
    confidence: float | None = None
    source_ids: list[uuid.UUID] = Field(default_factory=list)
    answered: bool = True
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    session_id: str | None = None
    outcome: Outcome = Outcome.ANSWER
    reasons: list[str] = Field(default_factory=list)
    citations: list[SourceRef] = Field(default_factory=list)
    answer_preview: str = ""


class EscalationStatus(StrEnum):
    OPEN = "open"
    SENT = "sent"
    RESOLVED = "resolved"


class EscalationDelivery(_Model):
    """How the hand-off packet was delivered: ``sink`` name, ``delivered``,
    ``attempts``, ``error`` (``""`` on success; never contains secrets) and
    ``delivered_at``."""

    sink: str
    delivered: bool = False
    attempts: int = 0
    error: str = ""
    delivered_at: UtcTimestamp | None = None


class Escalation(_Model):
    """A hand-off of a conversation to a human, with its context packet.

    ``reason`` is the comma-joined reason codes (the packet has the list).
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kb_id: NonEmptyStr = DEFAULT_KB_ID
    query_id: uuid.UUID | None = None
    reason: NonEmptyStr
    status: EscalationStatus = EscalationStatus.OPEN
    packet: dict[str, object] = Field(default_factory=dict)
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    session_id: str | None = None
    delivery: EscalationDelivery | None = None
    updated_at: UtcTimestamp | None = None


class TranscriptTurn(_Model):
    """One conversation turn supplied by the caller (redacted before storage)."""

    role: Literal["customer", "assistant", "agent", "system"]
    text: Annotated[str, StringConstraints(max_length=MAX_TURN_CHARS)]


class EscalationRequest(_Model):
    """Input of ``create_escalation`` / ``POST /escalations``.

    At least one of ``query_id``, ``session_id`` or ``customer_goal`` is
    required. Free text is redacted before it is stored or sent. Where the
    packet goes (sink, webhook URL) is server configuration, never input.
    """

    query_id: uuid.UUID | None = None
    session_id: SessionId | None = None
    customer_goal: Annotated[str, StringConstraints(max_length=1000)] = ""
    reason: Annotated[str, StringConstraints(max_length=500)] = ""
    transcript: list[TranscriptTurn] = Field(default_factory=list, max_length=MAX_TRANSCRIPT_TURNS)

    @model_validator(mode="after")
    def _has_context(self) -> EscalationRequest:
        if self.query_id is None and self.session_id is None and not self.customer_goal.strip():
            raise ValueError("give at least one of query_id, session_id or customer_goal")
        return self


class EscalationResult(_Model):
    """``create_escalation`` / ``get_escalation`` tool result.

    Fields: ``escalation_id``, ``kb_id``, ``status`` (``open`` | ``sent`` |
    ``resolved``; ``sent`` once a forwarding sink accepted it), ``query_id``,
    ``session_id``, ``reasons`` (codes), ``packet`` (the redacted hand-off
    packet, schema ``support.escalation/v1``), ``delivery``
    (:class:`EscalationDelivery`), ``created_at``.
    """

    escalation_id: uuid.UUID
    kb_id: str = DEFAULT_KB_ID
    status: EscalationStatus
    query_id: uuid.UUID | None = None
    session_id: str | None = None
    reasons: list[str]
    packet: dict[str, object]
    delivery: EscalationDelivery | None = None
    created_at: UtcTimestamp


def _assume_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class GapQuery(_Model):
    """Input of ``list_knowledge_gaps``: ``limit`` 1-100 clusters, ``min_count``
    queries per cluster, ``since`` (ISO-8601; naive = UTC; default 30 days ago)."""

    limit: int = Field(default=20, ge=1, le=100)
    min_count: int = Field(default=1, ge=1, le=1000)
    since: Annotated[datetime, AfterValidator(_assume_utc)] | None = None


class KnowledgeGap(_Model):
    """One cluster of similar unanswered / low-confidence / escalated queries.

    ``gap_id`` is a stable hash of the representative query; ``count`` the
    number of logged queries; ``example_queries`` up to 5 distinct (redacted)
    phrasings; ``outcomes`` / ``reasons`` count codes; ``avg_confidence`` the
    mean confidence score; ``first_seen`` / ``last_seen`` ISO-8601 UTC.
    """

    gap_id: str
    representative_query: str
    count: int
    example_queries: list[str]
    outcomes: dict[str, int]
    reasons: dict[str, int]
    avg_confidence: float
    first_seen: UtcTimestamp
    last_seen: UtcTimestamp


class KnowledgeGapList(_Model):
    """``list_knowledge_gaps`` tool result: a ranked content backlog.

    ``gaps`` are ordered by ``count`` then ``last_seen`` (most needed first);
    ``scanned`` is the number of gap-log rows considered since ``since``.
    """

    kb_id: str = DEFAULT_KB_ID
    since: UtcTimestamp
    scanned: int
    gaps: list[KnowledgeGap]


# ── Resolution metrics and the weekly gap report (Phase 6) ──────────────────

DEFAULT_METRICS_WINDOW_DAYS: Final = 7
MAX_METRICS_WINDOW_DAYS: Final = 365
DEFAULT_REPORT_GAPS: Final = 10
RECONTACT_HORIZON_HOURS: Final = 48

DEFAULT_METRICS_ROW_LIMIT: Final = 20_000
"""Default cap on activity rows read per table for one metrics computation."""

RECONTACT_DEFINITION: Final = (
    "Proxy, not a measured resolution rate: the share of eligible sessions in which the "
    "customer asked again in the same session within 48h after a query whose outcome was "
    "answer or answer_with_caveat. Eligible = sessions with such a query inside the window "
    "whose 48h follow-up period has ended or that already asked again; sessions still inside "
    "that period are left out of the rate (counted in observation_incomplete_sessions). "
    "Queries without a session_id are excluded (counted in queries_without_session)."
)
"""Stable text returned with :class:`RecontactMetric` (what the number does and does not mean)."""


class MetricsQuery(_Model):
    """Input of the metrics / report surfaces: ``window_days`` 1-365, ``limit`` 1-100 gaps."""

    window_days: int = Field(default=DEFAULT_METRICS_WINDOW_DAYS, ge=1, le=MAX_METRICS_WINDOW_DAYS)
    limit: int = Field(default=DEFAULT_REPORT_GAPS, ge=1, le=100)


class QueryActivity(_Model):
    """One ``query_log`` row reduced to what the resolution metrics need (no text)."""

    id: uuid.UUID
    session_id: str | None = None
    outcome: Outcome
    confidence: float | None = None
    created_at: UtcTimestamp


class EscalationActivity(_Model):
    """One ``escalations`` row reduced to its status and creation time."""

    status: EscalationStatus
    created_at: UtcTimestamp


class FeedbackActivity(_Model):
    """One ``feedback`` row reduced to its rating; negative = rating <= 2 or not helpful."""

    rating: int
    helpful: bool
    created_at: UtcTimestamp

    @property
    def negative(self) -> bool:
        return self.rating <= 2 or not self.helpful


class OutcomeCounts(_Model):
    """Logged queries per outcome."""

    answer: int = 0
    answer_with_caveat: int = 0
    escalate: int = 0


class RecontactMetric(_Model):
    """The 48-hour re-contact **proxy** for unresolved answers (see ``definition``).

    ``rate`` = ``recontacted_sessions`` / ``eligible_sessions`` (``null`` when
    there are no eligible sessions). ``sessions`` counts distinct session ids
    with any query in the window. ``observation_incomplete_sessions`` are
    sessions with an answer in the window that have not (yet) asked again and
    whose 48h follow-up period had not ended when the metrics were computed —
    their status may still change, so they are **not** in ``eligible_sessions``
    and do not bias the rate downwards.
    """

    horizon_hours: int = RECONTACT_HORIZON_HOURS
    definition: str = RECONTACT_DEFINITION
    sessions: int
    eligible_sessions: int
    recontacted_sessions: int
    rate: float | None
    observation_incomplete_sessions: int
    queries_without_session: int


class ResolutionMetrics(_Model):
    """``get_resolution_metrics`` tool result over ``[since, until)``.

    Rates are fractions in ``0..1`` rounded to 4 decimals, ``null`` when their
    denominator is zero. ``escalation_rate`` = escalate-outcome queries /
    ``total_queries``; ``escalations_created`` counts hand-off records created
    in the window (``escalations_by_status`` splits them). ``avg_confidence``
    is the mean stored confidence score (heuristic, ``confidence_calibrated``
    is false). ``negative_feedback_rate`` = negative ratings / feedback given
    in the window. ``knowledge_gaps`` is the number of gap clusters in the
    window; ``top_gaps`` the largest ones. ``gap_scan_truncated`` is true when
    the gap log held more rows than ``GAP_SCAN_LIMIT`` (oldest ones skipped).
    ``query_scan_truncated`` / ``escalation_scan_truncated`` /
    ``feedback_scan_truncated`` are true when that table held more rows for the
    window than ``METRICS_ROW_LIMIT``: only the newest rows were counted, so
    the totals and rates cover part of the window.
    """

    kb_id: str = DEFAULT_KB_ID
    window_days: int
    since: UtcTimestamp
    until: UtcTimestamp
    total_queries: int
    outcomes: OutcomeCounts
    escalation_rate: float | None
    escalations_created: int
    escalations_by_status: dict[str, int]
    recontact: RecontactMetric
    avg_confidence: float | None
    confidence_samples: int
    confidence_calibrated: bool = False
    feedback_count: int
    negative_feedback: int
    negative_feedback_rate: float | None
    knowledge_gaps: int
    gap_queries_scanned: int
    gap_scan_truncated: bool
    top_gaps: list[KnowledgeGap]
    query_scan_truncated: bool = False
    escalation_scan_truncated: bool = False
    feedback_scan_truncated: bool = False


type TopicAction = Literal["add_faq_entry", "expand_existing_content", "revise_existing_answer"]


class SuggestedTopic(_Model):
    """A content suggestion derived from one knowledge gap.

    ``action``: ``add_faq_entry`` (mostly escalated / no good passage),
    ``expand_existing_content`` (mostly answered with a caveat) or
    ``revise_existing_answer`` (answered, but rated badly). ``topic`` and
    ``example_queries`` are redacted customer phrasings.
    """

    gap_id: str
    topic: str
    queries: int
    action: TopicAction
    example_queries: list[str]
    top_reasons: list[str]


class WeeklyGapReport(_Model):
    """``GET /reports/weekly-gaps`` / ``--mode report`` result.

    ``gaps`` are the ranked clusters (up to ``limit``); ``suggested_topics``
    the content gaps among them — clusters escalated only for policy reasons
    (PII, legal / medical topic, frustration) are excluded and counted in
    ``policy_only_gaps``. ``markdown`` renders the same data for people.
    """

    kb_id: str = DEFAULT_KB_ID
    generated_at: UtcTimestamp
    window_days: int
    since: UtcTimestamp
    until: UtcTimestamp
    metrics: ResolutionMetrics
    gaps: list[KnowledgeGap]
    suggested_topics: list[SuggestedTopic]
    policy_only_gaps: int
    markdown: str
