"""Phase 6 resolution metrics and the weekly knowledge-gap report.

Pure math on hand-built activity rows, the SQLite read path on a seeded temp
database, the Markdown renderer, the trace-layer push, and every surface
(service, MCP tool, REST routes behind auth, ``--mode report`` CLI).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

import observability.tracing as tracing
from api.app import create_app
from bootstrap import assemble_runtime, report_gaps
from faq_core.confidence import Outcome
from faq_core.config import AppConfig
from faq_core.errors import InvalidRequestError
from faq_core.events import METRICS_REPORTED, Event, EventBus
from faq_core.gaps import cluster_embedded, cluster_gaps
from faq_core.metrics import (
    build_weekly_gap_report,
    collect_resolution_metrics,
    compute_resolution_metrics,
    md_text,
    metric_scores,
    recontact_proxy,
    safe_rate,
    suggest_topics,
)
from faq_core.models import (
    Escalation,
    EscalationActivity,
    EscalationStatus,
    FeedbackActivity,
    FeedbackEntry,
    KnowledgeGap,
    QueryActivity,
    QueryLogRecord,
    ResolutionMetrics,
    WeeklyGapReport,
)
from faq_core.repository import FAQRepository
from faq_core.retrieval.embedders import HashingEmbedder
from faq_core.service import SupportPolicy, SupportService
from server import create_mcp_server
from tests.conftest import (
    FakeFAQRepository,
    FakeRAGPipeline,
    bearer,
    make_entry,
    static_lifespan,
    token_security,
)

UNTIL = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
SINCE = UNTIL - timedelta(days=7)
H48 = timedelta(hours=48)


def _q(
    at: datetime,
    outcome: Outcome = Outcome.ANSWER,
    session: str | None = "s1",
    confidence: float | None = 0.8,
) -> QueryActivity:
    return QueryActivity(
        id=uuid.uuid4(), session_id=session, outcome=outcome, confidence=confidence, created_at=at
    )


def _metrics(queries: list[QueryActivity], **kwargs: object) -> ResolutionMetrics:
    values: dict[str, object] = {
        "window_days": 7,
        "since": SINCE,
        "until": UNTIL,
        "as_of": UNTIL + timedelta(days=30),
        "queries": queries,
        "escalations": [],
        "feedback": [],
        "gaps": [],
        "gap_queries_scanned": 0,
    }
    values.update(kwargs)
    return compute_resolution_metrics(**values)  # type: ignore[arg-type]


def _gap(
    query: str,
    count: int = 1,
    outcomes: dict[str, int] | None = None,
    reasons: dict[str, int] | None = None,
) -> KnowledgeGap:
    return KnowledgeGap(
        gap_id=uuid.uuid4().hex[:16],
        representative_query=query,
        count=count,
        example_queries=[query],
        outcomes=outcomes or {"escalate": count},
        reasons=reasons or {"low_confidence": count},
        avg_confidence=0.1,
        first_seen=SINCE + timedelta(days=1),
        last_seen=SINCE + timedelta(days=2),
    )


# ── pure math ────────────────────────────────────────────────────────────────


def test_safe_rate_returns_none_instead_of_dividing_by_zero() -> None:
    assert safe_rate(0, 0) is None
    assert safe_rate(1, 3) == 0.3333


def test_empty_window_has_zero_counts_and_null_rates() -> None:
    metrics = _metrics([])

    assert metrics.total_queries == 0
    assert metrics.escalation_rate is None
    assert metrics.avg_confidence is None
    assert metrics.negative_feedback_rate is None
    assert metrics.recontact.rate is None
    assert metrics.recontact.eligible_sessions == 0
    assert metric_scores(metrics)["support.escalation_rate"] is None


def test_counts_outcomes_escalation_rate_confidence_and_feedback() -> None:
    day = SINCE + timedelta(days=1)
    queries = [
        _q(day, Outcome.ANSWER, confidence=0.9),
        _q(day, Outcome.ANSWER_WITH_CAVEAT, session="s2", confidence=0.4),
        _q(day, Outcome.ESCALATE, session="s3", confidence=0.1),
        _q(day, Outcome.ESCALATE, session=None, confidence=None),
    ]
    escalations = [
        EscalationActivity(status=EscalationStatus.OPEN, created_at=day),
        EscalationActivity(status=EscalationStatus.SENT, created_at=day),
        EscalationActivity(status=EscalationStatus.OPEN, created_at=SINCE - timedelta(seconds=1)),
    ]
    feedback = [
        FeedbackActivity(rating=5, helpful=True, created_at=day),
        FeedbackActivity(rating=2, helpful=True, created_at=day),
        FeedbackActivity(rating=4, helpful=False, created_at=day),
        FeedbackActivity(rating=1, helpful=False, created_at=UNTIL),  # outside (exclusive end)
    ]

    metrics = _metrics(queries, escalations=escalations, feedback=feedback)

    assert metrics.total_queries == 4
    assert metrics.outcomes.model_dump() == {"answer": 1, "answer_with_caveat": 1, "escalate": 2}
    assert metrics.escalation_rate == 0.5
    assert metrics.escalations_created == 2
    assert metrics.escalations_by_status == {"open": 1, "sent": 1}
    assert metrics.avg_confidence == pytest.approx(0.4667, abs=1e-4)
    assert metrics.confidence_samples == 3
    assert metrics.confidence_calibrated is False
    assert (metrics.feedback_count, metrics.negative_feedback) == (3, 2)
    assert metrics.negative_feedback_rate == 0.6667


def test_window_is_half_open_since_inclusive_until_exclusive() -> None:
    queries = [
        _q(SINCE - timedelta(microseconds=1), session=None),
        _q(SINCE, session=None),
        _q(UNTIL - timedelta(microseconds=1), session=None),
        _q(UNTIL, session=None),
    ]

    metrics = _metrics(queries)

    assert metrics.total_queries == 2
    assert metrics.recontact.queries_without_session == 2


def test_sessions_without_session_id_are_excluded_and_counted() -> None:
    day = SINCE + timedelta(days=1)
    metrics = _metrics([_q(day, session=None), _q(day + timedelta(hours=1), session=None)])

    assert metrics.recontact.sessions == 0
    assert metrics.recontact.eligible_sessions == 0
    assert metrics.recontact.rate is None
    assert metrics.recontact.queries_without_session == 2


@pytest.mark.parametrize(
    ("gap", "recontacted"),
    [
        (timedelta(minutes=5), True),
        (H48, True),  # exactly 48h counts (inclusive)
        (H48 + timedelta(seconds=1), False),
    ],
)
def test_recontact_boundary_at_exactly_48_hours(gap: timedelta, recontacted: bool) -> None:
    anchor = SINCE + timedelta(days=1)

    metric = recontact_proxy(
        [_q(anchor), _q(anchor + gap, Outcome.ESCALATE)],
        since=SINCE,
        until=UNTIL,
        as_of=UNTIL + timedelta(days=30),
    )

    assert metric.eligible_sessions == 1
    assert metric.recontacted_sessions == int(recontacted)
    assert metric.rate == (1.0 if recontacted else 0.0)


def test_recontact_counts_sessions_not_queries_and_ignores_escalate_anchors() -> None:
    day = SINCE + timedelta(days=1)
    queries = [
        _q(day, session="a"),
        _q(day + timedelta(hours=1), session="a"),
        _q(day + timedelta(hours=2), session="a"),  # still one session
        _q(day, Outcome.ANSWER_WITH_CAVEAT, session="b"),  # never followed up
        _q(day, Outcome.ESCALATE, session="c"),  # escalated: not an answer anchor
        _q(day + timedelta(hours=1), Outcome.ESCALATE, session="c"),
    ]

    metric = recontact_proxy(queries, since=SINCE, until=UNTIL, as_of=UNTIL + H48)

    assert (metric.sessions, metric.eligible_sessions, metric.recontacted_sessions) == (3, 2, 1)
    assert metric.rate == 0.5


def test_follow_up_after_the_window_end_still_counts_but_anchor_must_be_inside() -> None:
    late_anchor = UNTIL - timedelta(hours=1)
    queries = [
        _q(late_anchor, session="in"),
        _q(UNTIL + timedelta(hours=2), session="in"),  # follow-up past `until`
        _q(UNTIL + timedelta(hours=3), session="out"),  # anchor outside the window
        _q(UNTIL + timedelta(hours=4), session="out"),
    ]

    metric = recontact_proxy(queries, since=SINCE, until=UNTIL, as_of=UNTIL + H48)

    assert (metric.sessions, metric.eligible_sessions, metric.recontacted_sessions) == (1, 1, 1)


def test_sessions_whose_48h_period_is_still_open_are_reported_not_rated() -> None:
    metric = recontact_proxy(
        [_q(UNTIL - timedelta(hours=10))], since=SINCE, until=UNTIL, as_of=UNTIL
    )

    assert metric.eligible_sessions == 0  # outcome unknown yet: not in the denominator
    assert metric.recontacted_sessions == 0
    assert metric.rate is None
    assert metric.observation_incomplete_sessions == 1


def test_open_sessions_do_not_bias_the_rate_downwards() -> None:
    day = SINCE + timedelta(days=1)
    queries = [
        _q(day, session="closed-recontacted"),
        _q(day + timedelta(hours=1), session="closed-recontacted"),
        _q(day, session="closed-resolved"),
        _q(UNTIL - timedelta(hours=3), session="open-a"),  # 48h period still running
        _q(UNTIL - timedelta(hours=2), session="open-b"),
        _q(UNTIL - timedelta(hours=2), session="open-recontacted"),
        _q(UNTIL - timedelta(hours=1), session="open-recontacted"),  # already asked again
    ]

    metric = recontact_proxy(queries, since=SINCE, until=UNTIL, as_of=UNTIL)

    assert (metric.eligible_sessions, metric.recontacted_sessions) == (3, 2)
    assert metric.observation_incomplete_sessions == 2
    assert metric.rate == 0.6667  # 2 of 3, not 2 of 5


# ── suggestions and the Markdown report ──────────────────────────────────────


def test_suggested_topics_pick_an_action_and_skip_policy_only_gaps() -> None:
    gaps = [
        _gap("how do I export invoices", 3),
        _gap("refund for damaged item", 2, outcomes={"answer_with_caveat": 2}),
        _gap("delivery times", 1, outcomes={"answer": 1}, reasons={}),
        _gap("my lawyer says", 2, reasons={"legal_topic": 2}),
        _gap("lawyer and no answer", 1, reasons={"legal_topic": 1, "low_confidence": 1}),
    ]

    topics, policy_only = suggest_topics(gaps)

    assert policy_only == 1
    assert [t.action for t in topics] == [
        "add_faq_entry",
        "expand_existing_content",
        "revise_existing_answer",
        "add_faq_entry",
    ]
    assert topics[-1].top_reasons == ["low_confidence"]  # policy reasons dropped


@pytest.mark.parametrize(
    "hostile",
    [
        "refund form at https://evil.example/login now",
        "see www.phish.example for help",
        "HTTP://Evil.Example/x?y=1",
        "mail me at someone@evil.example",
        "go to evil.example/reset",
    ],
)
def test_md_text_breaks_autolinks(hostile: str) -> None:
    escaped = md_text(hostile)

    assert "://" not in escaped
    assert "www." not in escaped.lower()
    assert "evil.example" not in escaped.lower() and "phish.example" not in escaped
    assert "@evil" not in escaped


def test_md_text_defangs_visibly_and_leaves_plain_words_alone() -> None:
    assert md_text("https://evil.example/login") == "https\\[:\\]//evil\\[.\\]example/login"
    assert md_text("how do I export invoices v2.0 e.g. today") == (
        "how do I export invoices v2.0 e.g. today"
    )


def test_md_text_neutralises_markdown_and_layout_injection() -> None:
    hostile = "![x](https://tracker.invalid/p.png)\n# heading | cell <b>"

    escaped = md_text(hostile)

    assert "\n" not in escaped
    assert "\\!\\[x\\]\\(" in escaped
    assert "\\#" in escaped and "\\|" in escaped and "\\<b\\>" in escaped


def test_report_renders_metrics_gaps_and_suggestions() -> None:
    gaps = [_gap("export [EMAIL] invoices", 3), _gap("my lawyer", 1, reasons={"legal_topic": 1})]
    metrics = _metrics([_q(SINCE + timedelta(days=1))], gaps=gaps, gap_queries_scanned=4)

    report = build_weekly_gap_report(metrics, gaps, generated_at=UNTIL, limit=10)

    assert isinstance(report, WeeklyGapReport)
    assert report.policy_only_gaps == 1
    assert [t.topic for t in report.suggested_topics] == ["export [EMAIL] invoices"]
    md = report.markdown
    assert md.startswith("# Knowledge-gap report — default\n")
    assert "| Queries | 1 |" in md
    assert "| 48h re-contact (proxy) | 0.0% (0 of 1 sessions) |" in md
    assert "Proxy, not a measured resolution rate" in md
    assert "1. **export \\[EMAIL\\] invoices** — 3 queries" in md
    assert "- [ ] **export \\[EMAIL\\] invoices** (3) — add faq entry" in md
    assert "1 gap cluster(s) were escalated only for policy reasons" in md


def test_report_for_an_empty_window_says_so() -> None:
    report = build_weekly_gap_report(_metrics([]), [], generated_at=UNTIL)

    assert "_No knowledge gaps in this window._" in report.markdown
    assert "| Escalated (outcome) | 0 (n/a) |" in report.markdown
    assert report.gaps == [] and report.suggested_topics == []


def test_report_limit_caps_the_gap_list() -> None:
    gaps = [_gap(f"topic {n}") for n in range(5)]

    report = build_weekly_gap_report(_metrics([], gaps=gaps), gaps, generated_at=UNTIL, limit=2)

    assert len(report.gaps) == 2
    assert report.metrics.knowledge_gaps == 5


# ── SQLite read path (seeded temp database) ──────────────────────────────────


def _record(
    at: datetime,
    outcome: Outcome = Outcome.ANSWER,
    session: str | None = "s1",
    query: str = "how do I export invoices",
    kb_id: str = "default",
) -> QueryLogRecord:
    return QueryLogRecord(
        kb_id=kb_id,
        query=query,
        surface="search_kb",
        top_k=3,
        confidence=0.2 if outcome is Outcome.ESCALATE else 0.8,
        session_id=session,
        outcome=outcome,
        reasons=["low_confidence"] if outcome is Outcome.ESCALATE else [],
        created_at=at,
    )


async def test_collect_metrics_from_a_seeded_sqlite_database(sqlite_repo: FAQRepository) -> None:
    day = SINCE + timedelta(days=2)
    rows = [
        _record(day, session="a"),
        _record(day + H48, Outcome.ESCALATE, session="a", query="export invoices to csv"),
        _record(day, Outcome.ANSWER_WITH_CAVEAT, session="b"),
        _record(day, Outcome.ESCALATE, session=None, query="how do I export invoices"),
        _record(SINCE - timedelta(seconds=1), Outcome.ESCALATE),  # before the window
        _record(day, Outcome.ESCALATE, kb_id="other"),  # other knowledge base
    ]
    for row in rows:
        await sqlite_repo.log_query(row)
    await sqlite_repo.create_escalation(Escalation(reason="requested", created_at=day))
    await sqlite_repo.save_feedback(
        FeedbackEntry(query_id=rows[0].id, rating=1, helpful=False, created_at=day)
    )

    metrics, gaps = await collect_resolution_metrics(
        sqlite_repo, HashingEmbedder(), kb_id="default", window_days=7, until=UNTIL
    )

    assert metrics.total_queries == 4
    assert metrics.outcomes.model_dump() == {"answer": 1, "answer_with_caveat": 1, "escalate": 2}
    assert metrics.escalation_rate == 0.5
    assert metrics.escalations_created == 1
    assert metrics.recontact.model_dump(include={"sessions", "eligible_sessions"}) == {
        "sessions": 2,
        "eligible_sessions": 2,
    }
    assert metrics.recontact.recontacted_sessions == 1  # exactly 48h later
    assert metrics.recontact.queries_without_session == 1
    assert (metrics.feedback_count, metrics.negative_feedback_rate) == (1, 1.0)
    # gap log: 2 escalations + the caveat answer + the badly rated answer, in window
    assert metrics.gap_queries_scanned == 4
    assert metrics.knowledge_gaps == len(gaps) >= 1
    assert all(gap.count >= 1 for gap in metrics.top_gaps)


async def test_repository_activity_reads_respect_kb_and_window(sqlite_repo: FAQRepository) -> None:
    inside = SINCE + timedelta(hours=1)
    await sqlite_repo.log_query(_record(inside))
    await sqlite_repo.log_query(_record(UNTIL))  # exclusive end
    await sqlite_repo.log_query(_record(inside, kb_id="other"))

    activity = await sqlite_repo.list_query_activity("default", SINCE, UNTIL, 100)
    escalations = await sqlite_repo.list_escalation_activity("default", SINCE, UNTIL, 100)
    feedback = await sqlite_repo.list_feedback_activity("default", SINCE, UNTIL, 100)

    assert [a.created_at for a in activity] == [inside]
    assert escalations == [] and feedback == []


async def test_activity_reads_are_capped_and_keep_the_newest_rows(
    sqlite_repo: FAQRepository,
) -> None:
    times = [SINCE + timedelta(hours=hour) for hour in range(1, 6)]
    for at in times:
        await sqlite_repo.log_query(_record(at))
        await sqlite_repo.create_escalation(Escalation(reason="requested", created_at=at))
        await sqlite_repo.save_feedback(
            FeedbackEntry(query_id=uuid.uuid4(), rating=1, created_at=at)
        )

    queries = await sqlite_repo.list_query_activity("default", SINCE, UNTIL, 3)
    escalations = await sqlite_repo.list_escalation_activity("default", SINCE, UNTIL, 3)
    feedback = await sqlite_repo.list_feedback_activity("default", SINCE, UNTIL, 3)

    for rows in (queries, escalations, feedback):
        assert [row.created_at for row in rows] == times[-3:]  # newest 3, oldest first


async def test_collect_flags_truncated_activity_and_gap_scans(sqlite_repo: FAQRepository) -> None:
    for hour in range(1, 5):
        at = SINCE + timedelta(hours=hour)
        await sqlite_repo.log_query(_record(at, Outcome.ESCALATE, query=f"topic {hour}"))
        await sqlite_repo.create_escalation(Escalation(reason="requested", created_at=at))

    capped, _ = await collect_resolution_metrics(
        sqlite_repo,
        HashingEmbedder(),
        kb_id="default",
        window_days=7,
        until=UNTIL,
        row_limit=3,
        gap_scan_limit=4,
    )
    exact, _ = await collect_resolution_metrics(
        sqlite_repo, HashingEmbedder(), kb_id="default", window_days=7, until=UNTIL, row_limit=4
    )

    assert capped.query_scan_truncated and capped.escalation_scan_truncated
    assert not capped.feedback_scan_truncated
    assert capped.total_queries == 3 and capped.escalations_created == 3
    assert capped.gap_scan_truncated is False  # exactly GAP_SCAN_LIMIT rows: nothing skipped
    assert not exact.query_scan_truncated and exact.total_queries == 4
    report = build_weekly_gap_report(capped, [], generated_at=UNTIL)
    assert "Partial window: only the newest rows of queries, escalations" in report.markdown


# ── service ──────────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_store() -> FakeFAQRepository:
    store = FakeFAQRepository()
    now = datetime.now(UTC)
    store.queries = [
        _record(now - timedelta(hours=5), session="x"),
        _record(now - timedelta(hours=4), Outcome.ESCALATE, session="x"),
        _record(now - timedelta(days=9), Outcome.ESCALATE),  # outside the 7-day window
    ]
    return store


@pytest.fixture
def metrics_service(seeded_store: FakeFAQRepository, event_bus: EventBus) -> SupportService:
    return SupportService(
        engine=FakeRAGPipeline(sources=[make_entry()]), store=seeded_store, events=event_bus
    )


@pytest.fixture
def telemetry(
    event_bus: EventBus, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, float | None]], list[Event]]:
    """Capture trace-score pushes and ``metrics.reported`` events."""
    pushed: list[dict[str, float | None]] = []
    monkeypatch.setattr(
        "faq_core.metrics.push_metric_scores", lambda scores, **_kw: pushed.append(dict(scores))
    )
    seen: list[Event] = []

    async def _record_event(event: Event) -> None:
        seen.append(event)

    event_bus.subscribe(METRICS_REPORTED, _record_event)
    return pushed, seen


async def test_service_metrics_read_has_no_telemetry_side_effects(
    metrics_service: SupportService,
    event_bus: EventBus,
    telemetry: tuple[list[dict[str, float | None]], list[Event]],
) -> None:
    metrics = await metrics_service.resolution_metrics()
    await event_bus.drain()

    assert metrics.window_days == 7
    assert metrics.total_queries == 2
    assert metrics.recontact.recontacted_sessions == 1
    assert telemetry == ([], [])  # GET /metrics/resolution and the MCP tool stay side-effect free


async def test_weekly_report_pushes_scores_and_emits_an_event(
    metrics_service: SupportService,
    event_bus: EventBus,
    telemetry: tuple[list[dict[str, float | None]], list[Event]],
) -> None:
    pushed, seen = telemetry

    await metrics_service.weekly_gap_report()
    await event_bus.drain()

    assert pushed[0]["support.escalation_rate"] == 0.5
    assert seen[0].payload["total_queries"] == 2
    assert seen[0].source == "weekly_gap_report"
    assert "query" not in seen[0].payload  # aggregates only


class _CountingStore(FakeFAQRepository):
    def __init__(self) -> None:
        super().__init__()
        self.activity_reads = 0

    async def list_query_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[QueryActivity]:
        self.activity_reads += 1
        return await super().list_query_activity(kb_id, since, until, limit)


async def test_metrics_are_cached_per_window_within_the_ttl(event_bus: EventBus) -> None:
    store = _CountingStore()
    service = SupportService(engine=FakeRAGPipeline(), store=store, events=event_bus)

    first = await service.resolution_metrics(7)
    again = await service.resolution_metrics(7)
    report = await service.weekly_gap_report(7, limit=3)
    other_window = await service.resolution_metrics(30)
    service.clear_answer_cache()  # cleared with the answer cache (KB changed)
    fresh = await service.resolution_metrics(7)

    assert again is first and report.metrics is first
    assert other_window is not first and fresh is not first
    assert store.activity_reads == 3


async def test_metrics_cache_can_be_disabled(event_bus: EventBus) -> None:
    store = _CountingStore()
    service = SupportService(
        engine=FakeRAGPipeline(),
        store=store,
        events=event_bus,
        policy=SupportPolicy(metrics_cache_ttl=0),
    )

    await service.resolution_metrics(7)
    await service.resolution_metrics(7)

    assert store.activity_reads == 2


async def test_gap_clustering_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InstantEmbedder(HashingEmbedder):
        async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [self.embed_sync(text) for text in texts]  # no await: stays on the loop

    real_cluster = cluster_embedded

    def _slow_cluster(*args: object, **kwargs: object) -> object:
        time.sleep(0.3)  # CPU-bound stand-in: would freeze the loop if run on it
        return real_cluster(*args, **kwargs)

    monkeypatch.setattr("faq_core.gaps.cluster_embedded", _slow_cluster)
    ticks = 0
    done = asyncio.Event()

    async def _ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(_ticker())
    rows = [_record(SINCE + timedelta(hours=n), Outcome.ESCALATE, query=f"q {n}") for n in range(5)]
    gaps = await cluster_gaps(rows, _InstantEmbedder())
    done.set()
    await ticker

    assert len(gaps) >= 1
    assert ticks >= 10  # the loop kept running other coroutines during clustering


async def test_service_weekly_report_includes_gaps(metrics_service: SupportService) -> None:
    report = await metrics_service.weekly_gap_report(window_days=7, limit=5)

    assert report.metrics.knowledge_gaps == 1
    assert report.gaps[0].representative_query == "how do I export invoices"
    assert report.suggested_topics[0].action == "add_faq_entry"
    assert "how do I export invoices" in report.markdown


@pytest.mark.parametrize(
    ("kwargs"), [{"window_days": 0}, {"window_days": 366}, {"limit": 0}, {"limit": 101}]
)
async def test_service_rejects_out_of_range_input(
    metrics_service: SupportService, kwargs: dict[str, int]
) -> None:
    with pytest.raises(InvalidRequestError):
        await metrics_service.weekly_gap_report(**kwargs)


# ── trace layer ──────────────────────────────────────────────────────────────


class _FakeSpan:
    def __init__(self, fail: bool) -> None:
        self.scores: dict[str, float] = {}
        self.ended = False
        self._fail = fail

    def score_trace(self, *, name: str, value: float) -> None:
        if self._fail:
            raise RuntimeError("network down")
        self.scores[name] = value

    def end(self) -> None:
        self.ended = True


class _FakeLangfuse:
    def __init__(self, fail: bool = False) -> None:
        self.span = _FakeSpan(fail)
        self.metadata: dict[str, object] = {}

    def start_observation(
        self, *, name: str, as_type: str, metadata: dict[str, object]
    ) -> _FakeSpan:
        self.metadata = metadata
        return self.span


def test_push_metric_scores_is_a_no_op_when_tracing_is_disabled() -> None:
    assert tracing.push_metric_scores({"support.total_queries": 3.0}) == 0
    tracing.flush_tracing()  # must not raise


def test_push_metric_scores_skips_nulls_and_ends_the_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_langfuse_get_client", lambda: client, raising=False)

    pushed = tracing.push_metric_scores(
        {"support.total_queries": 3, "support.escalation_rate": None}, metadata={"kb_id": "x"}
    )

    assert pushed == 1
    assert client.span.scores == {"support.total_queries": 3.0}
    assert client.span.ended is True
    assert client.metadata == {"kb_id": "x"}


def test_push_metric_scores_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeLangfuse(fail=True)
    monkeypatch.setattr(tracing, "_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_langfuse_get_client", lambda: client, raising=False)

    assert tracing.push_metric_scores({"support.total_queries": 1.0}) == 0
    assert client.span.ended is True


# ── MCP tool ─────────────────────────────────────────────────────────────────


async def test_get_resolution_metrics_tool(metrics_service: SupportService) -> None:
    server = create_mcp_server(static_lifespan(metrics_service), operator_tools=True)
    async with create_connected_server_and_client_session(server) as session:
        ok = await session.call_tool("get_resolution_metrics", {"window_days": 7})
        bad = await session.call_tool("get_resolution_metrics", {"window_days": 0})

    assert ok.isError is False
    payload = ok.structuredContent
    assert payload is not None
    assert payload["total_queries"] == 2
    assert payload["recontact"]["definition"].startswith("Proxy")
    assert payload["escalation_rate"] == 0.5
    assert bad.isError is True
    content = bad.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text)["error"]["code"] == "invalid_request"


# ── REST routes ──────────────────────────────────────────────────────────────


@pytest.fixture
def rest(metrics_service: SupportService, api_token: str) -> Iterator[TestClient]:
    app = create_app(static_lifespan(metrics_service), security=token_security(api_token))
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("path", ["/metrics/resolution", "/reports/weekly-gaps"])
def test_metrics_routes_require_auth(rest: TestClient, path: str) -> None:
    response = rest.get(path)

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_resolution_metrics_route(rest: TestClient, api_token: str) -> None:
    ok = rest.get("/metrics/resolution", params={"window_days": 7}, headers=bearer(api_token))
    bad = rest.get("/metrics/resolution", params={"window_days": 0}, headers=bearer(api_token))

    assert ok.status_code == 200
    assert ok.json()["total_queries"] == 2
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"


def test_weekly_gaps_route(rest: TestClient, api_token: str) -> None:
    response = rest.get("/reports/weekly-gaps", params={"limit": 3}, headers=bearer(api_token))

    assert response.status_code == 200
    body = response.json()
    assert body["markdown"].startswith("# Knowledge-gap report")
    assert body["metrics"]["window_days"] == 7
    assert body["suggested_topics"][0]["action"] == "add_faq_entry"


# ── CLI (--mode report) ──────────────────────────────────────────────────────


@pytest.fixture
def sample_report() -> WeeklyGapReport:
    gaps = [_gap("export invoices", 2)]
    return build_weekly_gap_report(_metrics([], gaps=gaps), gaps, generated_at=UNTIL)


def test_report_cli_prints_markdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_report: WeeklyGapReport,
) -> None:
    import app

    calls: list[tuple[int, int]] = []

    async def _report(window_days: int, limit: int) -> WeeklyGapReport:
        calls.append((window_days, limit))
        return sample_report

    monkeypatch.setattr("bootstrap.run_report", _report)

    assert app.run_report(14, 3, None) == 0
    assert capsys.readouterr().out == sample_report.markdown
    assert calls == [(14, 3)]


def test_report_cli_writes_the_output_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_report: WeeklyGapReport,
    tmp_path: Path,
) -> None:
    import app

    async def _report(window_days: int, limit: int) -> WeeklyGapReport:
        return sample_report

    monkeypatch.setattr("bootstrap.run_report", _report)
    target = tmp_path / "gaps.md"

    assert app.run_report(7, 10, str(target)) == 0
    assert target.read_text(encoding="utf-8") == sample_report.markdown
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(target) and summary["knowledge_gaps"] == 1

    assert app.run_report(7, 10, str(tmp_path / "missing" / "gaps.md")) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "report_write_failed"


def test_report_validates_input_before_building_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bootstrap

    def _never(_config: AppConfig) -> object:
        raise AssertionError("the runtime must not be built for invalid input")

    monkeypatch.setattr(bootstrap, "build_runtime", _never)

    with pytest.raises(InvalidRequestError):
        asyncio.run(bootstrap.run_report(0, 10))


def test_report_cli_maps_config_errors_to_exit_2_without_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import app

    async def _bad_setting(window_days: int, limit: int) -> WeeklyGapReport:
        AppConfig(_env_file=None, GAP_SCAN_LIMIT="lots-of-rows-secretish")  # type: ignore[call-arg, arg-type]
        raise AssertionError("unreachable")

    monkeypatch.setattr("bootstrap.run_report", _bad_setting)

    assert app.run_report(7, 10, None) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "config_error"
    assert "GAP_SCAN_LIMIT" in error["message"]
    assert "lots-of-rows-secretish" not in error["message"]


def test_report_cli_maps_workflow_value_errors_to_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import app

    async def _invalid(window_days: int, limit: int) -> WeeklyGapReport:
        raise ValueError("GAP_SCAN_LIMIT must be >= 1")

    monkeypatch.setattr("bootstrap.run_report", _invalid)

    assert app.run_report(7, 10, None) == 2
    assert json.loads(capsys.readouterr().err)["error"] == {
        "code": "config_error",
        "message": "GAP_SCAN_LIMIT must be >= 1",
    }


def test_preflight_reports_a_malformed_setting_as_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import app
    from faq_core.config import get_config

    monkeypatch.setenv("GAP_SCAN_LIMIT", "not-a-number")
    get_config.cache_clear()
    try:
        assert app.preflight() == 2
    finally:
        get_config.cache_clear()
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "config_error" and "not-a-number" not in error["message"]


def test_report_cli_maps_invalid_input_to_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import app

    async def _report(window_days: int, limit: int) -> WeeklyGapReport:
        raise InvalidRequestError("invalid request")

    monkeypatch.setattr("bootstrap.run_report", _report)

    assert app.run_report(0, 10, None) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "argv",
    [
        ["app.py", "--mode", "mcp", "--window-days", "3"],
        ["app.py", "--mode", "ingest", "--source", "x.md", "--output", "r.md"],
    ],
)
def test_report_options_are_rejected_outside_report_mode(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    import app

    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        app.main()

    assert exit_info.value.code == 2


def test_report_mode_dispatches_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import app

    calls: list[tuple[int, int, str | None]] = []
    monkeypatch.setattr("sys.argv", ["app.py", "--mode", "report", "--window-days", "0"])
    monkeypatch.setattr(app, "preflight", lambda _host=None: 0)
    monkeypatch.setattr(app, "run_report", lambda *args: calls.append(args) or 0)

    with pytest.raises(SystemExit) as exit_info:
        app.main()

    assert exit_info.value.code == 0
    assert calls == [(0, 10, None)]  # 0 is passed through for validation, not defaulted


async def test_report_gaps_runs_through_the_bootstrap_runtime(
    app_config: AppConfig, sqlite_repo: FAQRepository
) -> None:
    await sqlite_repo.log_query(
        _record(datetime.now(UTC) - timedelta(hours=1), Outcome.ESCALATE, session=None)
    )
    runtime = assemble_runtime(app_config, FakeRAGPipeline(), sqlite_repo)

    report = await report_gaps(runtime, 7, 10)

    assert report.metrics.total_queries == 1
    assert report.metrics.escalation_rate == 1.0
    assert sqlite_repo._conn is None  # closed afterwards


@pytest.mark.parametrize(
    ("setting", "value"), [("METRICS_ROW_LIMIT", 0), ("METRICS_CACHE_TTL", -1)]
)
def test_policy_rejects_invalid_metrics_settings(
    app_config: AppConfig, setting: str, value: int
) -> None:
    with pytest.raises(ValueError, match=setting):
        SupportPolicy.from_config(app_config.model_copy(update={setting: value}))


def test_policy_reads_metrics_settings(app_config: AppConfig) -> None:
    policy = SupportPolicy.from_config(
        app_config.model_copy(update={"METRICS_ROW_LIMIT": 50, "METRICS_CACHE_TTL": 5.0})
    )

    assert (policy.metrics_row_limit, policy.metrics_cache_ttl) == (50, 5.0)
