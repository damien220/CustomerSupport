"""Resolution metrics and the weekly knowledge-gap report (plan §4, Phase 6).

Everything here is computed from what the support workflow already stores —
``query_log`` (outcome, confidence, session id), ``escalations`` and
``feedback`` — over a half-open window ``[since, until)`` of one knowledge
base. The math lives in pure functions (:func:`compute_resolution_metrics`,
:func:`recontact_proxy`, :func:`suggest_topics`,
:func:`render_weekly_gap_report`) so it can be tested without a database;
:func:`collect_resolution_metrics` is the only coroutine and only reads.

**48h re-contact is a proxy.** There is no customer identity and no ticket
state, only the caller-supplied ``session_id``. A session counts as
re-contacted when, after a query whose outcome was ``answer`` or
``answer_with_caveat``, the same session asked again within 48 hours
(inclusive) — the answer probably did not resolve the problem. A legitimate
unrelated follow-up question also counts, and a customer who opens a new
session is missed. Queries without a ``session_id`` cannot be followed and are
excluded (reported separately). Sessions whose 48h period is still open and
that have not asked again yet are left out of the rate (reported as
``observation_incomplete_sessions``), so a fresh window is not biased towards
"resolved". Rates whose denominator is zero are ``None``.

**Bounded reads.** Each activity table is read with a row cap
(``METRICS_ROW_LIMIT``, newest rows kept); when a table holds more rows for the
window the result says so (``*_scan_truncated``) instead of loading it all.

Query text in gaps and the report comes from the query log, which is stored
PII-redacted; nothing here un-redacts or re-reads raw input.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final

from faq_core.confidence import Outcome
from faq_core.events import build_metrics_reported_event
from faq_core.gaps import DEFAULT_CLUSTER_THRESHOLD, cluster_gaps
from faq_core.models import (
    DEFAULT_KB_ID,
    DEFAULT_METRICS_ROW_LIMIT,
    DEFAULT_REPORT_GAPS,
    RECONTACT_HORIZON_HOURS,
    OutcomeCounts,
    RecontactMetric,
    ResolutionMetrics,
    SuggestedTopic,
    TopicAction,
    WeeklyGapReport,
)
from observability.tracing import push_metric_scores

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from faq_core.events import EventBus
    from faq_core.models import (
        EscalationActivity,
        FeedbackActivity,
        KnowledgeGap,
        QueryActivity,
    )
    from faq_core.ports import MetricsStore
    from faq_core.retrieval.embedders import Embedder

RECONTACT_HORIZON: Final = timedelta(hours=RECONTACT_HORIZON_HOURS)
DEFAULT_TOP_GAPS: Final = 5

RESOLVING_OUTCOMES: Final = frozenset({Outcome.ANSWER, Outcome.ANSWER_WITH_CAVEAT})
"""Outcomes after which the customer was given an answer (re-contact anchors)."""

POLICY_REASONS: Final = frozenset({"pii_detected", "legal_topic", "medical_topic", "frustration"})
"""Escalation reasons that say nothing about missing content."""

_RATE_DIGITS: Final = 4
_MD_TEXT_CHARS: Final = 160
_MD_EXAMPLES: Final = 3
_MD_SPECIAL: Final = re.compile(r"([\\`*_{}\[\]()<>#+!|~])")
_WHITESPACE: Final = re.compile(r"\s+")
_LINK_LIKE: Final = re.compile(
    r"\S*(?:://|www\.|\w@\w|(?:[a-z0-9-]+\.)+[a-z]{2,})\S*", re.IGNORECASE
)
"""A whitespace-free token a Markdown viewer could turn into a link (URL, www., e-mail, domain)."""


# ── pure math ────────────────────────────────────────────────────────────────


def window_bounds(window_days: int, until: datetime) -> tuple[datetime, datetime]:
    """Return ``(since, until)`` for a window of *window_days* ending at *until*."""
    return until - timedelta(days=window_days), until


def safe_rate(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator`` rounded to 4 decimals, ``None`` when nothing to divide by."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, _RATE_DIGITS)


def count_outcomes(queries: Iterable[QueryActivity]) -> OutcomeCounts:
    counts = Counter(query.outcome for query in queries)
    return OutcomeCounts(
        answer=counts[Outcome.ANSWER],
        answer_with_caveat=counts[Outcome.ANSWER_WITH_CAVEAT],
        escalate=counts[Outcome.ESCALATE],
    )


def recontact_proxy(
    queries: Sequence[QueryActivity],
    *,
    since: datetime,
    until: datetime,
    as_of: datetime,
    horizon: timedelta = RECONTACT_HORIZON,
) -> RecontactMetric:
    """Compute the 48h re-contact proxy (see the module docstring).

    Args:
        queries: Logged queries from ``since`` up to ``until + horizon`` (later
            ones may be follow-ups of in-window answers), oldest first; rows
            with equal timestamps keep their given order.
        since: Window start (inclusive). Anchors must lie in ``[since, until)``.
        until: Window end (exclusive).
        as_of: When the metrics are computed; decides whether a session's
            follow-up period is still open.
        horizon: Re-contact period after an anchor (inclusive).
    """
    in_window = [q for q in queries if since <= q.created_at < until]
    by_session: defaultdict[str, list[QueryActivity]] = defaultdict(list)
    for query in queries:
        if query.session_id is not None and since <= query.created_at < until + horizon:
            by_session[query.session_id].append(query)
    sessions = {q.session_id for q in in_window if q.session_id is not None}

    eligible = recontacted = incomplete = 0
    for session_id in sessions:
        rows = sorted(by_session[session_id], key=lambda q: q.created_at)  # stable
        anchors = [
            index
            for index, row in enumerate(rows)
            if row.outcome in RESOLVING_OUTCOMES and since <= row.created_at < until
        ]
        if not anchors:
            continue
        # Rows are sorted, so the next row is the earliest possible follow-up.
        if any(
            index + 1 < len(rows) and rows[index + 1].created_at - rows[index].created_at <= horizon
            for index in anchors
        ):
            recontacted += 1
        elif rows[anchors[-1]].created_at + horizon > as_of:
            # Still inside its 48h period: the outcome is unknown, so leave it
            # out of the denominator instead of counting it as "resolved".
            incomplete += 1
            continue
        eligible += 1

    return RecontactMetric(
        horizon_hours=int(horizon.total_seconds() // 3600),
        sessions=len(sessions),
        eligible_sessions=eligible,
        recontacted_sessions=recontacted,
        rate=safe_rate(recontacted, eligible),
        observation_incomplete_sessions=incomplete,
        queries_without_session=sum(1 for q in in_window if q.session_id is None),
    )


def compute_resolution_metrics(
    *,
    window_days: int,
    since: datetime,
    until: datetime,
    as_of: datetime,
    queries: Sequence[QueryActivity],
    escalations: Sequence[EscalationActivity],
    feedback: Sequence[FeedbackActivity],
    gaps: Sequence[KnowledgeGap],
    gap_queries_scanned: int,
    gap_scan_truncated: bool = False,
    top_n: int = DEFAULT_TOP_GAPS,
    kb_id: str = DEFAULT_KB_ID,
    truncated: frozenset[str] = frozenset(),
) -> ResolutionMetrics:
    """Combine the window's activity into :class:`ResolutionMetrics`.

    *queries* may extend past *until* (re-contact follow-ups); only rows in
    ``[since, until)`` count towards the totals. *escalations* and *feedback*
    are filtered to the window too, so callers may pass wider lists.
    *truncated* names the activity tables (``queries``, ``escalations``,
    ``feedback``) whose read hit the row cap.
    """
    in_window = [q for q in queries if since <= q.created_at < until]
    outcomes = count_outcomes(in_window)
    confidences = [q.confidence for q in in_window if q.confidence is not None]
    created = [e for e in escalations if since <= e.created_at < until]
    rated = [f for f in feedback if since <= f.created_at < until]
    negative = sum(1 for f in rated if f.negative)
    return ResolutionMetrics(
        kb_id=kb_id,
        window_days=window_days,
        since=since,
        until=until,
        total_queries=len(in_window),
        outcomes=outcomes,
        escalation_rate=safe_rate(outcomes.escalate, len(in_window)),
        escalations_created=len(created),
        escalations_by_status=dict(sorted(Counter(e.status.value for e in created).items())),
        recontact=recontact_proxy(queries, since=since, until=until, as_of=as_of),
        avg_confidence=(
            round(sum(confidences) / len(confidences), _RATE_DIGITS) if confidences else None
        ),
        confidence_samples=len(confidences),
        feedback_count=len(rated),
        negative_feedback=negative,
        negative_feedback_rate=safe_rate(negative, len(rated)),
        knowledge_gaps=len(gaps),
        gap_queries_scanned=gap_queries_scanned,
        gap_scan_truncated=gap_scan_truncated,
        top_gaps=list(gaps[:top_n]),
        query_scan_truncated="queries" in truncated,
        escalation_scan_truncated="escalations" in truncated,
        feedback_scan_truncated="feedback" in truncated,
    )


def metric_scores(metrics: ResolutionMetrics) -> dict[str, float | None]:
    """Numeric metrics under stable names, for the trace layer (``None`` = not measurable)."""
    return {
        "support.total_queries": float(metrics.total_queries),
        "support.escalation_rate": metrics.escalation_rate,
        "support.escalations_created": float(metrics.escalations_created),
        "support.recontact_rate_48h": metrics.recontact.rate,
        "support.knowledge_gaps": float(metrics.knowledge_gaps),
        "support.avg_confidence": metrics.avg_confidence,
        "support.negative_feedback_rate": metrics.negative_feedback_rate,
    }


# ── suggested topics and the weekly report ───────────────────────────────────


def is_policy_only(gap: KnowledgeGap) -> bool:
    """``True`` when every query in *gap* was escalated for policy reasons only."""
    return (
        bool(gap.reasons)
        and set(gap.outcomes) == {Outcome.ESCALATE.value}
        and set(gap.reasons) <= POLICY_REASONS
    )


def _topic_action(gap: KnowledgeGap) -> TopicAction:
    outcomes = Counter(gap.outcomes)
    leading, _count = outcomes.most_common(1)[0] if outcomes else (Outcome.ESCALATE.value, 0)
    if leading == Outcome.ANSWER.value:
        return "revise_existing_answer"
    if leading == Outcome.ANSWER_WITH_CAVEAT.value:
        return "expand_existing_content"
    return "add_faq_entry"


def suggest_topics(gaps: Sequence[KnowledgeGap]) -> tuple[list[SuggestedTopic], int]:
    """Turn ranked *gaps* into content suggestions; return them and the policy-only count."""
    topics: list[SuggestedTopic] = []
    policy_only = 0
    for gap in gaps:
        if is_policy_only(gap):
            policy_only += 1
            continue
        reasons = [
            reason
            for reason, _count in Counter(gap.reasons).most_common()
            if reason not in POLICY_REASONS
        ]
        topics.append(
            SuggestedTopic(
                gap_id=gap.gap_id,
                topic=gap.representative_query,
                queries=gap.count,
                action=_topic_action(gap),
                example_queries=list(gap.example_queries[:_MD_EXAMPLES]),
                top_reasons=reasons[:3],
            )
        )
    return topics, policy_only


def build_weekly_gap_report(
    metrics: ResolutionMetrics,
    gaps: Sequence[KnowledgeGap],
    *,
    generated_at: datetime,
    limit: int = DEFAULT_REPORT_GAPS,
) -> WeeklyGapReport:
    """Assemble the structured report and its Markdown rendering."""
    ranked = list(gaps[:limit])
    topics, policy_only = suggest_topics(ranked)
    markdown = render_weekly_gap_report(
        metrics, ranked, topics, policy_only=policy_only, generated_at=generated_at
    )
    return WeeklyGapReport(
        kb_id=metrics.kb_id,
        generated_at=generated_at,
        window_days=metrics.window_days,
        since=metrics.since,
        until=metrics.until,
        metrics=metrics,
        gaps=ranked,
        suggested_topics=topics,
        policy_only_gaps=policy_only,
        markdown=markdown,
    )


def md_text(text: str, max_chars: int = _MD_TEXT_CHARS) -> str:
    """Make stored (redacted) customer text safe inside one Markdown line.

    Collapses whitespace (no injected headings / table rows), truncates,
    defangs anything a viewer could auto-link (``https://evil.example`` becomes
    ``https[:]//evil[.]example``, ``a@b.example`` becomes ``a[@]b[.]example``,
    per GFM's extended autolinks and chat tools that link bare domains), and
    backslash-escapes Markdown syntax so links, images and HTML render as
    literal text. The report never contains a clickable customer-supplied link.
    """
    flat = _WHITESPACE.sub(" ", text).strip()
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1].rstrip() + "…"
    return _MD_SPECIAL.sub(r"\\\1", _LINK_LIKE.sub(_defang, flat))


def _defang(match: re.Match[str]) -> str:
    token = match.group(0)
    return token.replace("://", "[:]//").replace(".", "[.]").replace("@", "[@]")


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def _counts(counts: dict[str, int]) -> str:
    items = Counter(counts).most_common()
    return ", ".join(f"{md_text(name)} {count}" for name, count in items) or "none"


def _day(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def render_weekly_gap_report(
    metrics: ResolutionMetrics,
    gaps: Sequence[KnowledgeGap],
    topics: Sequence[SuggestedTopic],
    *,
    policy_only: int,
    generated_at: datetime,
) -> str:
    """Render the weekly knowledge-gap report as Markdown (customer text escaped)."""
    outcomes = metrics.outcomes
    total = metrics.total_queries
    recontact = metrics.recontact
    lines = [
        f"# Knowledge-gap report — {md_text(metrics.kb_id)}",
        "",
        f"Window: {_day(metrics.since)} to {_day(metrics.until)} ({metrics.window_days} days). "
        f"Generated {_day(generated_at)}.",
        "",
        "## Resolution metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Queries | {total} |",
        f"| Answered | {outcomes.answer} ({_pct(safe_rate(outcomes.answer, total))}) |",
        f"| Answered with caveat | {outcomes.answer_with_caveat} "
        f"({_pct(safe_rate(outcomes.answer_with_caveat, total))}) |",
        f"| Escalated (outcome) | {outcomes.escalate} ({_pct(metrics.escalation_rate)}) |",
        f"| Escalation hand-offs created | {metrics.escalations_created} |",
        f"| 48h re-contact (proxy) | {_pct(recontact.rate)} "
        f"({recontact.recontacted_sessions} of {recontact.eligible_sessions} sessions) |",
        f"| Sessions still inside the 48h period | {recontact.observation_incomplete_sessions} |",
        f"| Queries without session id (excluded) | {recontact.queries_without_session} |",
        "| Average confidence (uncalibrated) | "
        + ("n/a" if metrics.avg_confidence is None else f"{metrics.avg_confidence:.2f}")
        + " |",
        f"| Negative feedback | {metrics.negative_feedback} of {metrics.feedback_count} "
        f"({_pct(metrics.negative_feedback_rate)}) |",
        f"| Knowledge gaps (clusters) | {metrics.knowledge_gaps} |",
        "",
        f"> {recontact.definition}",
        "",
    ]
    capped = [
        name
        for name, flag in (
            ("queries", metrics.query_scan_truncated),
            ("escalations", metrics.escalation_scan_truncated),
            ("feedback", metrics.feedback_scan_truncated),
        )
        if flag
    ]
    if capped:
        lines += [
            f"> Partial window: only the newest rows of {', '.join(capped)} were read "
            "(METRICS_ROW_LIMIT); counts and rates cover part of the window.",
            "",
        ]
    if metrics.gap_scan_truncated:
        lines += [
            f"> Only the newest {metrics.gap_queries_scanned} gap-log rows were scanned "
            "(GAP_SCAN_LIMIT); older gaps in the window are not counted.",
            "",
        ]

    lines += ["## Top knowledge gaps", ""]
    if not gaps:
        lines += ["_No knowledge gaps in this window._", ""]
    for rank, gap in enumerate(gaps, 1):
        lines.append(
            f"{rank}. **{md_text(gap.representative_query)}** — {gap.count} "
            f"{'query' if gap.count == 1 else 'queries'}; outcomes: {_counts(gap.outcomes)}; "
            f"reasons: {_counts(gap.reasons)}; last seen {_day(gap.last_seen)}"
        )
        lines.extend(
            f"   - e.g. “{md_text(example)}”" for example in gap.example_queries[:_MD_EXAMPLES]
        )
    if gaps:
        lines.append("")

    lines += ["## Suggested FAQ topics", ""]
    if not topics:
        lines += ["_No content suggestions in this window._", ""]
    for topic in topics:
        action = topic.action.replace("_", " ")
        lines.append(f"- [ ] **{md_text(topic.topic)}** ({topic.queries}) — {action}")
    if topics:
        lines.append("")
    if policy_only:
        lines += [
            f"{policy_only} gap cluster(s) were escalated only for policy reasons "
            "(personal data, legal / medical topic, frustration) and are not content suggestions.",
            "",
        ]
    lines.append(
        "_Customer text above comes from the PII-redacted query log; markers such as "
        "[EMAIL] stand for removed personal data._"
    )
    return "\n".join(lines) + "\n"


# ── collection (the only I/O) ────────────────────────────────────────────────


async def collect_resolution_metrics(
    store: MetricsStore,
    embedder: Embedder,
    *,
    kb_id: str,
    window_days: int,
    until: datetime,
    as_of: datetime | None = None,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    gap_scan_limit: int = 1000,
    row_limit: int = DEFAULT_METRICS_ROW_LIMIT,
    top_n: int = DEFAULT_TOP_GAPS,
) -> tuple[ResolutionMetrics, list[KnowledgeGap]]:
    """Read the window's activity from *store* and compute the metrics.

    Returns the metrics and every gap cluster of the window (ranked), so the
    weekly report can list more than ``top_n``. *as_of* defaults to *until*.
    Each activity table is read with at most *row_limit* rows (newest kept);
    one extra row is requested to detect truncation.
    """
    since, until = window_bounds(window_days, until)
    probe = row_limit + 1
    queries = await store.list_query_activity(kb_id, since, until + RECONTACT_HORIZON, probe)
    escalations = await store.list_escalation_activity(kb_id, since, until, probe)
    feedback = await store.list_feedback_activity(kb_id, since, until, probe)
    truncated = frozenset(
        name
        for name, rows in (
            ("queries", queries),
            ("escalations", escalations),
            ("feedback", feedback),
        )
        if len(rows) > row_limit
    )
    gap_rows = await store.list_gap_queries(kb_id, since, gap_scan_limit + 1)
    in_window = [row for row in gap_rows[:gap_scan_limit] if row.created_at < until]
    gaps = await cluster_gaps(in_window, embedder, threshold=cluster_threshold)
    metrics = compute_resolution_metrics(
        kb_id=kb_id,
        window_days=window_days,
        since=since,
        until=until,
        as_of=as_of or until,
        # Lists are oldest first: keep the newest row_limit rows of each.
        queries=queries[-row_limit:],
        escalations=escalations[-row_limit:],
        feedback=feedback[-row_limit:],
        gaps=gaps,
        gap_queries_scanned=len(in_window),
        gap_scan_truncated=len(gap_rows) > gap_scan_limit,
        top_n=top_n,
        truncated=truncated,
    )
    return metrics, gaps


def publish_metrics(metrics: ResolutionMetrics, events: EventBus, *, source: str) -> None:
    """Push *metrics* to the trace layer (no-op without Langfuse) and emit ``metrics.reported``.

    Only aggregates leave this function — never query text.
    """
    push_metric_scores(
        metric_scores(metrics),
        metadata={"kb_id": metrics.kb_id, "window_days": metrics.window_days, "source": source},
    )
    events.emit(
        build_metrics_reported_event(
            metrics.kb_id,
            metrics.window_days,
            source=source,
            total_queries=metrics.total_queries,
            escalation_rate=metrics.escalation_rate,
            recontact_rate=metrics.recontact.rate,
            knowledge_gaps=metrics.knowledge_gaps,
        )
    )
