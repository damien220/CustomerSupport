"""MCP surface: the support tools as thin adapters over :class:`SupportService`.

Tool contract (agent-first, plan decision #4): every tool returns a pydantic
model, so FastMCP publishes an ``outputSchema`` and each result carries both
``structuredContent`` and the same JSON as text. Field names are stable.

=====================  ======================================================
tool                   result fields
=====================  ======================================================
search_kb              query_id, query, kb_id, llm_mode, passages[] (rank,
                       citation_id, source_type, source_id, title, heading,
                       text, source_uri, updated_at, score), citations[],
                       confidence {score, level, basis, calibrated, signals},
                       has_passages, handling_instructions[], outcome,
                       escalate, reasons[], caveat, pii_detected[],
                       session_id                             — all modes
answer_question        same fields as search_faq     — api / offline only
search_faq             query_id, query, answer, confidence, sources[], cached,
                       citations[] (id, source_type, source_id, title, heading,
                       snippet, source_uri, updated_at, score), generated,
                       outcome, escalate, reasons[], caveat,
                       confidence_detail, pii_detected[], session_id
add_faq_entry          id, question, answer, category, kb_id, created_at
list_faq_categories    kb_id, categories[]
get_faq_entry          id, question, answer, category, kb_id, created_at
create_escalation      escalation_id, kb_id, status, query_id, session_id,
                       reasons[], packet, delivery, created_at
get_escalation         same fields as create_escalation
list_knowledge_gaps    kb_id, since, scanned, gaps[] (gap_id,
                       representative_query, count, example_queries[],
                       outcomes, reasons, avg_confidence, first_seen,
                       last_seen)
get_resolution_metrics kb_id, window_days, since, until, total_queries,
                       outcomes {answer, answer_with_caveat, escalate},
                       escalation_rate, escalations_created,
                       escalations_by_status, recontact {horizon_hours,
                       definition, sessions, eligible_sessions,
                       recontacted_sessions, rate,
                       observation_incomplete_sessions,
                       queries_without_session}, avg_confidence,
                       confidence_samples, confidence_calibrated,
                       feedback_count, negative_feedback,
                       negative_feedback_rate, knowledge_gaps,
                       gap_queries_scanned, gap_scan_truncated, top_gaps[],
                       query_scan_truncated, escalation_scan_truncated,
                       feedback_scan_truncated   — MCP_OPERATOR_TOOLS only
=====================  ======================================================

Phase 6 added ``get_resolution_metrics`` (rates are ``null`` when their
denominator is zero; ``recontact`` is a documented 48h proxy). It is an
**operator tool**: it returns aggregated, redacted customer questions
(``top_gaps``), so it is registered only when ``MCP_OPERATOR_TOOLS=true`` —
keep that off on any server a customer-facing agent talks to. The REST routes
``/metrics/resolution`` and ``/reports/weekly-gaps`` are API-key gated instead.

Phase 4 added ``outcome`` (``answer`` | ``answer_with_caveat`` |
``escalate``), ``escalate`` and ``reasons`` to the search tools (additive), an
optional ``session_id`` argument that groups a conversation, and the three
escalation / knowledge-gap tools. Where escalations are delivered (sink,
webhook URL) is server configuration, never a tool argument.

``LLM_MODE=client`` (the caller's model writes the reply): ``answer_question``
is **not registered** — an agent never sees a tool it cannot use; calling it
anyway returns the ``answer_generation_unavailable`` envelope — and
``search_faq`` keeps working but returns ``generated: false`` with an empty
``answer``. ``search_kb`` is the primary tool in that mode.

Failures are tool errors (``isError: true``) whose text is the JSON envelope
``{"error": {"code": ..., "message": ..., ...}}`` — e.g. ``faq_not_found``,
``invalid_request`` or ``llm_unavailable``.

Importing this module builds nothing: the production server (:data:`mcp`) is
created on first access (reading ``LLM_MODE``), and services are created by
the server lifespan when it starts, so ``import server`` stays cheap.

No ``from __future__ import annotations`` here: FastMCP inspects the tool
signatures (``Context`` parameter, return model) when the tools are registered.
"""

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.session import ServerSession
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ContentBlock, TextContent
from pydantic import ValidationError

from bootstrap import production_service
from faq_core.config import get_config
from faq_core.errors import AnswerGenerationUnavailableError, InvalidRequestError, SupportError
from faq_core.models import (
    DEFAULT_METRICS_WINDOW_DAYS,
    CategoryList,
    EscalationResult,
    FAQEntry,
    KBSearchResult,
    KnowledgeGapList,
    LLMMode,
    ResolutionMetrics,
    SearchResult,
    TranscriptTurn,
)
from faq_core.service import ServiceLifespan, SupportService

ToolContext = Context[ServerSession, SupportService]

_INSTRUCTIONS_GENERATING = (
    "Customer-support knowledge base. For a customer question call answer_question "
    "(grounded answer + citations) or search_kb (passages only, if you write the reply "
    "yourself). Answer only from the returned citations and pass query_id along when "
    "reporting feedback. Pass the same session_id for every turn of one conversation. When "
    "a result has escalate=true, tell the customer a human will help and call "
    "create_escalation with its query_id. All tools return JSON objects with stable fields."
)
_INSTRUCTIONS_CLIENT = (
    "Customer-support knowledge base. This server does not generate answers: call "
    "search_kb for any customer question, then write the reply yourself from its passages, "
    "following its handling_instructions and citing citation_id values. Pass the same "
    "session_id for every turn of one conversation. When escalate is true, do not answer from "
    "the passages: tell the customer a human will help and call create_escalation with the "
    "query_id. All tools return JSON objects with stable fields."
)


class SupportMCPServer(FastMCP[SupportService]):
    """FastMCP whose tool errors are machine-readable JSON envelopes.

    FastMCP reports a failing tool as the prose ``"Error executing tool X: ..."``.
    Overriding :meth:`call_tool` (the low-level ``tools/call`` handler) lets
    domain and argument-validation errors return a ``CallToolResult`` whose text
    is ``{"error": {...}}`` instead; the low-level server passes a
    ``CallToolResult`` through unchanged.
    """

    llm_mode: LLMMode = LLMMode.API

    async def call_tool(  # type: ignore[override]  # low-level handler also accepts CallToolResult
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any] | CallToolResult:
        try:
            return await super().call_tool(name, arguments)
        except ToolError as err:
            envelope = _error_envelope(err.__cause__)
            if envelope is None and name not in {tool.name for tool in await self.list_tools()}:
                envelope = _unknown_tool_envelope(name, self.llm_mode)
            if envelope is None:
                raise
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(envelope))], isError=True
            )


def _error_envelope(cause: BaseException | None) -> dict[str, Any] | None:
    if isinstance(cause, SupportError):
        return cause.to_dict()
    if isinstance(cause, ValidationError):
        fields = [
            {"field": ".".join(str(part) for part in error["loc"]), "problem": error["msg"]}
            for error in cause.errors()
        ]
        return InvalidRequestError("invalid tool arguments", fields=fields).to_dict()
    return None


def _unknown_tool_envelope(name: str, llm_mode: LLMMode) -> dict[str, Any]:
    if name == "answer_question" and not llm_mode.generates_answers:
        return AnswerGenerationUnavailableError(llm_mode.value).to_dict()
    return {"error": {"code": "unknown_tool", "message": f"unknown tool {name!r}", "tool": name}}


def create_mcp_server(
    service_lifespan: ServiceLifespan,
    *,
    llm_mode: LLMMode = LLMMode.API,
    transport_security: TransportSecuritySettings | None = None,
    operator_tools: bool = False,
) -> SupportMCPServer:
    """Build the MCP server; *service_lifespan* provides the service per run.

    *llm_mode* decides the tool list: ``answer_question`` is registered only
    when the server generates answers (``api`` / ``offline``).
    *operator_tools* (``MCP_OPERATOR_TOOLS``) adds ``get_resolution_metrics``,
    which exposes aggregated customer questions — off for customer-facing agents.
    *transport_security* configures the SDK's own Host/Origin checks for the
    HTTP transport (:mod:`remote_mcp`); stdio ignores it.

    Note: the low-level server enters *service_lifespan* once per run — per
    stdio process, but per *session* over streamable HTTP, so
    :mod:`remote_mcp` passes a lifespan that only yields a shared service.
    """

    @asynccontextmanager
    async def lifespan(_server: FastMCP[SupportService]) -> AsyncIterator[SupportService]:
        async with service_lifespan() as service:
            yield service

    instructions = _INSTRUCTIONS_GENERATING if llm_mode.generates_answers else _INSTRUCTIONS_CLIENT
    mcp = SupportMCPServer(
        "Customer Support FAQ",
        instructions=instructions,
        lifespan=lifespan,
        transport_security=transport_security,
    )
    mcp.llm_mode = llm_mode

    @mcp.tool()
    async def search_kb(
        ctx: ToolContext, query: str, top_k: int = 3, session_id: str | None = None
    ) -> KBSearchResult:
        """Retrieve knowledge-base passages for a question — no generated answer.

        Returns the passages (full text, best first), citations, confidence
        {score, level, basis, calibrated, signals}, outcome (answer |
        answer_with_caveat | escalate), escalate, reasons and
        handling_instructions the answering model must follow. Write the reply
        yourself from the passages in the customer's language and cite
        citation_id values; if escalate is true, do not answer — call
        create_escalation with query_id.

        Args:
            query: The customer's question in natural language.
            top_k: Number of passages to retrieve (1-20).
            session_id: Optional conversation id; reuse it for every turn.
        """
        return await _service(ctx).search_kb(query, top_k, session_id=session_id)

    if llm_mode.generates_answers:

        @mcp.tool()
        async def answer_question(
            ctx: ToolContext, query: str, top_k: int = 3, session_id: str | None = None
        ) -> SearchResult:
            """Answer a customer question with a grounded, generated answer and citations.

            The answer is written only from the retrieved passages, listed in
            ``citations`` (best first, with snippet, freshness date and score),
            in the customer's language. ``outcome`` says how to use it; when
            ``escalate`` is true no answer was generated — call
            create_escalation with ``query_id``.

            Args:
                query: The customer's question in natural language.
                top_k: Number of passages to ground the answer on (1-20).
                session_id: Optional conversation id; reuse it for every turn.
            """
            return await _service(ctx).answer(query, top_k, session_id=session_id)

    @mcp.tool()
    async def search_faq(
        ctx: ToolContext, query: str, top_k: int = 3, session_id: str | None = None
    ) -> SearchResult:
        """Search the knowledge base and return a grounded answer with its sources.

        ``citations`` lists every retrieved passage (FAQ entry or ingested
        document) best first, with a snippet, freshness date and score. When
        ``generated`` is false (this server runs without an LLM) ``answer`` is
        empty: write the reply from ``citations``, or prefer ``search_kb``.

        Args:
            query: The customer's question in natural language.
            top_k: Number of FAQ sources to retrieve (1-20).
            session_id: Optional conversation id; reuse it for every turn.
        """
        return await _service(ctx).search(query, top_k, session_id=session_id)

    @mcp.tool()
    async def add_faq_entry(
        ctx: ToolContext, question: str, answer: str, category: str
    ) -> FAQEntry:
        """Add a new FAQ entry to the knowledge base; it is indexed in the background."""
        return await _service(ctx).add_entry(question, answer, category)

    @mcp.tool()
    async def list_faq_categories(ctx: ToolContext) -> CategoryList:
        """List all FAQ categories in the knowledge base."""
        return await _service(ctx).list_categories()

    @mcp.tool()
    async def get_faq_entry(ctx: ToolContext, faq_id: str) -> FAQEntry:
        """Retrieve one FAQ entry by its id (a UUID returned by the other tools)."""
        return await _service(ctx).get_entry(faq_id)

    @mcp.tool()
    async def create_escalation(
        ctx: ToolContext,
        query_id: str | None = None,
        session_id: str | None = None,
        customer_goal: str = "",
        reason: str = "",
        transcript: list[TranscriptTurn] | None = None,
    ) -> EscalationResult:
        """Hand the conversation to a human support agent.

        Builds a hand-off packet (customer goal, sources cited so far,
        attempts, reasons, redacted transcript), stores it and delivers it to
        the destination configured on the server. Give at least one of
        query_id, session_id or customer_goal. Personal data in the text is
        redacted before it is stored or sent.

        Args:
            query_id: The query_id of the result that should be escalated.
            session_id: The conversation id used with the search tools.
            customer_goal: What the customer is trying to achieve (<= 1000 chars).
            reason: Why you are escalating, in a few words (<= 500 chars).
            transcript: Optional conversation turns [{role, text}], at most 50;
                role is customer | assistant | agent | system.
        """
        return await _service(ctx).create_escalation(
            query_id=query_id,
            session_id=session_id,
            customer_goal=customer_goal,
            reason=reason,
            transcript=transcript or [],
        )

    @mcp.tool()
    async def get_escalation(ctx: ToolContext, escalation_id: str) -> EscalationResult:
        """Inspect one escalation: its packet, status and delivery result."""
        return await _service(ctx).get_escalation(escalation_id)

    @mcp.tool()
    async def list_knowledge_gaps(
        ctx: ToolContext, limit: int = 20, min_count: int = 1, since: str | None = None
    ) -> KnowledgeGapList:
        """List what the knowledge base is missing: clusters of unanswered queries.

        Clusters low-confidence, escalated and badly rated queries (stored
        redacted) into a backlog ranked by how often they were asked.

        Args:
            limit: Maximum clusters to return (1-100).
            min_count: Only clusters with at least this many queries.
            since: ISO-8601 date/time (UTC if no offset); default 30 days ago.
        """
        return await _service(ctx).list_knowledge_gaps(
            limit=limit, min_count=min_count, since=since
        )

    if operator_tools:
        _register_operator_tools(mcp)

    return mcp


def _register_operator_tools(mcp: SupportMCPServer) -> None:
    """Tools for support operators, not for agents that talk to customers."""

    @mcp.tool()
    async def get_resolution_metrics(
        ctx: ToolContext, window_days: int = DEFAULT_METRICS_WINDOW_DAYS
    ) -> ResolutionMetrics:
        """Report how well support questions were resolved over the last window_days.

        Returns query counts per outcome, the escalation rate, escalation
        records created, a 48h re-contact proxy (share of sessions that asked
        again within 48h of an answer — likely unresolved; see
        recontact.definition), average confidence, the negative-feedback rate
        and the number of knowledge-gap clusters with the largest ones. Rates
        are fractions 0-1, or null when there is nothing to divide by. Query
        text is PII-redacted.

        Args:
            window_days: Days back from now to cover (1-365, default 7).
        """
        return await _service(ctx).resolution_metrics(window_days)


def _service(ctx: ToolContext) -> SupportService:
    return ctx.request_context.lifespan_context


@cache
def production_server() -> SupportMCPServer:
    """The production server for the configured ``LLM_MODE`` (built once)."""
    config = get_config()
    return create_mcp_server(
        production_service,
        llm_mode=config.LLM_MODE,
        operator_tools=config.MCP_OPERATOR_TOOLS,
    )


def __getattr__(name: str) -> SupportMCPServer:
    """``server.mcp`` — the production server, built on first access.

    ``python app.py --mode mcp`` runs it over stdio (unauthenticated: the MCP
    client launched this process). The HTTP transport is :mod:`remote_mcp`.
    """
    if name == "mcp":
        return production_server()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    production_server().run()
