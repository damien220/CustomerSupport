"""REST + WebSocket routes — thin adapters over :class:`SupportService`.

GET  /health            liveness, knowledge-base id, LLM mode
POST /search            answer a question               → SearchResult
                        (generated=false, answer="" in LLM_MODE=client)
POST /kb/search         passages only, every LLM mode   → KBSearchResult
POST /answer            generated answer                → SearchResult
                        (501 answer_generation_unavailable in LLM_MODE=client)
GET  /faq/{faq_id}      one entry (404 when unknown)    → FAQEntry
POST /faq               create an entry (201)           → FAQEntry
GET  /categories        list categories                 → CategoryList
POST /feedback          rate a logged answer (201; 404 query_not_found;
                        a repeat rating replaces the earlier one) → FeedbackReceipt
POST /escalations       hand off to a human (201)       → EscalationResult
GET  /escalations/{id}  inspect an escalation           → EscalationResult
GET  /knowledge-gaps    ?limit&min_count&since          → KnowledgeGapList
GET  /metrics/resolution ?window_days (1-365, default 7) → ResolutionMetrics
GET  /reports/weekly-gaps ?window_days&limit (1-100)     → WeeklyGapReport
WS   /ws/chat           send {"query", "top_k", "session_id"?}; receive one
                        SearchResult (or an error envelope) per message

The search bodies accept an optional ``session_id`` (Phase 4). The escalation
destination is server configuration, never request input.
"""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from api.deps import ServiceDep
from faq_core.errors import InvalidRequestError, SupportError
from faq_core.models import (
    DEFAULT_METRICS_WINDOW_DAYS,
    DEFAULT_REPORT_GAPS,
    CategoryList,
    EscalationRequest,
    EscalationResult,
    FAQDraft,
    FAQEntry,
    FeedbackReceipt,
    FeedbackRequest,
    KBSearchResult,
    KnowledgeGapList,
    LLMMode,
    QueryRequest,
    ResolutionMetrics,
    SearchResult,
    WeeklyGapReport,
)
from faq_core.service import SupportService

router = APIRouter()


class HealthStatus(BaseModel):
    status: str
    kb_id: str
    llm_mode: LLMMode
    answer_generation: bool


@router.get("/health", tags=["ops"])
async def health(service: ServiceDep) -> HealthStatus:
    return HealthStatus(
        status="ok",
        kb_id=service.kb_id,
        llm_mode=service.llm_mode,
        answer_generation=service.generates_answers,
    )


@router.post("/search", tags=["faq"])
async def search(request: QueryRequest, service: ServiceDep) -> SearchResult:
    return await service.search(request.query, request.top_k, session_id=request.session_id)


@router.post("/kb/search", tags=["faq"])
async def search_kb(request: QueryRequest, service: ServiceDep) -> KBSearchResult:
    return await service.search_kb(request.query, request.top_k, session_id=request.session_id)


@router.post("/answer", tags=["faq"])
async def answer(request: QueryRequest, service: ServiceDep) -> SearchResult:
    return await service.answer(request.query, request.top_k, session_id=request.session_id)


@router.get("/faq/{faq_id}", tags=["faq"])
async def get_faq(faq_id: str, service: ServiceDep) -> FAQEntry:
    return await service.get_entry(faq_id)


@router.post("/faq", status_code=status.HTTP_201_CREATED, tags=["faq"])
async def add_faq(draft: FAQDraft, service: ServiceDep) -> FAQEntry:
    return await service.add_entry(draft.question, draft.answer, draft.category)


@router.get("/categories", tags=["faq"])
async def list_categories(service: ServiceDep) -> CategoryList:
    return await service.list_categories()


@router.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["feedback"])
async def submit_feedback(feedback: FeedbackRequest, service: ServiceDep) -> FeedbackReceipt:
    return await service.record_feedback(
        feedback.query_id, feedback.rating, feedback.comment, helpful=feedback.helpful
    )


@router.post("/escalations", status_code=status.HTTP_201_CREATED, tags=["escalation"])
async def create_escalation(request: EscalationRequest, service: ServiceDep) -> EscalationResult:
    return await service.create_escalation(
        query_id=request.query_id,
        session_id=request.session_id,
        customer_goal=request.customer_goal,
        reason=request.reason,
        transcript=request.transcript,
    )


@router.get("/escalations/{escalation_id}", tags=["escalation"])
async def get_escalation(escalation_id: str, service: ServiceDep) -> EscalationResult:
    return await service.get_escalation(escalation_id)


@router.get("/knowledge-gaps", tags=["escalation"])
async def list_knowledge_gaps(
    service: ServiceDep, limit: int = 20, min_count: int = 1, since: str | None = None
) -> KnowledgeGapList:
    return await service.list_knowledge_gaps(limit=limit, min_count=min_count, since=since)


@router.get("/metrics/resolution", tags=["metrics"])
async def resolution_metrics(
    service: ServiceDep, window_days: int = DEFAULT_METRICS_WINDOW_DAYS
) -> ResolutionMetrics:
    return await service.resolution_metrics(window_days)


@router.get("/reports/weekly-gaps", tags=["metrics"])
async def weekly_gap_report(
    service: ServiceDep,
    window_days: int = DEFAULT_METRICS_WINDOW_DAYS,
    limit: int = DEFAULT_REPORT_GAPS,
) -> WeeklyGapReport:
    return await service.weekly_gap_report(window_days, limit)


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    service = cast("SupportService", websocket.app.state.service)
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            try:
                query, top_k, session_id = _parse_ws_query(message)
                result = await service.search(query, top_k, session_id=session_id)
            except SupportError as err:
                await websocket.send_json(err.to_dict())
                continue
            await websocket.send_json(result.model_dump(mode="json"))
    except WebSocketDisconnect:
        return


def _parse_ws_query(message: object) -> tuple[str, int, str | None]:
    if not isinstance(message, dict):
        raise InvalidRequestError("expected a JSON object with a 'query' field")
    query = message.get("query", "")
    top_k = message.get("top_k", 3)
    session_id = message.get("session_id")
    if not isinstance(query, str) or isinstance(top_k, bool) or not isinstance(top_k, int):
        raise InvalidRequestError("'query' must be a string and 'top_k' an integer")
    if session_id is not None and not isinstance(session_id, str):
        raise InvalidRequestError("'session_id' must be a string")
    return query, top_k, session_id
