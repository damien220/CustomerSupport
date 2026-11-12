"""The four MCP tools, exercised over the real in-memory MCP protocol.

Each test builds the server with ``create_mcp_server`` and a lifespan yielding a
``SupportService`` over boundary fakes, so the full FastMCP stack (schema
generation, argument validation, lifespan context, result serialisation) runs.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from faq_core.events import FAQ_ADDED, Event, EventBus
from faq_core.service import SupportService
from server import create_mcp_server
from tests.conftest import FakeFAQRepository, FakeRAGPipeline, make_entry, static_lifespan

_TOOL_NAMES = {"search_faq", "add_faq_entry", "list_faq_categories", "get_faq_entry"}
# Phase 3 (additive): search_kb in every mode, answer_question in api/offline.
_PHASE3_TOOLS = {"search_kb", "answer_question"}
# Phase 4 (additive): escalation hand-off and the knowledge-gap backlog.
_PHASE4_TOOLS = {"create_escalation", "get_escalation", "list_knowledge_gaps"}
# Phase 6 operator tools: registered only with MCP_OPERATOR_TOOLS (operator_tools=True).
_OPERATOR_TOOLS = {"get_resolution_metrics"}


@asynccontextmanager
async def _client(service: SupportService) -> AsyncIterator[ClientSession]:
    server = create_mcp_server(static_lifespan(service))
    async with create_connected_server_and_client_session(server) as session:
        yield session


def _json(result: CallToolResult) -> dict[str, Any]:
    content = result.content[0]
    assert isinstance(content, TextContent)
    payload: dict[str, Any] = json.loads(content.text)
    return payload


# ── discovery ────────────────────────────────────────────────────────────────


async def test_lists_the_four_original_tools_plus_the_phase3_ones(
    service: SupportService,
) -> None:
    async with _client(service) as session:
        listing = await session.list_tools()

    assert {tool.name for tool in listing.tools} == _TOOL_NAMES | _PHASE3_TOOLS | _PHASE4_TOOLS


async def test_operator_tools_are_registered_only_when_enabled(service: SupportService) -> None:
    operator = create_mcp_server(static_lifespan(service), operator_tools=True)
    async with create_connected_server_and_client_session(operator) as session:
        enabled = {tool.name for tool in (await session.list_tools()).tools}
    async with _client(service) as session:
        hidden = await session.call_tool("get_resolution_metrics", {"window_days": 7})

    assert enabled == _TOOL_NAMES | _PHASE3_TOOLS | _PHASE4_TOOLS | _OPERATOR_TOOLS
    assert hidden.isError is True
    assert _json(hidden)["error"]["code"] == "unknown_tool"


async def test_search_kb_publishes_the_passages_only_output_schema(
    service: SupportService,
) -> None:
    async with _client(service) as session:
        listing = await session.list_tools()

    schema = next(tool.outputSchema for tool in listing.tools if tool.name == "search_kb")
    assert schema is not None
    assert set(schema["properties"]) >= {
        "query_id",
        "llm_mode",
        "passages",
        "citations",
        "confidence",
        "has_passages",
        "handling_instructions",
    }


async def test_every_tool_publishes_an_output_schema(service: SupportService) -> None:
    async with _client(service) as session:
        listing = await session.list_tools()

    schemas = {tool.name: tool.outputSchema for tool in listing.tools}
    assert all(schemas.values()), schemas
    assert set(schemas["search_faq"]["properties"]) >= {  # type: ignore[index]
        "query_id",
        "query",
        "answer",
        "confidence",
        "sources",
        "cached",
        "citations",
    }
    citation = schemas["search_faq"]["$defs"]["Citation"]["properties"]  # type: ignore[index]
    assert set(citation) >= {"id", "title", "snippet", "updated_at", "source_type", "source_id"}


# ── search_faq ───────────────────────────────────────────────────────────────


async def test_search_faq_returns_structured_answer(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    source = make_entry()
    engine = FakeRAGPipeline(sources=[source])
    service = SupportService(engine=engine, store=fake_faq_repository, events=event_bus)

    async with _client(service) as session:
        result = await session.call_tool("search_faq", {"query": "returns policy", "top_k": 7})

    assert result.isError is False
    payload = _json(result)
    assert payload == result.structuredContent
    assert payload["answer"] == "answer to returns policy"
    assert payload["sources"][0]["id"] == str(source.id)
    assert payload["cached"] is False
    uuid.UUID(payload["query_id"])
    assert engine.search_calls == [("returns policy", 7)]


async def test_search_faq_defaults_top_k_to_three(
    service: SupportService, fake_rag_pipeline: FakeRAGPipeline
) -> None:
    async with _client(service) as session:
        await session.call_tool("search_faq", {"query": "hours"})

    assert fake_rag_pipeline.search_calls == [("hours", 3)]


async def test_search_faq_invalid_top_k_is_structured_error(service: SupportService) -> None:
    async with _client(service) as session:
        result = await session.call_tool("search_faq", {"query": "hours", "top_k": 50})

    assert result.isError is True
    assert _json(result)["error"]["code"] == "invalid_request"


# ── add_faq_entry ────────────────────────────────────────────────────────────


async def test_add_faq_entry_persists_and_emits_faq_added(
    service: SupportService, fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    emitted: list[Event] = []

    async def _record(event: Event) -> None:
        emitted.append(event)

    event_bus.subscribe(FAQ_ADDED, _record)

    async with _client(service) as session:
        result = await session.call_tool(
            "add_faq_entry",
            {"question": "Do you ship overseas?", "answer": "Yes.", "category": "Shipping"},
        )
    await event_bus.drain()

    payload = _json(result)
    assert payload["question"] == "Do you ship overseas?"
    assert payload["kb_id"] == "default"
    assert await fake_faq_repository.get(payload["id"]) is not None
    assert [e.payload["faq_id"] for e in emitted] == [payload["id"]]


# ── list_faq_categories ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        ((), []),
        (("Shipping", "Account", "Shipping", "Billing"), ["Account", "Billing", "Shipping"]),
    ],
)
async def test_list_faq_categories_returns_sorted_unique(
    service: SupportService,
    fake_faq_repository: FakeFAQRepository,
    categories: tuple[str, ...],
    expected: list[str],
) -> None:
    for category in categories:
        await fake_faq_repository.create(make_entry(category=category))

    async with _client(service) as session:
        result = await session.call_tool("list_faq_categories", {})

    assert _json(result) == {"kb_id": "default", "categories": expected}


# ── get_faq_entry ────────────────────────────────────────────────────────────


async def test_get_faq_entry_returns_matching_entry(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    entry = await fake_faq_repository.create(make_entry())

    async with _client(service) as session:
        result = await session.call_tool("get_faq_entry", {"faq_id": str(entry.id)})

    assert result.isError is False
    assert _json(result)["id"] == str(entry.id)


async def test_get_unknown_faq_entry_is_structured_not_found_error(service: SupportService) -> None:
    missing = str(uuid.uuid4())

    async with _client(service) as session:
        result = await session.call_tool("get_faq_entry", {"faq_id": missing})

    assert result.isError is True
    assert _json(result)["error"] == {
        "code": "faq_not_found",
        "message": f"FAQ entry {missing!r} not found",
        "faq_id": missing,
    }


async def test_unknown_tool_is_a_structured_error(service: SupportService) -> None:
    async with _client(service) as session:
        result = await session.call_tool("no_such_tool", {})

    assert result.isError
    assert _json(result)["error"] == {
        "code": "unknown_tool",
        "message": "unknown tool 'no_such_tool'",
        "tool": "no_such_tool",
    }


# ── Phase 4: outcome fields, escalation and knowledge-gap tools ──────────────


async def test_search_tools_publish_the_additive_outcome_fields(service: SupportService) -> None:
    async with _client(service) as session:
        listing = await session.list_tools()

    schemas = {tool.name: tool for tool in listing.tools}
    phase4 = {"outcome", "escalate", "reasons", "caveat", "pii_detected", "session_id"}
    for name in ("search_kb", "search_faq", "answer_question"):
        assert set(schemas[name].outputSchema["properties"]) >= phase4  # type: ignore[index]
        assert "session_id" in schemas[name].inputSchema["properties"]
    escalation_input = schemas["create_escalation"].inputSchema["properties"]
    assert "url" not in str(escalation_input).lower()  # the destination is not agent input


async def test_escalation_round_trip_over_mcp(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    async with _client(service) as session:
        searched = _json(
            await session.call_tool(
                "search_kb", {"query": "Reset password?", "session_id": "conv-1"}
            )
        )
        created = await session.call_tool(
            "create_escalation",
            {
                "query_id": searched["query_id"],
                "reason": "customer is upset",
                "transcript": [{"role": "customer", "text": "mail me at a@b.io"}],
            },
        )
        fetched = await session.call_tool(
            "get_escalation", {"escalation_id": _json(created)["escalation_id"]}
        )
        gaps = await session.call_tool("list_knowledge_gaps", {"limit": 5})

    assert searched["outcome"] == "answer" and searched["escalate"] is False
    assert created.isError is False
    payload = _json(created)
    assert payload == created.structuredContent
    assert payload["session_id"] == "conv-1"
    assert payload["packet"]["transcript"] == [{"role": "customer", "text": "mail me at [EMAIL]"}]
    assert _json(fetched)["escalation_id"] == payload["escalation_id"]
    assert set(_json(gaps)) == {"kb_id", "since", "scanned", "gaps"}
    assert len(fake_faq_repository.escalations) == 1


@pytest.mark.parametrize(
    ("tool", "arguments", "code"),
    [
        ("create_escalation", {}, "invalid_request"),
        ("create_escalation", {"query_id": str(uuid.uuid4())}, "query_not_found"),
        ("get_escalation", {"escalation_id": str(uuid.uuid4())}, "escalation_not_found"),
        ("list_knowledge_gaps", {"limit": 500}, "invalid_request"),
        ("search_kb", {"query": "q", "session_id": "has spaces"}, "invalid_request"),
    ],
)
async def test_phase4_tool_errors_are_structured(
    service: SupportService, tool: str, arguments: dict[str, Any], code: str
) -> None:
    async with _client(service) as session:
        result = await session.call_tool(tool, arguments)

    assert result.isError is True
    assert _json(result)["error"]["code"] == code
