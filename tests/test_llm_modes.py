"""The three LLM access modes end to end, against fakes (no network, no real keys).

Each test builds the real runtime (``bootstrap.build_runtime``: guard, gateway
factory, lite hybrid retrieval over a temp SQLite file, seeded sample FAQ) and
drives it through the MCP protocol or the REST app:

* ``api`` — a fake gateway generates; ``answer_question`` is registered.
* ``client`` — the factory's ``NullGateway``; ``search_kb`` returns passages,
  ``answer_question`` is not registered, ``search_faq`` reports ``generated: false``.
* ``offline`` — a local endpoint passes the guard; generation works.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from api.app import create_app
from bootstrap import Runtime, build_runtime, serve
from faq_core.config import AppConfig
from faq_core.errors import AnswerGenerationUnavailableError
from faq_core.events import EventBus
from faq_core.models import LLMMode
from faq_core.service import HANDLING_INSTRUCTIONS, NO_PASSAGES_INSTRUCTION, ServiceLifespan
from server import create_mcp_server
from tests.conftest import (
    FakeFAQRepository,
    FakeGateway,
    FakeRAGPipeline,
    bearer,
    make_api_token,
    token_security,
)

_COMMON_TOOLS = {
    "search_kb",
    "search_faq",
    "add_faq_entry",
    "list_faq_categories",
    "get_faq_entry",
    "create_escalation",  # Phase 4: every mode
    "get_escalation",
    "list_knowledge_gaps",
}  # get_resolution_metrics only with MCP_OPERATOR_TOOLS (default off)


@pytest.fixture(autouse=True)
def _restore_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE the offline guard sets."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def _config(tmp_path: Path, mode: str, **values: object) -> AppConfig:
    settings: dict[str, object] = {
        "LLM_MODE": mode,
        "DB_PATH": str(tmp_path / f"{mode}.db"),
        "EMBED_PROVIDER": "hashing",
        "RERANKER": "none",
    }
    settings.update(values)
    return AppConfig(_env_file=None, **settings)  # type: ignore[arg-type]


def _lifespan(runtime: Runtime) -> ServiceLifespan:
    @asynccontextmanager
    async def _run() -> AsyncIterator[Any]:
        async with serve(runtime) as service:
            yield service

    return _run


@asynccontextmanager
async def _mcp(runtime: Runtime) -> AsyncIterator[ClientSession]:
    server = create_mcp_server(_lifespan(runtime), llm_mode=runtime.config.LLM_MODE)
    async with create_connected_server_and_client_session(server) as session:
        yield session


def _json(result: CallToolResult) -> dict[str, Any]:
    content = result.content[0]
    assert isinstance(content, TextContent)
    payload: dict[str, Any] = json.loads(content.text)
    return payload


# ── api ──────────────────────────────────────────────────────────────────────


async def test_api_mode_answer_question_generates_from_retrieved_passages(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway(answer="Click 'Forgot Password' on the login page.")
    runtime = build_runtime(_config(tmp_path, "api"), llm=gateway)

    async with _mcp(runtime) as session:
        tools = {tool.name for tool in (await session.list_tools()).tools}
        result = await session.call_tool(
            "answer_question", {"query": "How do I reset my password?"}
        )

    assert tools == _COMMON_TOOLS | {"answer_question"}
    assert not result.isError
    payload = _json(result)
    assert payload["answer"] == "Click 'Forgot Password' on the login page."
    assert payload["generated"] is True
    assert payload["citations"][0]["title"] == "How do I reset my password?"
    assert "Forgot Password" in gateway.generate_contexts[0][0]
    assert gateway.closed  # serve() closes the gateway on shutdown


async def test_search_kb_never_calls_the_llm_even_in_api_mode(tmp_path: Path) -> None:
    gateway = FakeGateway()
    runtime = build_runtime(_config(tmp_path, "api"), llm=gateway)

    async with _mcp(runtime) as session:
        result = await session.call_tool("search_kb", {"query": "How long does shipping take?"})

    payload = _json(result)
    assert payload["llm_mode"] == "api"
    assert payload["has_passages"] is True
    assert gateway.generate_calls == []


# ── client ───────────────────────────────────────────────────────────────────


async def test_client_mode_needs_no_key_and_hides_answer_question(tmp_path: Path) -> None:
    runtime = build_runtime(_config(tmp_path, "client", ANTHROPIC_API_KEY=""))

    async with _mcp(runtime) as session:
        listing = await session.list_tools()
        missing = await session.call_tool("answer_question", {"query": "hi"})

    assert runtime.llm is not None and not runtime.llm.can_generate
    assert {tool.name for tool in listing.tools} == _COMMON_TOOLS
    assert missing.isError
    error = _json(missing)["error"]
    assert error["code"] == "answer_generation_unavailable"
    assert error["use_instead"] == "search_kb"


async def test_client_mode_search_kb_returns_passages_citations_and_instructions(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(_config(tmp_path, "client"))

    async with _mcp(runtime) as session:
        result = await session.call_tool(
            "search_kb", {"query": "How do I reset my password?", "top_k": 2}
        )

    assert not result.isError
    payload = _json(result)
    assert result.structuredContent == payload
    assert payload["llm_mode"] == "client"
    assert payload["has_passages"] is True
    assert 1 <= len(payload["passages"]) <= 2
    best = payload["passages"][0]
    assert best["rank"] == 1
    assert best["title"] == "How do I reset my password?"
    assert "Forgot Password" in best["text"]
    assert best["citation_id"] == payload["citations"][0]["id"]
    confidence = payload["confidence"]
    assert set(confidence) == {"score", "level", "basis", "calibrated", "signals"}
    assert (confidence["basis"], confidence["calibrated"]) == ("retrieval_signals", False)
    assert confidence["level"] == "high"
    assert 0.0 < confidence["score"] <= 1.0
    assert (payload["outcome"], payload["escalate"]) == ("answer", False)
    assert payload["handling_instructions"] == list(HANDLING_INSTRUCTIONS)


async def test_client_mode_search_faq_keeps_its_contract_without_generating(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(_config(tmp_path, "client"))

    async with _mcp(runtime) as session:
        result = await session.call_tool("search_faq", {"query": "How long does shipping take?"})

    payload = _json(result)
    assert not result.isError
    assert payload["generated"] is False
    assert payload["answer"] == ""
    assert payload["citations"]
    assert set(payload) >= {"query_id", "query", "answer", "confidence", "sources", "cached"}


def test_client_mode_rest_surface(tmp_path: Path) -> None:
    runtime = build_runtime(_config(tmp_path, "client"))

    token = make_api_token()
    app = create_app(_lifespan(runtime), security=token_security(token))
    with TestClient(app, headers=bearer(token)) as client:
        health = client.get("/health").json()
        passages = client.post("/kb/search", json={"query": "What payment methods?"})
        answer = client.post("/answer", json={"query": "What payment methods?"})
        search = client.post("/search", json={"query": "What payment methods?"})

    assert health == {
        "status": "ok",
        "kb_id": "default",
        "llm_mode": "client",
        "answer_generation": False,
    }
    assert passages.status_code == 200
    assert passages.json()["passages"]
    assert answer.status_code == 501
    assert answer.json()["error"]["code"] == "answer_generation_unavailable"
    assert search.status_code == 200
    assert search.json()["generated"] is False


async def test_service_answer_raises_when_generation_is_disabled(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    from faq_core.service import SupportService

    service = SupportService(
        engine=FakeRAGPipeline(generates_answers=False),
        store=fake_faq_repository,
        events=event_bus,
        llm_mode=LLMMode.CLIENT,
    )

    with pytest.raises(AnswerGenerationUnavailableError):
        await service.answer("anything")


async def test_search_kb_without_hits_flags_it_and_adds_the_no_passages_rule(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    from faq_core.service import SupportService

    service = SupportService(
        engine=FakeRAGPipeline(generates_answers=False),
        store=fake_faq_repository,
        events=event_bus,
        llm_mode=LLMMode.CLIENT,
    )

    result = await service.search_kb("quantum chromodynamics")

    assert result.has_passages is False
    assert result.passages == [] and result.citations == []
    assert result.confidence.score == 0.0
    assert result.handling_instructions[0] == NO_PASSAGES_INSTRUCTION


# ── offline ──────────────────────────────────────────────────────────────────


async def test_offline_mode_with_a_local_endpoint_generates(tmp_path: Path) -> None:
    config = _config(
        tmp_path, "offline", LLM_MODEL="llama3.2", LLM_BASE_URL="http://127.0.0.1:9931/v1"
    )
    runtime = build_runtime(config, llm=FakeGateway(answer="Five to seven business days."))

    async with _mcp(runtime) as session:
        tools = {tool.name for tool in (await session.list_tools()).tools}
        result = await session.call_tool(
            "answer_question", {"query": "How long does shipping take?"}
        )

    assert "answer_question" in tools
    assert _json(result)["answer"] == "Five to seven business days."


async def test_offline_mode_real_gateway_targets_the_local_runtime(tmp_path: Path) -> None:
    from faq_core.llm.openai_compatible_gateway import OpenAICompatibleGateway

    config = _config(tmp_path, "offline", LLM_MODEL="llama3.2")
    runtime = build_runtime(config)
    await runtime.repo.close()

    assert isinstance(runtime.llm, OpenAICompatibleGateway)
    assert runtime.llm.base_url == "http://localhost:11434/v1"


# ── Phase 4 over the real lite stack ─────────────────────────────────────────


async def test_api_mode_out_of_kb_question_escalates_without_an_llm_call(tmp_path: Path) -> None:
    gateway = FakeGateway(answer="made-up answer")
    runtime = build_runtime(_config(tmp_path, "api"), llm=gateway)

    async with _mcp(runtime) as session:
        result = await session.call_tool(
            "answer_question", {"query": "who won the football game last night"}
        )
        caveat = await session.call_tool("answer_question", {"query": "how to change my username"})

    payload = _json(result)
    assert (payload["outcome"], payload["escalate"], payload["generated"]) == (
        "escalate",
        True,
        False,
    )
    assert "made-up" not in payload["answer"]
    assert gateway.generate_calls == ["how to change my username"]  # only the caveat case
    assert _json(caveat)["outcome"] == "answer_with_caveat"
    assert gateway.caveat_flags == [True]


async def test_client_mode_end_to_end_escalation_and_gap_listing(tmp_path: Path) -> None:
    runtime = build_runtime(_config(tmp_path, "client"))

    async with _mcp(runtime) as session:
        searched = _json(
            await session.call_tool(
                "search_kb", {"query": "recommend a good movie", "session_id": "e2e"}
            )
        )
        escalation = _json(
            await session.call_tool("create_escalation", {"query_id": searched["query_id"]})
        )
        gaps = _json(await session.call_tool("list_knowledge_gaps", {}))

    assert searched["escalate"] is True
    assert escalation["status"] == "open" and escalation["delivery"]["sink"] == "log"
    assert gaps["gaps"][0]["representative_query"] == "recommend a good movie"
