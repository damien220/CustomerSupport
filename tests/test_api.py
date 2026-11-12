"""REST + WebSocket surface over the real FastAPI stack (``TestClient`` runs the lifespan)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from bootstrap import assemble_runtime, build_runtime, serve
from faq_core.config import AppConfig
from faq_core.models import FAQEntry
from faq_core.repository import FAQRepository
from faq_core.service import SupportService
from tests.conftest import (
    FakeFAQRepository,
    FakeLLMClient,
    FakeRAGPipeline,
    bearer,
    make_entry,
    static_lifespan,
    token_security,
)


@pytest.fixture
def seeded_entry(fake_faq_repository: FakeFAQRepository) -> FAQEntry:
    entry = make_entry()
    fake_faq_repository._entries[str(entry.id)] = entry
    return entry


@pytest.fixture
def client(service: SupportService, seeded_entry: FAQEntry, api_token: str) -> Iterator[TestClient]:
    app = create_app(static_lifespan(service), security=token_security(api_token))
    with TestClient(app, headers=bearer(api_token)) as test_client:
        yield test_client


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "kb_id": "default",
        "llm_mode": "api",  # additive in Phase 3
        "answer_generation": True,  # additive in Phase 3
    }


def test_search_returns_structured_answer(client: TestClient) -> None:
    response = client.post("/search", json={"query": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer to hello"
    assert body["cached"] is False
    uuid.UUID(body["query_id"])


def test_search_rejects_blank_query(client: TestClient) -> None:
    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422


def test_get_faq_returns_entry(client: TestClient, seeded_entry: FAQEntry) -> None:
    response = client.get(f"/faq/{seeded_entry.id}")

    assert response.status_code == 200
    assert response.json()["question"] == "Reset password?"


def test_get_faq_unknown_id_returns_404_envelope(client: TestClient) -> None:
    response = client.get(f"/faq/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "faq_not_found"


def test_add_faq_returns_201_and_is_retrievable(client: TestClient) -> None:
    created = client.post("/faq", json={"question": "Q?", "answer": "A.", "category": "billing"})

    assert created.status_code == 201
    fetched = client.get(f"/faq/{created.json()['id']}")
    assert fetched.json()["category"] == "billing"


def test_categories_lists_sorted_unique(client: TestClient) -> None:
    client.post("/faq", json={"question": "Q?", "answer": "A.", "category": "billing"})

    response = client.get("/categories")

    assert response.json() == {"kb_id": "default", "categories": ["account", "billing"]}


def test_submit_feedback_records_and_returns_201(
    client: TestClient, fake_faq_repository: FakeFAQRepository
) -> None:
    query_id = client.post("/kb/search", json={"query": "hello"}).json()["query_id"]

    response = client.post(
        "/feedback", json={"query_id": query_id, "rating": 5, "comment": "Great"}
    )
    again = client.post("/feedback", json={"query_id": query_id, "rating": 1})

    assert response.status_code == 201
    assert response.json()["status"] == "recorded"
    assert again.status_code == 201
    [saved] = fake_faq_repository.saved_feedback  # the repeat replaced the first rating
    assert saved.rating == 1


def test_submit_feedback_for_an_unknown_query_returns_404(
    client: TestClient, fake_faq_repository: FakeFAQRepository
) -> None:
    response = client.post("/feedback", json={"query_id": str(uuid.uuid4()), "rating": 1})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "query_not_found"
    assert fake_faq_repository.saved_feedback == []


def test_submit_feedback_rejects_out_of_range_rating(client: TestClient) -> None:
    response = client.post("/feedback", json={"query_id": str(uuid.uuid4()), "rating": 9})

    assert response.status_code == 422


def test_ws_chat_relays_full_answer(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"query": "hello", "top_k": 3})
        data = websocket.receive_json()

    assert data["answer"] == "answer to hello"
    assert data["confidence"] == data["confidence_detail"]["score"]  # Phase 4 score


def test_ws_chat_reports_errors_and_keeps_the_socket_open(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"query": ""})
        error = websocket.receive_json()
        websocket.send_json({"query": "hello"})
        answer = websocket.receive_json()

    assert error["error"]["code"] == "invalid_request"
    assert answer["answer"] == "answer to hello"


def test_app_lifespan_seeds_runs_scheduler_and_closes_db(
    app_config: AppConfig, tmp_path: Path, api_token: str
) -> None:
    """Full bootstrap path: real repository + migrations, real scheduler, fake engine."""
    seed = tmp_path / "seed.json"
    seed.write_text('[{"question": "Q1", "answer": "A1", "category": "billing"}]')
    engine = FakeRAGPipeline()
    repo = FAQRepository(app_config)
    runtime = assemble_runtime(app_config, engine, repo)

    app = create_app(lambda: serve(runtime, seed_path=seed), security=token_security(api_token))
    with TestClient(app, headers=bearer(api_token)) as test_client:
        health = test_client.get("/health")
        categories = test_client.get("/categories").json()
        created = test_client.post(
            "/faq", json={"question": "Q2", "answer": "A2", "category": "account"}
        ).json()
        # faq.added → scheduler job → engine.add_document runs in the background.
        deadline = time.monotonic() + 5
        while not engine.added and time.monotonic() < deadline:
            time.sleep(0.02)

    assert health.status_code == 200
    assert categories["categories"] == ["billing"]
    assert [e.question for e in engine.seeded] == ["Q1"]
    assert [str(e.id) for e in engine.added] == [created["id"]]
    assert repo._conn is None  # closed by the lifespan


def test_search_over_the_real_lite_retrieval_stack(app_config: AppConfig, api_token: str) -> None:
    """POST /search against a temp DB: real migrations, chunking, FTS5, sqlite-vec, RRF.

    Only the LLM is faked (answer text); the bundled sample FAQ is seeded.
    """
    runtime = build_runtime(app_config, llm=FakeLLMClient(answer="Use Forgot Password."))

    app = create_app(lambda: serve(runtime), security=token_security(api_token))
    with TestClient(app, headers=bearer(api_token)) as test_client:
        response = test_client.post("/search", json={"query": "I forgot my password", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use Forgot Password."
    assert body["sources"][0]["question"] == "How do I reset my password?"
    citation = body["citations"][0]
    assert citation["source_type"] == "faq"
    assert citation["source_id"] == body["sources"][0]["id"]
    assert citation["title"] == "How do I reset my password?"
    assert "Forgot Password" in citation["snippet"]
    assert citation["updated_at"].endswith("Z")
    assert 0 < body["confidence"] <= 1


# ── Phase 4 routes ───────────────────────────────────────────────────────────


def test_search_routes_return_outcome_fields_and_accept_session_id(client: TestClient) -> None:
    for path in ("/search", "/kb/search", "/answer"):
        session = f"web{path.replace('/', '-')}"  # a repeated question would escalate
        body = client.post(path, json={"query": "hello", "session_id": session}).json()

        assert body["outcome"] == "answer" and body["escalate"] is False
        assert body["session_id"] == session


def test_escalation_routes(client: TestClient) -> None:
    searched = client.post("/kb/search", json={"query": "hello", "session_id": "web-2"}).json()

    created = client.post(
        "/escalations", json={"query_id": searched["query_id"], "reason": "needs a human"}
    )
    escalation_id = created.json()["escalation_id"]
    fetched = client.get(f"/escalations/{escalation_id}")
    missing = client.get(f"/escalations/{uuid.uuid4()}")
    empty = client.post("/escalations", json={})
    smuggled = client.post(
        "/escalations", json={"customer_goal": "x", "webhook_url": "https://evil.example"}
    )

    assert created.status_code == 201
    assert created.json()["packet"]["session_id"] == "web-2"
    assert fetched.status_code == 200 and fetched.json()["escalation_id"] == escalation_id
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "escalation_not_found"
    assert empty.status_code == 422
    assert smuggled.status_code == 422  # unknown fields are rejected, never used


def test_knowledge_gaps_route(client: TestClient) -> None:
    ok = client.get("/knowledge-gaps", params={"limit": 3, "since": "2026-01-01"})
    bad = client.get("/knowledge-gaps", params={"min_count": 0})

    assert ok.status_code == 200 and ok.json()["gaps"] == []
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"


def test_ws_chat_accepts_a_session_id(client: TestClient) -> None:
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"query": "hello", "session_id": "ws-1"})
        data = websocket.receive_json()
        websocket.send_json({"query": "hello", "session_id": 5})
        error = websocket.receive_json()

    assert data["session_id"] == "ws-1"
    assert error["error"]["code"] == "invalid_request"
