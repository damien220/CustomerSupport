"""Remote MCP (Streamable HTTP) behind authentication; stdio stays unauthenticated.

Drives the real :func:`remote_mcp.create_mcp_http_app` with raw JSON-RPC over
``TestClient`` (the SDK answers each POST as a short SSE stream).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from faq_core.events import EventBus
from faq_core.models import LLMMode
from faq_core.security.errors import INVALID_TOKEN_DESCRIPTION
from faq_core.security.policy import SecurityPolicy, build_security
from faq_core.service import SupportService
from remote_mcp import create_mcp_http_app, principal_user, sdk_transport_security
from server import create_mcp_server
from tests.conftest import (
    FakeFAQRepository,
    FakeRAGPipeline,
    bearer,
    make_api_token,
    make_entry,
    security_config,
    static_lifespan,
    token_security,
)
from tests.test_security_oauth import RSA_KEY, FakeFetcher, _claims, _jwk, _sign

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
CLIENT_MODE_TOOLS = {
    "search_kb",
    "search_faq",
    "add_faq_entry",
    "list_faq_categories",
    "get_faq_entry",
    "create_escalation",
    "get_escalation",
    "list_knowledge_gaps",
}
OPERATOR_TOOLS = {"get_resolution_metrics"}  # only with MCP_OPERATOR_TOOLS


def _service() -> SupportService:
    entry = make_entry()
    store = FakeFAQRepository()
    store._entries[str(entry.id)] = entry
    return SupportService(engine=FakeRAGPipeline(sources=[entry]), store=store, events=EventBus())


def _client(
    policy: SecurityPolicy, mode: LLMMode = LLMMode.CLIENT, *, operator_tools: bool = False
) -> TestClient:
    app = create_mcp_http_app(
        static_lifespan(_service()), security=policy, llm_mode=mode, operator_tools=operator_tools
    )
    return TestClient(app)


def _rpc(method: str, request_id: int | None = 1, params: dict[str, Any] | None = None) -> str:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        message["id"] = request_id
    return json.dumps(message)


_INITIALIZE = _rpc(
    "initialize",
    params={
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1"},
    },
)


def _payload(response: Response) -> dict[str, Any]:
    """The JSON-RPC message of a JSON or single-event SSE response."""
    if response.headers["content-type"].startswith("application/json"):
        return dict(response.json())
    data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    return dict(json.loads(data[-1]))


def _open_session(client: TestClient, headers: dict[str, str]) -> dict[str, str]:
    response = client.post("/mcp", content=_INITIALIZE, headers={**MCP_HEADERS, **headers})
    assert response.status_code == 200, response.text
    session = {**headers, "mcp-session-id": response.headers["mcp-session-id"]}
    session["mcp-protocol-version"] = _payload(response)["result"]["protocolVersion"]
    notified = client.post(
        "/mcp",
        content=_rpc("notifications/initialized", None),
        headers={**MCP_HEADERS, **session},
    )
    assert notified.status_code == 202
    return session


@pytest.fixture
def token() -> str:
    return make_api_token()


@pytest.fixture
def mcp(token: str) -> Iterator[TestClient]:
    with _client(token_security(token)) as client:
        yield client


# ── authentication ───────────────────────────────────────────────────────────


def test_mcp_http_without_a_token_is_401(mcp: TestClient) -> None:
    response = mcp.post("/mcp", content=_INITIALIZE, headers=MCP_HEADERS)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Bearer ")
    assert "mcp-session-id" not in response.headers


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_every_mcp_http_method_requires_a_token(mcp: TestClient, method: str) -> None:
    response = mcp.request(method, "/mcp", headers=MCP_HEADERS)

    assert response.status_code == 401


def test_mcp_http_with_a_wrong_token_is_401(mcp: TestClient) -> None:
    response = mcp.post(
        "/mcp", content=_INITIALIZE, headers={**MCP_HEADERS, **bearer(make_api_token())}
    )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == INVALID_TOKEN_DESCRIPTION


def test_tools_list_with_a_token_has_the_unchanged_client_mode_tools(
    mcp: TestClient, token: str
) -> None:
    session = _open_session(mcp, bearer(token))

    listed = mcp.post("/mcp", content=_rpc("tools/list", 2), headers={**MCP_HEADERS, **session})

    names = {tool["name"] for tool in _payload(listed)["result"]["tools"]}
    assert names == CLIENT_MODE_TOOLS


def test_operator_tools_are_listed_over_http_only_when_enabled(token: str) -> None:
    with _client(token_security(token), operator_tools=True) as client:
        session = _open_session(client, bearer(token))
        listed = client.post(
            "/mcp", content=_rpc("tools/list", 2), headers={**MCP_HEADERS, **session}
        )

    names = {tool["name"] for tool in _payload(listed)["result"]["tools"]}
    assert names == CLIENT_MODE_TOOLS | OPERATOR_TOOLS


def test_tool_call_over_http_returns_the_structured_contract(mcp: TestClient, token: str) -> None:
    session = _open_session(mcp, bearer(token))

    called = mcp.post(
        "/mcp",
        content=_rpc("tools/call", 3, {"name": "search_kb", "arguments": {"query": "password"}}),
        headers={**MCP_HEADERS, **session},
    )

    result = _payload(called)["result"]
    assert result["isError"] is False
    assert {"query_id", "passages", "citations", "confidence", "outcome"} <= set(
        result["structuredContent"]
    )


def test_a_session_cannot_be_used_with_another_key() -> None:
    first, second = make_api_token(), make_api_token()
    policy = build_security(
        security_config(API_KEYS=f"a:{first},b:{second}"), bind_host="127.0.0.1"
    )

    with _client(policy) as client:
        session = _open_session(client, bearer(first))
        hijack = {**session, **bearer(second)}
        response = client.post(
            "/mcp", content=_rpc("tools/list", 2), headers={**MCP_HEADERS, **hijack}
        )

    assert response.status_code == 404  # the SDK binds sessions to the principal


def test_mcp_http_is_rate_limited_per_key(token: str) -> None:
    with _client(token_security(token, RATE_LIMIT_REQUESTS=2)) as client:
        session = _open_session(client, bearer(token))  # 2 requests
        limited = client.post(
            "/mcp", content=_rpc("tools/list", 2), headers={**MCP_HEADERS, **session}
        )

    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


# ── Origin / Host ────────────────────────────────────────────────────────────


def test_bad_origin_is_403_before_authentication(mcp: TestClient, token: str) -> None:
    headers = {**MCP_HEADERS, **bearer(token), "Origin": "https://evil.example.com"}

    response = mcp.post("/mcp", content=_INITIALIZE, headers=headers)

    assert response.status_code == 403


def test_allowlisted_origin_passes_both_layers(token: str) -> None:
    policy = token_security(token, ALLOWED_ORIGINS="https://app.example.com")
    headers = {**MCP_HEADERS, **bearer(token), "Origin": "https://app.example.com"}

    with _client(policy) as client:
        response = client.post("/mcp", content=_INITIALIZE, headers=headers)

    assert response.status_code == 200


def test_dns_rebinding_host_is_refused(token: str) -> None:
    policy = token_security(token, ALLOWED_HOSTS="")
    app = create_mcp_http_app(static_lifespan(_service()), security=policy, llm_mode=LLMMode.CLIENT)

    with TestClient(app, base_url="http://rebound.attacker.example:8001") as client:
        response = client.post(
            "/mcp", content=_INITIALIZE, headers={**MCP_HEADERS, **bearer(token)}
        )

    assert response.status_code == 421


def test_sdk_transport_security_mirrors_the_allowlists(token: str) -> None:
    policy = token_security(
        token, ALLOWED_HOSTS="mcp.example.com,testserver:8001", ALLOWED_ORIGINS="https://a.example"
    )

    settings = sdk_transport_security(policy.origins)

    assert settings.enable_dns_rebinding_protection
    assert {"mcp.example.com", "mcp.example.com:*", "testserver:8001", "localhost:*"} <= set(
        settings.allowed_hosts
    )
    assert settings.allowed_origins == ["https://a.example"]


# ── OAuth: protected resource metadata over the MCP app ──────────────────────


def _oauth_policy() -> SecurityPolicy:
    config = security_config(
        AUTH_MODE="oauth",
        OAUTH_ISSUER="https://auth.example.com",
        OAUTH_RESOURCE_URL="https://mcp.example.com/mcp",
        OAUTH_JWKS_URL="https://auth.example.com/jwks",
    )
    return build_security(
        config, bind_host="127.0.0.1", jwks_fetcher=FakeFetcher([_jwk(RSA_KEY, "rsa-1")])
    )


def test_mcp_401_carries_resource_metadata_and_prm_is_served() -> None:
    with _client(_oauth_policy()) as client:
        challenge = client.post("/mcp", content=_INITIALIZE, headers=MCP_HEADERS)
        metadata = client.get("/.well-known/oauth-protected-resource/mcp")

    assert challenge.status_code == 401
    assert (
        'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"'
        in challenge.headers["WWW-Authenticate"]
    )
    assert metadata.status_code == 200
    assert metadata.json()["resource"] == "https://mcp.example.com/mcp"


def test_mcp_accepts_a_jwt_for_this_resource_only() -> None:
    with _client(_oauth_policy()) as client:
        ok = client.post(
            "/mcp", content=_INITIALIZE, headers={**MCP_HEADERS, **bearer(_sign(_claims()))}
        )
        foreign = client.post(
            "/mcp",
            content=_INITIALIZE,
            headers={**MCP_HEADERS, **bearer(_sign(_claims(aud="https://api.other.example")))},
        )

    assert ok.status_code == 200
    assert foreign.status_code == 401


def test_the_sdk_user_never_holds_the_token() -> None:
    principal = asyncio.run(_oauth_policy().verifier.verify(_sign(_claims())))  # type: ignore[union-attr]

    user = principal_user(principal)

    assert user.access_token.token == ""
    assert user.access_token.subject == "user-123"


# ── stdio is unchanged and unauthenticated ───────────────────────────────────


def test_stdio_server_has_no_auth_and_the_same_tools() -> None:
    server = create_mcp_server(static_lifespan(_service()), llm_mode=LLMMode.CLIENT)

    names = {tool.name for tool in asyncio.run(server.list_tools())}

    assert server.settings.auth is None
    assert names == CLIENT_MODE_TOOLS
