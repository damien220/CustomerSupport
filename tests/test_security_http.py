"""REST + WebSocket security: auth per route, Origin/Host, rate limits, no passthrough.

Runs the real FastAPI app and :class:`~api.security.SecurityMiddleware` through
``TestClient``; tokens are generated per run.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.security import SecurityMiddleware
from faq_core.events import EventBus
from faq_core.handoff import WebhookSink
from faq_core.security.errors import INVALID_TOKEN_DESCRIPTION
from faq_core.security.policy import SecurityPolicy, build_security
from faq_core.service import SupportService
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
from tests.test_security_oauth import EC_KEY, RSA_KEY, FakeFetcher, _claims, _jwk, _sign

ENTRY = make_entry()

# (method, path, json body) for every authenticated REST route.
PROTECTED_ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/search", {"query": "reset password"}),
    ("POST", "/kb/search", {"query": "reset password"}),
    ("POST", "/answer", {"query": "reset password"}),
    ("GET", f"/faq/{ENTRY.id}", None),
    ("POST", "/faq", {"question": "Q?", "answer": "A.", "category": "misc"}),
    ("GET", "/categories", None),
    ("POST", "/feedback", {"query_id": str(uuid.uuid4()), "rating": 4}),
    ("POST", "/escalations", {"customer_goal": "talk to a human"}),
    ("GET", f"/escalations/{uuid.uuid4()}", None),
    ("GET", "/knowledge-gaps", None),
    ("GET", "/openapi.json", None),
    ("GET", "/docs", None),
]


def _service(**kwargs: Any) -> SupportService:  # noqa: ANN401
    store = FakeFAQRepository()
    store._entries[str(ENTRY.id)] = ENTRY
    return SupportService(
        engine=FakeRAGPipeline(sources=[ENTRY]), store=store, events=EventBus(), **kwargs
    )


@pytest.fixture
def token() -> str:
    return make_api_token()


def _client(policy: SecurityPolicy, **kwargs: Any) -> TestClient:  # noqa: ANN401
    return TestClient(create_app(static_lifespan(_service()), security=policy), **kwargs)


@pytest.fixture
def anon(token: str) -> Iterator[TestClient]:
    with _client(token_security(token)) as client:
        yield client


@pytest.fixture
def authed(token: str) -> Iterator[TestClient]:
    with _client(token_security(token), headers=bearer(token)) as client:
        yield client


# ── authentication per route ─────────────────────────────────────────────────


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ROUTES)
def test_every_rest_route_requires_a_token(
    anon: TestClient, method: str, path: str, body: dict[str, Any] | None
) -> None:
    response = anon.request(method, path, json=body)

    assert response.status_code == 401
    scheme, _, params = response.headers["WWW-Authenticate"].partition(" ")
    assert scheme == "Bearer"
    assert params.startswith('realm="customer-support"')
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ROUTES)
def test_a_wrong_token_is_refused_on_every_route(
    anon: TestClient, method: str, path: str, body: dict[str, Any] | None
) -> None:
    response = anon.request(method, path, json=body, headers=bearer(make_api_token()))

    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["WWW-Authenticate"]
    assert response.json()["error"]["message"] == INVALID_TOKEN_DESCRIPTION


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED_ROUTES)
def test_the_right_token_is_allowed_on_every_route(
    authed: TestClient, method: str, path: str, body: dict[str, Any] | None
) -> None:
    response = authed.request(method, path, json=body)

    assert response.status_code not in (401, 403, 429)
    assert response.status_code < 500


def test_health_stays_open(anon: TestClient) -> None:
    assert anon.get("/health").status_code == 200


@pytest.mark.parametrize(
    "authorization",
    ["Basic dXNlcjpwYXNz", "Bearer", "Token abc", "bearer   "],
)
def test_other_schemes_and_empty_bearer_are_refused(anon: TestClient, authorization: str) -> None:
    response = anon.get("/categories", headers={"Authorization": authorization})

    assert response.status_code == 401


def test_malformed_and_unknown_tokens_get_identical_answers(anon: TestClient) -> None:
    malformed = anon.get("/categories", headers=bearer("not a token!"))
    unknown = anon.get("/categories", headers=bearer(make_api_token()))

    assert malformed.status_code == unknown.status_code == 401
    assert malformed.json() == unknown.json()
    assert malformed.headers["WWW-Authenticate"] == unknown.headers["WWW-Authenticate"]


def test_two_authorization_headers_are_refused(anon: TestClient, token: str) -> None:
    headers = [("Authorization", bearer(token)["Authorization"])] * 2
    response = anon.get("/categories", headers=httpx.Headers(headers))

    # 400 when both headers arrive; some transports fold them into one value,
    # which is then an invalid token (401). Either way the request is refused.
    assert response.status_code in (400, 401)


def test_tokens_in_the_query_string_are_refused(anon: TestClient, token: str) -> None:
    response = anon.get("/categories", params={"access_token": token})

    assert response.status_code == 400
    assert token not in response.text


def test_failed_tokens_never_reach_logs_or_responses(
    anon: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    attempted = make_api_token()

    with caplog.at_level(logging.DEBUG):
        response = anon.get("/categories", headers=bearer(attempted))

    assert attempted not in response.text
    assert attempted not in str(response.headers)
    rendered = [str(vars(record)) for record in caplog.records]
    assert all(attempted not in line for line in rendered)
    # The reason is logged; the token is not.
    assert any(getattr(r, "reason", None) == "unknown_static_key" for r in caplog.records)


# ── WebSocket ────────────────────────────────────────────────────────────────


def test_websocket_without_a_token_is_closed_before_accept(anon: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as caught, anon.websocket_connect("/ws/chat"):
        pass

    assert caught.value.code == 1008


def test_websocket_with_a_wrong_token_is_closed(anon: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        anon.websocket_connect("/ws/chat", headers=bearer(make_api_token())),
    ):
        pass


def test_websocket_with_the_token_in_the_query_string_is_refused(
    anon: TestClient, token: str
) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        anon.websocket_connect(f"/ws/chat?access_token={token}"),
    ):
        pass


def test_websocket_with_a_token_works(authed: TestClient) -> None:
    with authed.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"query": "reset password"})
        result = websocket.receive_json()

    assert "query_id" in result


def test_websocket_from_a_foreign_origin_is_refused(authed: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        authed.websocket_connect("/ws/chat", headers={"Origin": "https://evil.example.com"}),
    ):
        pass


def test_websocket_messages_count_against_the_rate_limit(token: str) -> None:
    policy = token_security(token, RATE_LIMIT_REQUESTS=2)

    with (
        _client(policy, headers=bearer(token)) as client,
        client.websocket_connect("/ws/chat") as websocket,
    ):
        websocket.send_json({"query": "reset password"})  # handshake = 1, message = 2
        websocket.receive_json()
        websocket.send_json({"query": "reset password"})
        with pytest.raises(WebSocketDisconnect) as caught:
            websocket.receive_json()

    assert caught.value.code == 1013


# ── Origin / Host ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "origin", ["https://evil.example.com", "null", "https://app.example.com:8443"]
)
def test_a_foreign_origin_gets_403_even_with_a_valid_token(token: str, origin: str) -> None:
    policy = token_security(token, ALLOWED_ORIGINS="https://app.example.com")

    with _client(policy, headers=bearer(token)) as client:
        response = client.get("/categories", headers={"Origin": origin})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "origin_not_allowed"


def test_an_allowlisted_origin_and_no_origin_are_allowed(token: str) -> None:
    policy = token_security(token, ALLOWED_ORIGINS="https://app.example.com")

    with _client(policy, headers=bearer(token)) as client:
        browser = client.get("/categories", headers={"Origin": "https://APP.example.com"})
        agent = client.get("/categories")

    assert browser.status_code == agent.status_code == 200


def test_origin_is_checked_on_open_routes_too(token: str) -> None:
    with _client(token_security(token)) as client:
        response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert response.status_code == 403


def test_a_rebound_host_name_gets_421_on_a_loopback_bind(token: str) -> None:
    policy = token_security(token, ALLOWED_HOSTS="")

    with _client(policy, base_url="http://attacker.example", headers=bearer(token)) as client:
        response = client.get("/categories")

    assert response.status_code == 421


# ── rate limits ──────────────────────────────────────────────────────────────


def test_rate_limit_is_per_key_with_retry_after() -> None:
    first, second = make_api_token(), make_api_token()
    config = security_config(API_KEYS=f"a:{first},b:{second}", RATE_LIMIT_REQUESTS=2)
    policy = build_security(config, bind_host="127.0.0.1")

    with _client(policy) as client:
        statuses = [client.get("/categories", headers=bearer(first)).status_code for _ in range(3)]
        limited = client.get("/categories", headers=bearer(first))
        other_key = client.get("/categories", headers=bearer(second))

    assert statuses == [200, 200, 429]
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert other_key.status_code == 200


def test_spoofed_forwarded_for_does_not_change_the_key_limit(token: str) -> None:
    policy = token_security(token, RATE_LIMIT_REQUESTS=1)

    with _client(policy, headers=bearer(token)) as client:
        client.get("/categories", headers={"X-Forwarded-For": "203.0.113.1"})
        again = client.get("/categories", headers={"X-Forwarded-For": "203.0.113.2"})

    assert again.status_code == 429


def test_failed_sign_ins_are_limited_per_client_address(token: str) -> None:
    policy = token_security(token, RATE_LIMIT_ANON_REQUESTS=3)

    with _client(policy) as client:
        statuses = [
            client.get("/categories", headers=bearer(make_api_token())).status_code
            for _ in range(4)
        ]

    assert statuses == [401, 401, 401, 429]


# ── no token passthrough ─────────────────────────────────────────────────────


def _echo_app(policy: SecurityPolicy) -> Starlette:
    async def echo(request: Request) -> JSONResponse:
        principal = request.scope["state"]["principal"]
        return JSONResponse({"headers": dict(request.headers), "key_id": principal.key_id})

    return Starlette(
        routes=[Route("/echo", echo)],
        middleware=[Middleware(SecurityMiddleware, policy=policy)],
    )


def test_the_authorization_header_never_reaches_the_app(token: str) -> None:
    with TestClient(_echo_app(token_security(token)), headers=bearer(token)) as client:
        body = client.get("/echo").json()

    assert "authorization" not in body["headers"]
    assert token not in str(body)
    assert body["key_id"] == "key:tests"


def test_outbound_escalation_webhook_never_carries_the_callers_token(token: str) -> None:
    sent: list[httpx.Request] = []

    def receiver(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(204)

    sink = WebhookSink(
        url="https://hooks.example.com/escalations",
        secret=SecretStr(make_api_token()),
        transport=httpx.MockTransport(receiver),
    )
    app = create_app(static_lifespan(_service(sink=sink)), security=token_security(token))

    with TestClient(app, headers=bearer(token)) as client:
        response = client.post("/escalations", json={"customer_goal": "talk to a human"})
        deadline = time.monotonic() + 5
        while not sent and time.monotonic() < deadline:
            time.sleep(0.02)

    assert response.status_code == 201
    assert sent, "the webhook was not called"
    for request in sent:
        assert "authorization" not in request.headers
        assert token not in str(request.url)
        assert all(token not in value for value in request.headers.values())
        assert token.encode() not in request.content


# ── OAuth on the REST surface: PRM + challenges ──────────────────────────────

_OAUTH = {
    "AUTH_MODE": "oauth",
    "OAUTH_ISSUER": "https://auth.example.com",
    "OAUTH_RESOURCE_URL": "https://mcp.example.com/mcp",
    "OAUTH_JWKS_URL": "https://auth.example.com/.well-known/jwks.json",
}
_PRM_URL = "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"


def _oauth_client(**overrides: Any) -> TestClient:  # noqa: ANN401
    fetcher = FakeFetcher([_jwk(RSA_KEY, "rsa-1"), _jwk(EC_KEY, "ec-1")])
    config = security_config(**{**_OAUTH, **overrides})
    return _client(build_security(config, bind_host="127.0.0.1", jwks_fetcher=fetcher))


@pytest.mark.parametrize(
    "path",
    ["/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"],
)
def test_protected_resource_metadata_is_served_openly(path: str) -> None:
    with _oauth_client() as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json()["resource"] == "https://mcp.example.com/mcp"
    assert response.json()["authorization_servers"] == ["https://auth.example.com"]


def test_401_points_at_the_resource_metadata() -> None:
    with _oauth_client(OAUTH_REQUIRED_SCOPES="support.read") as client:
        response = client.post("/kb/search", json={"query": "x"})

    challenge = response.headers["WWW-Authenticate"]
    assert response.status_code == 401
    assert f'resource_metadata="{_PRM_URL}"' in challenge
    assert 'scope="support.read"' in challenge


def test_a_valid_jwt_is_accepted_and_wrong_audience_refused() -> None:
    with _oauth_client() as client:
        ok = client.get("/categories", headers=bearer(_sign(_claims())))
        other = client.get(
            "/categories", headers=bearer(_sign(_claims(aud="https://other.example.com")))
        )

    assert ok.status_code == 200
    assert other.status_code == 401
    assert other.json()["error"]["message"] == INVALID_TOKEN_DESCRIPTION


def test_insufficient_scope_is_403_with_a_scope_challenge() -> None:
    with _oauth_client(OAUTH_REQUIRED_SCOPES="support.admin") as client:
        response = client.get("/categories", headers=bearer(_sign(_claims())))

    challenge = response.headers["WWW-Authenticate"]
    assert response.status_code == 403
    assert 'error="insufficient_scope"' in challenge
    assert 'scope="support.admin"' in challenge
    assert f'resource_metadata="{_PRM_URL}"' in challenge


def test_static_keys_are_refused_in_pure_oauth_mode() -> None:
    with _oauth_client(API_KEYS=f"k:{make_api_token()}") as client:
        response = client.get("/categories", headers=bearer(make_api_token()))

    assert response.status_code == 401


def test_both_mode_accepts_a_static_key_and_a_jwt() -> None:
    key = make_api_token()

    with _oauth_client(AUTH_MODE="both", API_KEYS=f"crm:{key}") as client:
        by_key = client.get("/categories", headers=bearer(key))
        by_jwt = client.get("/categories", headers=bearer(_sign(_claims())))

    assert by_key.status_code == by_jwt.status_code == 200


def test_a_jwks_outage_answers_503_not_200() -> None:
    fetcher = FakeFetcher([], fail=True)
    policy = build_security(security_config(**_OAUTH), bind_host="127.0.0.1", jwks_fetcher=fetcher)

    with _client(policy) as client:
        response = client.get("/categories", headers=bearer(_sign(_claims())))

    assert response.status_code == 503
    assert "Retry-After" in response.headers


def test_a_verifier_bug_fails_closed(token: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = token_security(token)
    assert policy.verifier is not None

    async def broken(_token: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(policy.verifier, "verify", broken)
    with _client(policy, headers=bearer(token)) as client:
        response = client.get("/categories")

    assert response.status_code == 401


def test_jwt_decoding_never_happens_for_none_alg_on_rest() -> None:
    forged = jwt.encode(_claims(), key=None, algorithm="none")

    with _oauth_client() as client:
        response = client.get("/categories", headers=bearer(forged))

    assert response.status_code == 401
