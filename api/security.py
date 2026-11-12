"""ASGI security middleware shared by the REST/WebSocket app and the MCP HTTP app.

For every HTTP request and WebSocket handshake, in this order:

1. **Host** not allowlisted → 421 (DNS-rebinding protection; see
   :mod:`faq_core.security.origins`).
2. **Origin** present but not allowlisted → 403.
3. Tokens in the query string (``?access_token=``) → 400 (never accepted).
4. **Open paths** (``/health``, OAuth protected-resource metadata) skip
   authentication but are rate limited per client IP.
5. **Authentication**: exactly one ``Authorization: Bearer <token>`` header,
   verified by the policy's verifier. Failure → 401 with an RFC 6750
   ``WWW-Authenticate`` challenge (``resource_metadata`` in OAuth modes), 403
   for a missing scope, 503 when keys cannot be fetched. Failed attempts count
   against the client IP's limit.
6. **Rate limit** per authenticated key → 429 with ``Retry-After``.
7. The ``Authorization`` header is **removed** from the scope before the app
   runs, so no route, tool or outbound client can forward the caller's token
   (no token passthrough); the verified :class:`Principal` is placed in
   ``scope["state"]["principal"]`` instead.

WebSocket handshakes are checked before ``accept``; a refusal closes with
code 1008 (the server answers the handshake with HTTP 403). Each message on an
open WebSocket counts against the key's rate limit; over the limit the socket
is closed with 1013 (try again later).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from starlette.datastructures import Headers, QueryParams
from starlette.responses import JSONResponse
from starlette.routing import Route

from faq_core.security.errors import AuthError, invalid_token
from faq_core.security.policy import REALM
from faq_core.security.tokens import Principal, is_well_formed_token
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from faq_core.security.policy import SecurityPolicy
    from faq_core.security.ratelimit import SlidingWindowRateLimiter

    UserBinder = Callable[[Principal], object]

_log = get_logger(__name__)

WS_POLICY_VIOLATION: Final = 1008
WS_TRY_AGAIN_LATER: Final = 1013
AUTH_UNAVAILABLE_RETRY_AFTER: Final = 30


class SecurityMiddleware:
    """Enforces a :class:`~faq_core.security.policy.SecurityPolicy` on an ASGI app."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        policy: SecurityPolicy,
        open_paths: frozenset[str] = frozenset(),
        bind_user: UserBinder | None = None,
    ) -> None:
        """Args:
        app: The wrapped application.
        policy: What to enforce.
        open_paths: Exact paths served without authentication.
        bind_user: Builds ``scope["user"]`` from the principal (the MCP
            transport binds sessions to it).
        """
        self.app = app
        self.policy = policy
        self.open_paths = open_paths | frozenset(policy.resource_metadata_paths)
        self.bind_user = bind_user

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        client_key = _client_key(scope)
        path: str = scope["path"]
        if not self.policy.origins.host_allowed(headers.get("host")):
            _log.warning("request refused: host not allowed", extra={"path": path})
            await self._refuse(scope, receive, send, 421, "host_not_allowed", "Host not allowed")
            return
        if not self.policy.origins.origin_allowed(headers.get("origin")):
            _log.warning(
                "request refused: origin not allowed",
                extra={"path": path, "origin": (headers.get("origin") or "")[:120]},
            )
            await self._refuse(
                scope, receive, send, 403, "origin_not_allowed", "Origin not allowed"
            )
            return
        if "access_token" in QueryParams(scope.get("query_string", b"")):
            await self._refuse(
                scope,
                receive,
                send,
                400,
                "invalid_request",
                "access tokens are not accepted in the URL; use the Authorization header",
            )
            return
        if path in self.open_paths:
            if await self._limited(self.policy.anon_limiter, client_key, scope, receive, send):
                return
            await self.app(scope, receive, send)
            return
        principal = await self._authenticate(headers, client_key, scope, receive, send)
        if principal is None:
            return
        if await self._limited(self.policy.key_limiter, principal.key_id, scope, receive, send):
            return
        if scope["type"] == "websocket" and self.policy.key_limiter is not None:
            receive = _metered_receive(receive, send, self.policy.key_limiter, principal.key_id)
        await self.app(self._authenticated_scope(scope, principal), receive, send)

    async def _authenticate(
        self, headers: Headers, client_key: str, scope: Scope, receive: Receive, send: Send
    ) -> Principal | None:
        verifier = self.policy.verifier
        if verifier is None:  # AUTH_MODE=none (loopback only, enforced at startup)
            return Principal(key_id=client_key, method="none", client_id="anonymous")
        try:
            principal = await verifier.verify(_bearer_token(headers))
        except AuthError as err:
            await self._auth_failed(err, client_key, scope, receive, send)
            return None
        except Exception as err:  # noqa: BLE001 — trust boundary: any verifier bug fails closed
            _log.error(
                "token verification raised unexpectedly; refusing the request",
                extra={"path": scope["path"], "error_type": type(err).__name__},
            )
            await self._auth_failed(
                invalid_token("verifier_error"), client_key, scope, receive, send
            )
            return None
        _log.debug("request authenticated", extra={"key_id": principal.key_id})
        return principal

    async def _auth_failed(
        self, err: AuthError, client_key: str, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Count the failure against the client IP, log its reason, send the challenge."""
        if await self._limited(self.policy.anon_limiter, client_key, scope, receive, send):
            return
        _log.warning(
            "request refused: authentication failed",
            extra={"path": scope["path"], "reason": err.reason, "status": err.status_code},
        )
        await self._challenge(scope, receive, send, err)

    def _authenticated_scope(self, scope: Scope, principal: Principal) -> Scope:
        """A copy of *scope* without the Authorization header, carrying *principal*."""
        authed = dict(scope)
        authed["headers"] = [
            (name, value) for name, value in scope["headers"] if name.lower() != b"authorization"
        ]
        authed["state"] = {**scope.get("state", {}), "principal": principal}
        if self.bind_user is not None:
            authed["user"] = self.bind_user(principal)
        return authed

    async def _limited(
        self,
        limiter: SlidingWindowRateLimiter | None,
        key: str,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> bool:
        """Count one request for *key*; refuse it (and return True) when over the limit."""
        if limiter is None:
            return False
        decision = limiter.check(key)
        if decision.allowed:
            return False
        _log.warning("request refused: rate limited", extra={"path": scope["path"]})
        await self._refuse(
            scope,
            receive,
            send,
            429,
            "rate_limited",
            "too many requests; retry later",
            headers={"Retry-After": str(decision.retry_after)},
            extra={"retry_after": decision.retry_after},
            ws_code=WS_TRY_AGAIN_LATER,
        )
        return True

    async def _challenge(self, scope: Scope, receive: Receive, send: Send, err: AuthError) -> None:
        headers: dict[str, str] = {}
        if err.status_code == 503:
            headers["Retry-After"] = str(AUTH_UNAVAILABLE_RETRY_AFTER)
        else:
            headers["WWW-Authenticate"] = www_authenticate(self.policy, err)
        await self._refuse(
            scope,
            receive,
            send,
            err.status_code,
            err.error or "unauthorized",
            err.description,
            headers=headers,
        )

    @staticmethod
    async def _refuse(
        scope: Scope,
        receive: Receive,
        send: Send,
        status: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, object] | None = None,
        ws_code: int = WS_POLICY_VIOLATION,
    ) -> None:
        if scope["type"] == "websocket":
            # Closing before accept makes the server answer the handshake with 403.
            await send({"type": "websocket.close", "code": ws_code, "reason": code})
            return
        body = {"error": {"code": code, "message": message, **(extra or {})}}
        response = JSONResponse(body, status_code=status, headers=headers)
        await response(scope, receive, send)


def _bearer_token(headers: Headers) -> str:
    """The single bearer token of the request.

    Raises:
        AuthError: no credentials or another scheme than bearer (401 without an
            error code), several Authorization headers (400
            invalid_request), or a syntactically invalid token (401
            invalid_token — the same answer as an unknown token).
    """
    values = headers.getlist("authorization")
    if not values:
        raise AuthError("authentication required", error=None)
    if len(values) > 1:
        raise AuthError(
            "send exactly one Authorization header", error="invalid_request", status_code=400
        )
    scheme, _, token = values[0].strip().partition(" ")
    if scheme.lower() != "bearer":
        raise AuthError("use the bearer authorization scheme", error=None, reason="wrong_scheme")
    token = token.strip()
    if not is_well_formed_token(token):
        raise invalid_token("malformed_token")
    return token


def www_authenticate(policy: SecurityPolicy, err: AuthError) -> str:
    """The RFC 6750 / RFC 9728 challenge for *err* (fixed strings only, no token)."""
    params = [f'realm="{REALM}"']
    if err.error is not None:
        params.append(f'error="{err.error}"')
        params.append(f'error_description="{_quoted(err.description)}"')
    scopes = err.required_scopes or (policy.oauth.required_scopes if policy.oauth else ())
    if scopes:
        params.append(f'scope="{_quoted(" ".join(scopes))}"')
    if policy.resource_metadata_url is not None:
        params.append(f'resource_metadata="{policy.resource_metadata_url}"')
    return "Bearer " + ", ".join(params)


def _quoted(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _client_key(scope: Scope) -> str:
    client = scope.get("client")
    host = client[0] if client else "unknown"
    return f"ip:{host}"


def _metered_receive(
    receive: Receive, send: Send, limiter: SlidingWindowRateLimiter, key: str
) -> Receive:
    """Count each WebSocket message; close the socket (1013) when over the limit."""

    async def metered() -> Message:
        message = await receive()
        if message["type"] == "websocket.receive" and not limiter.check(key).allowed:
            _log.warning("websocket closed: rate limited")
            await send({"type": "websocket.close", "code": WS_TRY_AGAIN_LATER})
            return {"type": "websocket.disconnect", "code": WS_TRY_AGAIN_LATER}
        return message

    return metered


def protected_resource_metadata_routes(policy: SecurityPolicy) -> list[Route]:
    """RFC 9728 metadata routes (path-inserted and root); empty without OAuth."""
    document = policy.protected_resource_metadata()
    if document is None:
        return []

    async def metadata(_request: Request) -> JSONResponse:
        return JSONResponse(document, headers={"Cache-Control": "public, max-age=3600"})

    return [Route(path, metadata, methods=["GET"]) for path in policy.resource_metadata_paths]
