"""Remote MCP: the support tools over Streamable HTTP, behind authentication.

``python app.py --mode mcp --transport http`` serves :func:`create_mcp_http_app`
(default ``http://127.0.0.1:8001/mcp``). It is the same :mod:`server` tool set
(names and result contracts unchanged) on the MCP Python SDK 1.x
``StreamableHTTPSessionManager`` (protocol revisions up to 2025-11-25), wrapped
in :class:`~api.security.SecurityMiddleware`:

* bearer authentication on every request (static keys and/or OAuth JWTs),
  with ``WWW-Authenticate`` + RFC 9728 protected-resource metadata in OAuth
  modes (served at ``/.well-known/oauth-protected-resource[/mcp]``);
* Origin (403) and Host (421) allowlists — also passed to the SDK's own
  transport checks as a second layer;
* per-key rate limits; the ``Authorization`` header is stripped before the
  SDK sees the request, and the verified principal becomes ``scope["user"]``
  so the SDK binds each MCP session to the credential that opened it.

The support service is built **once** in this app's lifespan and shared by all
sessions (the SDK would otherwise enter the server lifespan per session).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Mount

from api.security import SecurityMiddleware, protected_resource_metadata_routes
from faq_core.models import LLMMode
from server import create_mcp_server

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from faq_core.security.origins import HostOriginPolicy
    from faq_core.security.policy import SecurityPolicy
    from faq_core.security.tokens import Principal
    from faq_core.service import ServiceLifespan, SupportService

MCP_PATH: Final = "/mcp"


class _SharedService:
    """Holds the one running service for the per-session MCP lifespans."""

    def __init__(self) -> None:
        self.service: SupportService | None = None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[SupportService]:
        if self.service is None:
            raise RuntimeError("the support service is not running")
        yield self.service


def create_mcp_http_app(
    service_lifespan: ServiceLifespan,
    *,
    security: SecurityPolicy,
    llm_mode: LLMMode = LLMMode.API,
    operator_tools: bool = False,
) -> Starlette:
    """Build the authenticated Streamable-HTTP MCP app.

    *service_lifespan* is entered once when the app starts; *security* is
    enforced on every request and closed with the app. *operator_tools*
    (``MCP_OPERATOR_TOOLS``) adds the operator-only tools.
    """
    shared = _SharedService()
    mcp = create_mcp_server(
        shared.lifespan,
        llm_mode=llm_mode,
        transport_security=sdk_transport_security(security.origins),
        operator_tools=operator_tools,
    )
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            async with service_lifespan() as service:
                shared.service = service
                try:
                    async with mcp.session_manager.run():
                        yield
                finally:
                    shared.service = None
        finally:
            await security.aclose()

    routes = [*protected_resource_metadata_routes(security), Mount("/", app=mcp_app)]
    middleware = [Middleware(SecurityMiddleware, policy=security, bind_user=principal_user)]
    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


def principal_user(principal: Principal) -> AuthenticatedUser:
    """The SDK user for *principal* — it carries identity only, never the token."""
    claims = {"iss": principal.issuer} if principal.issuer else None
    return AuthenticatedUser(
        AccessToken(
            token="",  # the caller's token is deliberately not handed to the SDK
            client_id=principal.client_id or principal.key_id,
            scopes=sorted(principal.scopes),
            subject=principal.subject or principal.key_id,
            claims=claims,
        )
    )


def sdk_transport_security(origins: HostOriginPolicy) -> TransportSecuritySettings:
    """Mirror the Host/Origin allowlists into the SDK's transport checks.

    The SDK checks Host and Origin together, so its check is on exactly when
    the Host allowlist is (always on loopback binds).
    """
    if origins.allowed_hosts is None:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    hosts: list[str] = []
    for entry in sorted(origins.allowed_hosts):
        has_port = entry.rsplit("]", 1)[-1].count(":") == 1
        hosts.extend([entry] if has_port else [entry, f"{entry}:*"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=sorted(origins.allowed_origins),
    )
