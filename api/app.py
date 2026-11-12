"""FastAPI application factory with a plain, self-owned lifespan.

Every route except ``GET /health`` requires authentication
(:class:`~api.security.SecurityMiddleware`; Phase 5). The security policy is
built by ``bootstrap.build_security`` and passed in — there is no unauthenticated
default.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from api.security import SecurityMiddleware, protected_resource_metadata_routes
from faq_core.errors import SupportError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from faq_core.security.policy import SecurityPolicy
    from faq_core.service import ServiceLifespan

OPEN_PATHS: Final = frozenset({"/health"})
"""REST paths served without authentication (no sensitive data)."""


def create_app(service_lifespan: ServiceLifespan, *, security: SecurityPolicy) -> FastAPI:
    """Build the REST/WS app; *service_lifespan* provides the service per run.

    The service is entered when the app starts (uvicorn or ``TestClient``) and
    closed when it stops, so nothing is constructed at import time. *security*
    is enforced on every request and closed with the app.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            async with service_lifespan() as service:
                app.state.service = service
                yield
        finally:
            await security.aclose()

    app = FastAPI(title="Customer Support FAQ Assistant", lifespan=lifespan)
    app.add_exception_handler(SupportError, _support_error_handler)  # type: ignore[arg-type]
    app.include_router(router)
    app.router.routes.extend(protected_resource_metadata_routes(security))
    app.add_middleware(SecurityMiddleware, policy=security, open_paths=OPEN_PATHS)
    return app


async def _support_error_handler(_request: Request, exc: SupportError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
