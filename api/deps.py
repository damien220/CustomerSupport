"""Typed FastAPI dependency aliases.

Routes never touch ``request.app.state`` directly — they declare
:data:`ServiceDep`, which keeps the injected type visible and lets tests
override it with ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from faq_core.service import SupportService


def get_service(request: Request) -> SupportService:
    """Return the service the application lifespan placed on ``app.state``."""
    return cast("SupportService", request.app.state.service)


ServiceDep = Annotated[SupportService, Depends(get_service)]
