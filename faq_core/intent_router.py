"""LLM-based intent routing for free-form (non-MCP) input.

The MCP surface lets the client model pick tools via the protocol, so no routing
is needed there. Free-form surfaces can instead classify a query into one of the
tool/action names that :class:`~faq_core.service.SupportService` exposes via
:meth:`LLMGateway.classify_intent`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from observability.log import get_logger
from observability.tracing import observe

if TYPE_CHECKING:
    from faq_core.llm.gateway import LLMGateway

_log = get_logger(__name__)

# Maps an LLM-classified intent (one of the four MCP tool names the
# INTENT_SYSTEM prompt can emit) to the action that handles free-form input.
# ``get_faq_entry`` needs an explicit id that free text does not carry, so it
# falls through to search.
INTENT_TO_ACTION: Final[dict[str, str]] = {
    "search_faq": "search_faq",
    "add_faq_entry": "add_faq_entry",
    "list_faq_categories": "list_faq_categories",
    "get_faq_entry": "search_faq",
}

#: Action returned when the LLM emits an unrecognised intent string.
DEFAULT_ACTION: Final[str] = "search_faq"


class IntentRouter:
    """Classify a query into an action name via the LLM."""

    def __init__(self, llm: LLMGateway) -> None:
        self._llm = llm

    @observe(name="intent_router.route")
    async def route(self, query: str) -> str:
        """Return the action name for *query*.

        Unrecognised intents fall back to :data:`DEFAULT_ACTION` and are logged
        so that frequent fallbacks (a prompt or model regression) are visible.
        """
        intent = (await self._llm.classify_intent(query)).strip()
        action = INTENT_TO_ACTION.get(intent)
        if action is None:
            _log.info("intent_router fallback to default", extra={"raw_intent": intent})
            return DEFAULT_ACTION
        return action
