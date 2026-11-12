from __future__ import annotations

import pytest

from faq_core.intent_router import DEFAULT_ACTION, IntentRouter
from tests.conftest import FakeLLMClient


@pytest.mark.parametrize(
    ("intent", "expected_action"),
    [
        ("search_faq", "search_faq"),
        ("add_faq_entry", "add_faq_entry"),
        ("list_faq_categories", "list_faq_categories"),
        ("get_faq_entry", "search_faq"),
    ],
)
async def test_route_maps_known_intents_to_actions(intent: str, expected_action: str) -> None:
    router = IntentRouter(FakeLLMClient(intent=intent))

    action = await router.route("some query")

    assert action == expected_action


async def test_route_falls_back_to_default_on_unknown_intent() -> None:
    router = IntentRouter(FakeLLMClient(intent="not_a_real_intent"))

    action = await router.route("some query")

    assert action == DEFAULT_ACTION


async def test_route_strips_whitespace_from_llm_response() -> None:
    router = IntentRouter(FakeLLMClient(intent="  list_faq_categories\n"))

    action = await router.route("show me topics")

    assert action == "list_faq_categories"


async def test_route_forwards_the_query_to_the_llm() -> None:
    llm = FakeLLMClient(intent="search_faq")
    router = IntentRouter(llm)

    await router.route("how do I reset my password")

    assert llm.classify_calls == ["how do I reset my password"]
