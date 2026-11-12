"""LLMGateway: factory per mode/provider, the two adapters over fake HTTP, NullGateway,
and the circuit breaker. No network and no real keys: the HTTP boundary is a mock
transport (httpx for the OpenAI-compatible adapter, httpx2 for the Anthropic SDK).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from pydantic import SecretStr

from faq_core.config import AppConfig
from faq_core.errors import AnswerGenerationUnavailableError, LLMUnavailableError
from faq_core.llm.anthropic_gateway import AnthropicGateway
from faq_core.llm.factory import create_llm_gateway
from faq_core.llm.gateway import NullGateway
from faq_core.llm.openai_compatible_gateway import OpenAICompatibleGateway
from faq_core.llm.resilience import CircuitBreaker, CircuitState
from faq_core.models import LLMMode

_FAKE_CREDENTIAL = "test-credential-123"


def _config(**values: object) -> AppConfig:
    return AppConfig(_env_file=None, **values)  # type: ignore[arg-type]


# ── factory ──────────────────────────────────────────────────────────────────


def test_client_mode_builds_a_null_gateway_without_any_key() -> None:
    gateway = create_llm_gateway(_config(LLM_MODE="client"))

    assert isinstance(gateway, NullGateway)
    assert not gateway.can_generate


def test_anthropic_provider_builds_the_anthropic_adapter() -> None:
    gateway = create_llm_gateway(_config(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="k"))

    assert isinstance(gateway, AnthropicGateway)
    assert gateway.name == "anthropic:claude-opus-5-5"


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
        ("groq", "https://api.groq.com/openai/v1"),
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("mistral", "https://api.mistral.ai/v1"),
        ("cohere", "https://api.cohere.com/compatibility/v1"),
        ("together", "https://api.together.xyz/v1"),
    ],
)
def test_hosted_openai_compatible_providers_use_their_preset_url(
    provider: str, expected_url: str
) -> None:
    gateway = create_llm_gateway(_config(LLM_PROVIDER=provider, LLM_API_KEY="k"))

    assert isinstance(gateway, OpenAICompatibleGateway)
    assert gateway.base_url == expected_url


@pytest.mark.parametrize(
    ("provider", "setting", "url"),
    [
        ("ollama", "OLLAMA_BASE_URL", "http://remote-host:11434/v1"),
        ("vllm", "VLLM_BASE_URL", "http://gpu-box:8000/v1"),
        ("lmstudio", "LMSTUDIO_BASE_URL", "http://laptop:1234/v1"),
    ],
)
def test_local_runtimes_use_their_configurable_base_url_over_plain_http(
    provider: str, setting: str, url: str
) -> None:
    gateway = create_llm_gateway(_config(LLM_PROVIDER=provider, LLM_MODEL="m", **{setting: url}))

    assert isinstance(gateway, OpenAICompatibleGateway)
    assert gateway.base_url == url


def test_offline_mode_builds_a_generating_gateway() -> None:
    gateway = create_llm_gateway(_config(LLM_MODE="offline", LLM_MODEL="llama3.2"))

    assert isinstance(gateway, OpenAICompatibleGateway)
    assert gateway.name == "ollama:llama3.2"


def test_hosted_provider_without_key_fails_fast() -> None:
    with pytest.raises(ValueError, match="requires LLM_API_KEY"):
        create_llm_gateway(_config(LLM_PROVIDER="groq"))


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        create_llm_gateway(_config(LLM_PROVIDER="not-a-real-provider", ANTHROPIC_API_KEY="k"))


def test_hosted_url_override_must_use_tls() -> None:
    config = _config(LLM_PROVIDER="openai", LLM_API_KEY="k", LLM_BASE_URL="http://proxy.lan/v1")

    with pytest.raises(ValueError, match="https"):
        create_llm_gateway(config)


# ── OpenAI-compatible adapter ────────────────────────────────────────────────


def _openai_gateway(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = _FAKE_CREDENTIAL,
    breaker: CircuitBreaker | None = None,
) -> OpenAICompatibleGateway:
    return OpenAICompatibleGateway(
        provider="groq",
        base_url="https://api.example.test/v1/",
        model="m-1",
        max_tokens=512,
        breaker=breaker or CircuitBreaker("llm:groq"),
        api_key=SecretStr(api_key),
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )


def _chat_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


async def test_openai_compatible_generate_sends_chat_completions_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _chat_reply("Five days.")

    gateway = _openai_gateway(handler)

    answer = await gateway.generate_answer("how long?", ["Q: shipping\nA: five days"])

    assert answer == "Five days."
    [request] = seen
    assert str(request.url) == "https://api.example.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {_FAKE_CREDENTIAL}"
    body = json.loads(request.content)
    assert body["model"] == "m-1"
    assert body["max_tokens"] == 512
    assert body["messages"][0]["role"] == "system"
    assert "five days" in body["messages"][1]["content"]
    assert "how long?" in body["messages"][1]["content"]


async def test_openai_compatible_without_key_sends_no_authorization_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _chat_reply("search_faq\n")

    gateway = _openai_gateway(handler, api_key="")

    assert await gateway.classify_intent("reset my password") == "search_faq"
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "bad key"}),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
    ],
)
async def test_openai_compatible_failures_become_llm_unavailable_without_secrets(
    response: httpx.Response,
) -> None:
    gateway = _openai_gateway(lambda _request: response)

    with pytest.raises(LLMUnavailableError) as excinfo:
        await gateway.generate_answer("q", ["ctx"])

    assert _FAKE_CREDENTIAL not in json.dumps(excinfo.value.to_dict())
    assert excinfo.value.to_dict()["error"]["code"] == "llm_unavailable"


async def test_open_circuit_fails_fast_without_calling_the_provider() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    gateway = _openai_gateway(handler, breaker=CircuitBreaker("llm:groq", failure_threshold=2))
    for _ in range(2):
        with pytest.raises(LLMUnavailableError):
            await gateway.generate_answer("q", [])

    with pytest.raises(LLMUnavailableError, match="circuit breaker open"):
        await gateway.generate_answer("q", [])

    assert calls == 2
    assert gateway.circuit_breaker.state is CircuitState.OPEN


# ── Anthropic adapter (official SDK over a mock httpx2 transport) ────────────


def _anthropic_message(
    text: str = "Use Forgot Password.", stop_reason: str = "end_turn"
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _anthropic_gateway(
    handler: Callable[[httpx2.Request], httpx2.Response], *, model: str = "claude-opus-5-5"
) -> AnthropicGateway:
    client = anthropic.AsyncAnthropic(
        api_key=_FAKE_CREDENTIAL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicGateway(
        api_key=SecretStr(_FAKE_CREDENTIAL),
        model=model,
        max_tokens=16_000,
        breaker=CircuitBreaker("llm:anthropic"),
        client=client,
    )


async def test_anthropic_generate_uses_messages_api_with_refusal_fallback() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_anthropic_message())

    gateway = _anthropic_gateway(handler)

    answer = await gateway.generate_answer("reset password?", ["Q: reset\nA: Forgot Password"])
    await gateway.aclose()

    assert answer == "Use Forgot Password."
    [request] = seen
    assert request.url.path == "/v1/messages"
    assert request.headers["authorization"] == f"Bearer {_FAKE_CREDENTIAL}"  # SDK 2.x auth header
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16_000
    assert body["fallbacks"] == "default"
    assert "reset password?" in body["messages"][0]["content"]


async def test_anthropic_models_without_fallback_support_use_the_plain_endpoint() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_anthropic_message("search_faq"))

    gateway = _anthropic_gateway(handler, model="claude-haiku-5-5")

    assert await gateway.classify_intent("hi") == "search_faq"
    body = json.loads(seen[0].content)
    assert "fallbacks" not in body
    assert "anthropic-beta" not in seen[0].headers


async def test_anthropic_refusal_becomes_llm_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_anthropic_message("", stop_reason="refusal"))

    gateway = _anthropic_gateway(handler)

    with pytest.raises(LLMUnavailableError) as excinfo:
        await gateway.generate_answer("q", [])

    assert excinfo.value.details["reason"] == "refusal"


async def test_anthropic_http_error_is_structured_and_leaks_no_key() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "x"}}
        )

    gateway = _anthropic_gateway(handler)

    with pytest.raises(LLMUnavailableError, match="HTTP 401") as excinfo:
        await gateway.generate_answer("q", [])

    assert _FAKE_CREDENTIAL not in json.dumps(excinfo.value.to_dict())


# ── NullGateway ──────────────────────────────────────────────────────────────


async def test_null_gateway_refuses_to_generate_with_a_structured_error() -> None:
    gateway = NullGateway(LLMMode.CLIENT)

    with pytest.raises(AnswerGenerationUnavailableError) as excinfo:
        await gateway.generate_answer("q", ["ctx"])
    with pytest.raises(AnswerGenerationUnavailableError):
        await gateway.classify_intent("q")

    error = excinfo.value.to_dict()["error"]
    assert error["code"] == "answer_generation_unavailable"
    assert error["llm_mode"] == "client"
    assert error["use_instead"] == "search_kb"
    await gateway.aclose()


# ── circuit breaker ──────────────────────────────────────────────────────────


def test_breaker_half_opens_after_the_recovery_timeout_and_closes_on_success() -> None:
    now = 0.0
    transitions: list[tuple[CircuitState, CircuitState]] = []
    breaker = CircuitBreaker(
        "llm:t",
        failure_threshold=1,
        recovery_timeout=30.0,
        clock=lambda: now,
        on_state_change=lambda _name, old, new: transitions.append((old, new)),
    )

    breaker.record_failure()
    assert not breaker.allow_request()
    now = 31.0
    assert breaker.allow_request()
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    assert transitions == [
        (CircuitState.CLOSED, CircuitState.OPEN),
        (CircuitState.OPEN, CircuitState.HALF_OPEN),
        (CircuitState.HALF_OPEN, CircuitState.CLOSED),
    ]


def test_breaker_reopens_when_the_half_open_trial_fails() -> None:
    now = 0.0
    breaker = CircuitBreaker("llm:t", failure_threshold=3, recovery_timeout=5.0, clock=lambda: now)
    for _ in range(3):
        breaker.record_failure()
    now = 6.0
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN


def test_breaker_rejects_a_zero_threshold() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker("llm:t", failure_threshold=0)


# ── Phase 4: language, caveat, self-check ────────────────────────────────────


def test_system_prompt_asks_for_the_customers_language_and_passages_only() -> None:
    from faq_core.llm.prompts import RAG_SYSTEM

    assert "same language as the customer's question" in RAG_SYSTEM
    assert "only the numbered passages" in RAG_SYSTEM


def test_caveat_note_is_added_only_when_asked() -> None:
    from faq_core.llm.gateway import build_rag_prompt
    from faq_core.llm.prompts import CAVEAT_NOTE

    assert CAVEAT_NOTE in build_rag_prompt("q", ["p"], caveat=True)
    assert CAVEAT_NOTE not in build_rag_prompt("q", ["p"])


@pytest.mark.parametrize(
    ("text", "verdict"),
    [("SUPPORTED", True), ("unsupported.", False), ("Verdict: **SUPPORTED**", True), ("", None)],
)
def test_parse_self_check(text: str, verdict: bool | None) -> None:
    from faq_core.llm.gateway import parse_self_check

    assert parse_self_check(text) is verdict


async def test_generate_and_verify_send_the_language_rule_and_draft() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _chat_reply("UNSUPPORTED" if len(seen) == 2 else "Cinq jours.")

    gateway = _openai_gateway(handler)

    answer = await gateway.generate_answer("Combien de temps ?", ["Q: shipping\nA: 5 days"])
    supported = await gateway.verify_answer(
        "Combien de temps ?", ["Q: shipping\nA: 5 days"], answer
    )

    assert (answer, supported) == ("Cinq jours.", False)
    generate_system = seen[0]["messages"][0]["content"]  # type: ignore[index]
    assert "same language" in generate_system
    verify_user = seen[1]["messages"][1]["content"]  # type: ignore[index]
    assert "Draft answer:\nCinq jours." in verify_user


async def test_null_gateway_never_self_checks() -> None:
    from faq_core.errors import AnswerGenerationUnavailableError
    from faq_core.llm.gateway import NullGateway

    with pytest.raises(AnswerGenerationUnavailableError):
        await NullGateway().verify_answer("q", ["p"], "a")
