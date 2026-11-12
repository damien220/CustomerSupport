"""Builds the :class:`~faq_core.llm.gateway.LLMGateway` for the configured mode.

The only place an LLM adapter is chosen. ``LLM_MODE=client`` gets a
:class:`~faq_core.llm.gateway.NullGateway`; ``api`` and ``offline`` resolve the
``LLM_PROVIDER`` preset (:mod:`faq_core.llm.presets`) and build the adapter for
the preset's wire protocol — no per-provider branching. Called from
:func:`bootstrap.build_runtime` after Langfuse is initialised (the adapters
carry ``@observe`` spans).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faq_core.llm.presets import WireProtocol, api_key_for, resolve_llm_settings
from faq_core.llm.resilience import CircuitBreaker
from observability.alerting import build_circuit_breaker_alert

if TYPE_CHECKING:
    from faq_core.config import AppConfig
    from faq_core.llm.gateway import LLMGateway
    from faq_core.llm.presets import ResolvedLLM


def create_llm_gateway(config: AppConfig) -> LLMGateway:
    """Return the gateway for ``LLM_MODE`` / ``LLM_PROVIDER``.

    Raises:
        ValueError: unknown provider, missing model / base URL, or a required
            API key that is not configured (fail fast at startup).
    """
    from faq_core.llm.gateway import NullGateway

    if not config.LLM_MODE.generates_answers:
        return NullGateway(config.LLM_MODE)

    resolved = resolve_llm_settings(config)
    breaker = CircuitBreaker(
        f"llm:{resolved.provider}", on_state_change=build_circuit_breaker_alert(config)
    )
    if resolved.protocol is WireProtocol.ANTHROPIC:
        return _anthropic(config, resolved, breaker)
    return _openai_compatible(config, resolved, breaker)


def _anthropic(config: AppConfig, resolved: ResolvedLLM, breaker: CircuitBreaker) -> LLMGateway:
    from faq_core.llm.anthropic_gateway import AnthropicGateway

    return AnthropicGateway(
        api_key=api_key_for(config, resolved),
        model=resolved.model,
        max_tokens=resolved.max_tokens,
        breaker=breaker,
        base_url=resolved.base_url,
        timeout=config.LLM_TIMEOUT,
        refusal_fallback=config.ANTHROPIC_REFUSAL_FALLBACK,
    )


def _openai_compatible(
    config: AppConfig, resolved: ResolvedLLM, breaker: CircuitBreaker
) -> LLMGateway:
    from faq_core.llm.openai_compatible_gateway import OpenAICompatibleGateway

    return OpenAICompatibleGateway(
        provider=resolved.provider,
        base_url=resolved.base_url,
        model=resolved.model,
        max_tokens=resolved.max_tokens,
        breaker=breaker,
        api_key=api_key_for(config, resolved),
        # Local runtimes often listen on plain http on the operator's own network.
        allow_plain_http=resolved.local,
        timeout=config.LLM_TIMEOUT,
    )
