"""Gateway adapter for the Anthropic Messages API, through the official ``anthropic`` SDK.

The SDK (already a core dependency) owns the timeout and the retries (connection
errors, 408/409/429/5xx with backoff). For models that support it, a request
the model declines is re-run server-side on a fallback model
(``fallbacks="default"``, beta ``server-side-fallback-2026-07-01``); a refusal
that survives the fallback becomes :class:`~faq_core.errors.LLMUnavailableError`.

The SDK is imported when the gateway is constructed, so ``import app, server``
never pays for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from faq_core.errors import LLMUnavailableError
from faq_core.llm.gateway import ChatCompletionGateway

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from pydantic import SecretStr

    from faq_core.llm.resilience import CircuitBreaker

PROVIDER: Final = "anthropic"
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
FALLBACK_MODELS: Final = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
"""Models whose requests accept ``fallbacks="default"`` on the Claude API."""


class AnthropicGateway(ChatCompletionGateway):
    """:class:`~faq_core.llm.gateway.LLMGateway` over the Anthropic Messages API.

    Args:
        api_key: ``ANTHROPIC_API_KEY``; read only here, never stored in clear.
        model: Model id, e.g. ``claude-opus-5-5``.
        max_tokens: Output cap for generated answers (thinking counts toward it).
        breaker: Circuit breaker for this provider.
        base_url: API root (``None`` = the SDK default).
        timeout: Seconds per request.
        max_retries: SDK retries after the first attempt.
        refusal_fallback: Send ``fallbacks="default"`` when *model* supports it.
        client: A pre-built SDK client (tests); *api_key*/*base_url*/*timeout*/
            *max_retries* are then ignored.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        max_tokens: int,
        breaker: CircuitBreaker,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        refusal_fallback: bool = True,
        client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(provider=PROVIDER, model=model, max_tokens=max_tokens, breaker=breaker)
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=api_key.get_secret_value(),
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        self._client = client
        self._use_fallback = refusal_fallback and model in FALLBACK_MODELS

    async def aclose(self) -> None:
        await self._client.close()

    async def _complete(self, *, system: str, user: str, max_tokens: int) -> str:
        import anthropic
        from anthropic.types import TextBlock
        from anthropic.types.beta import BetaTextBlock

        try:
            if self._use_fallback:
                message: (
                    anthropic.types.Message | anthropic.types.beta.BetaMessage
                ) = await self._client.beta.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                message = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
        except anthropic.APIStatusError as err:
            raise LLMUnavailableError(
                f"{PROVIDER} returned HTTP {err.status_code}", provider=PROVIDER
            ) from err
        except anthropic.APIConnectionError as err:  # includes timeouts
            raise LLMUnavailableError(
                f"{PROVIDER} request failed: {type(err).__name__}", provider=PROVIDER
            ) from err
        except anthropic.APIError as err:
            raise LLMUnavailableError(
                f"{PROVIDER} request failed: {type(err).__name__}", provider=PROVIDER
            ) from err

        if message.stop_reason == "refusal":
            raise LLMUnavailableError(
                f"{PROVIDER} declined the request", provider=PROVIDER, reason="refusal"
            )
        text = "".join(
            block.text for block in message.content if isinstance(block, TextBlock | BetaTextBlock)
        )
        if not text and message.stop_reason == "max_tokens":
            raise LLMUnavailableError(
                f"{PROVIDER} hit max_tokens before answering (raise LLM_MAX_TOKENS)",
                provider=PROVIDER,
                reason="max_tokens",
            )
        return text
