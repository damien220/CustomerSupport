"""Gateway adapter for OpenAI-compatible ``POST {base_url}/chat/completions`` APIs.

Covers hosted providers (OpenAI, Gemini, Groq, OpenRouter, ...) and local
runtimes (Ollama, vLLM, llama.cpp, LM Studio). Uses the shared
:func:`~faq_core.retrieval.http_json.post_json` helper (httpx, already a core
dependency): timeout, retries with backoff on transport errors / 429 / 5xx, and
errors that never carry request headers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from faq_core.errors import LLMUnavailableError
from faq_core.llm.gateway import ChatCompletionGateway
from faq_core.retrieval.http_json import RemoteServiceError, post_json, require_tls

if TYPE_CHECKING:
    import httpx
    from pydantic import SecretStr

    from faq_core.llm.resilience import CircuitBreaker


class OpenAICompatibleGateway(ChatCompletionGateway):
    """:class:`~faq_core.llm.gateway.LLMGateway` over ``/chat/completions``.

    Args:
        provider: Preset name (used in logs and error messages).
        base_url: API root, e.g. ``https://api.openai.com/v1``.
        model: Model id.
        max_tokens: Output cap for generated answers.
        breaker: Circuit breaker for this provider.
        api_key: Sent as a bearer token when non-empty (local runtimes usually
            need none).
        allow_plain_http: Permit ``http://`` to non-loopback hosts — for local
            runtimes on the operator's own network. Otherwise non-local hosts
            must use https.
        timeout: Seconds per request.
        max_retries: Retries after the first attempt.
        transport: Custom httpx transport (tests).

    Raises:
        ValueError: *base_url* is not http(s), or plain http to a remote host
            without *allow_plain_http*.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        max_tokens: int,
        breaker: CircuitBreaker,
        api_key: SecretStr,
        allow_plain_http: bool = False,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if allow_plain_http:
            if not base_url.startswith(("http://", "https://")):
                raise ValueError("LLM_BASE_URL must be an http(s) URL")
        else:
            require_tls(base_url, setting="LLM_BASE_URL")
        super().__init__(provider=provider, model=model, max_tokens=max_tokens, breaker=breaker)
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/chat/completions"
        key = api_key.get_secret_value()
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _complete(self, *, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            body = await post_json(
                self._url,
                payload,
                service=self._provider,
                headers=self._headers,
                request_timeout=self._timeout,
                max_retries=self._max_retries,
                transport=self._transport,
            )
        except RemoteServiceError as err:
            raise LLMUnavailableError(str(err), provider=self._provider) from err
        return _reply_text(body, provider=self._provider)


def _reply_text(body: object, *, provider: str) -> str:
    """Extract ``choices[0].message.content`` from a chat-completions response."""
    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as err:
        raise LLMUnavailableError(
            f"{provider} returned an unexpected response shape", provider=provider
        ) from err
    if not isinstance(content, str):
        raise LLMUnavailableError(f"{provider} returned no text content", provider=provider)
    return content
