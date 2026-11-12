"""``LLMGateway`` — the one interface the core uses to reach a language model.

Adapters:

* :class:`~faq_core.llm.anthropic_gateway.AnthropicGateway` — Anthropic
  Messages API through the official SDK.
* :class:`~faq_core.llm.openai_compatible_gateway.OpenAICompatibleGateway` —
  any ``/chat/completions`` server: OpenAI, Gemini, Groq, OpenRouter, Ollama,
  vLLM, llama.cpp, LM Studio, ...
* :class:`NullGateway` — ``LLM_MODE=client``: never generates; the calling
  agent writes the reply.

:class:`ChatCompletionGateway` holds what the two real adapters share: prompt
construction, the Langfuse ``@observe`` spans and the circuit breaker; an
adapter only implements :meth:`ChatCompletionGateway._complete`.

``@observe`` decides at decoration time whether to trace, so this module is
imported only by :func:`faq_core.llm.factory.create_llm_gateway`, after
``init_langfuse`` has run in :func:`bootstrap.build_runtime`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final, Protocol

from faq_core.errors import AnswerGenerationUnavailableError, LLMUnavailableError
from faq_core.llm.prompts import CAVEAT_NOTE, INTENT_SYSTEM, RAG_SYSTEM, SELF_CHECK_SYSTEM
from faq_core.models import LLMMode
from observability.tracing import observe

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faq_core.llm.resilience import CircuitBreaker

INTENT_MAX_TOKENS: Final = 1024
"""Intent labels are one word; the headroom covers models that think first."""

SELF_CHECK_MAX_TOKENS: Final = 1024
"""The verdict is one word; same headroom as intent classification."""


class LLMGateway(Protocol):
    """What the answer engine and intent router need from a language model."""

    @property
    def name(self) -> str:
        """``provider:model`` for logs, e.g. ``"anthropic:claude-opus-5-5"``."""
        ...

    @property
    def can_generate(self) -> bool:
        """``False`` for :class:`NullGateway` — callers must not ask it to generate."""
        ...

    async def generate_answer(
        self, query: str, context_docs: Sequence[str], *, caveat: bool = False
    ) -> str:
        """Answer *query* from *context_docs* (retrieved passages, best first).

        The reply is in the language of *query*. ``caveat=True`` asks the model
        to open with a short "this may be incomplete" sentence (medium
        confidence). *query* must already be PII-redacted.

        Raises:
            LLMUnavailableError: the provider failed or refused.
            AnswerGenerationUnavailableError: this gateway never generates.
        """
        ...

    async def classify_intent(self, query: str) -> str:
        """Return one intent name (see ``faq_core.llm.prompts.INTENT_SYSTEM``)."""
        ...

    async def verify_answer(self, query: str, context_docs: Sequence[str], answer: str) -> bool:
        """LLM self-check: is *answer* fully supported by *context_docs*?

        Raises:
            LLMUnavailableError: the provider failed or refused.
            AnswerGenerationUnavailableError: this gateway never generates.
        """
        ...

    async def aclose(self) -> None:
        """Release HTTP connections."""
        ...


def build_rag_prompt(query: str, context_docs: Sequence[str], *, caveat: bool = False) -> str:
    """The user turn for answer generation: numbered passages, then the question."""
    context = "\n\n".join(f"[Passage {i + 1}]\n{doc}" for i, doc in enumerate(context_docs))
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return f"{prompt}\n\n{CAVEAT_NOTE}" if caveat else prompt


def build_self_check_prompt(query: str, context_docs: Sequence[str], answer: str) -> str:
    """The user turn for the self-check: passages, question, then the draft answer."""
    return f"{build_rag_prompt(query, context_docs)}\n\nDraft answer:\n{answer}"


def parse_self_check(text: str) -> bool | None:
    """``True`` for SUPPORTED, ``False`` for UNSUPPORTED, ``None`` when unparseable."""
    words = text.strip().upper().split()
    verdict = words[-1].strip(".!*\"'") if words else ""
    if verdict == "SUPPORTED":
        return True
    if verdict == "UNSUPPORTED":
        return False
    return None


class ChatCompletionGateway(ABC):
    """Shared behaviour of the provider adapters (see module docstring).

    Args:
        provider: Preset name, e.g. ``"anthropic"`` or ``"ollama"``.
        model: Model id sent to the provider.
        max_tokens: Output cap for generated answers.
        breaker: Circuit breaker guarding this provider.
    """

    def __init__(
        self, *, provider: str, model: str, max_tokens: int, breaker: CircuitBreaker
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._breaker = breaker

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._model}"

    @property
    def can_generate(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._model

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    @observe(name="llm.generate_answer")
    async def generate_answer(
        self, query: str, context_docs: Sequence[str], *, caveat: bool = False
    ) -> str:
        return await self._guarded(
            system=RAG_SYSTEM,
            user=build_rag_prompt(query, context_docs, caveat=caveat),
            max_tokens=self._max_tokens,
        )

    @observe(name="llm.verify_answer")
    async def verify_answer(self, query: str, context_docs: Sequence[str], answer: str) -> bool:
        """Self-check; an unparseable verdict counts as unsupported (fail safe)."""
        text = await self._guarded(
            system=SELF_CHECK_SYSTEM,
            user=build_self_check_prompt(query, context_docs, answer),
            max_tokens=min(SELF_CHECK_MAX_TOKENS, self._max_tokens),
        )
        return parse_self_check(text) is True

    @observe(name="llm.classify_intent")
    async def classify_intent(self, query: str) -> str:
        text = await self._guarded(
            system=INTENT_SYSTEM, user=query, max_tokens=min(INTENT_MAX_TOKENS, self._max_tokens)
        )
        return text.strip()

    async def aclose(self) -> None:
        """Release HTTP connections (no-op unless the adapter holds a client)."""
        return None

    async def _guarded(self, *, system: str, user: str, max_tokens: int) -> str:
        if not self._breaker.allow_request():
            raise LLMUnavailableError(
                f"{self._provider} is temporarily unavailable (circuit breaker open)",
                provider=self._provider,
                reason="circuit_open",
            )
        try:
            text = await self._complete(system=system, user=user, max_tokens=max_tokens)
        except LLMUnavailableError:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return text

    @abstractmethod
    async def _complete(self, *, system: str, user: str, max_tokens: int) -> str:
        """Send one system + user turn; return the reply text.

        Raises:
            LLMUnavailableError: any provider failure (after the adapter's
                retries), refusal or malformed response. The message must not
                contain secrets.
        """


class NullGateway:
    """``LLM_MODE=client``: there is no LLM on this server.

    Every generation call raises :class:`AnswerGenerationUnavailableError`, so a
    caller that forgets to check :attr:`can_generate` gets a clear structured
    error instead of a silent empty answer.
    """

    def __init__(self, llm_mode: LLMMode = LLMMode.CLIENT) -> None:
        self._llm_mode = llm_mode

    @property
    def name(self) -> str:
        return "null"

    @property
    def can_generate(self) -> bool:
        return False

    async def generate_answer(
        self, query: str, context_docs: Sequence[str], *, caveat: bool = False
    ) -> str:
        raise AnswerGenerationUnavailableError(self._llm_mode.value)

    async def classify_intent(self, query: str) -> str:
        raise AnswerGenerationUnavailableError(self._llm_mode.value)

    async def verify_answer(self, query: str, context_docs: Sequence[str], answer: str) -> bool:
        raise AnswerGenerationUnavailableError(self._llm_mode.value)

    async def aclose(self) -> None:
        return None
