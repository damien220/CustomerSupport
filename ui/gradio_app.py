"""Gradio dev chat UI — a thin adapter over :class:`SupportService`.

``chat_fn`` yields the complete formatted answer once per question
(whole-answer relay, no token streaming).

This module imports ``gradio``, which is part of the heavy dependency set
(``requirements-rag.txt``). It is not imported in lite runs; ``app.run_ui``
imports it lazily.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import gradio as gr

from faq_core.errors import SupportError
from formatters import default_registry

if TYPE_CHECKING:
    from faq_core.service import SupportService

_EXAMPLES = [
    "How do I reset my password?",
    "What are your business hours?",
    "How do I return a product?",
]


def build_demo(service: SupportService, *, formatter_name: str = "markdown") -> Any:
    """Build the ``gr.ChatInterface`` wired to *service*."""
    formatter = default_registry().get(formatter_name)

    async def chat_fn(message: str, _history: list[dict[str, str]]) -> AsyncIterator[str]:
        try:
            result = await service.search(message)
        except SupportError as err:
            yield f"Sorry, I couldn't process that: {err.message}"
            return
        yield formatter.format(result)

    return gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="Customer Support FAQ Assistant",
        examples=_EXAMPLES,
    )
