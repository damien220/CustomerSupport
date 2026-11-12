"""Plain-text answer formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faq_core.models import SourceType

if TYPE_CHECKING:
    from faq_core.models import SearchResult


class PlainFormatter:
    """Render an answer as plain text with a simple source list."""

    name = "plain"

    def format(self, result: SearchResult) -> str:
        parts = [result.answer]
        if result.sources:
            parts.append("")
            parts.append("Sources:")
            parts.extend(f"- {source.question} ({source.category})" for source in result.sources)
        documents = [c for c in result.citations if c.source_type is SourceType.DOCUMENT]
        if documents:
            parts.append("")
            parts.append("Documents:")
            parts.extend(
                f"- {' > '.join(p for p in (c.title, c.heading) if p)} "
                f"(updated {c.updated_at.date().isoformat()})"
                for c in documents
            )
        return "\n".join(parts)
