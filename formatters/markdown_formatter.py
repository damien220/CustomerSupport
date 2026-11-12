"""Markdown answer formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faq_core.models import SourceType

if TYPE_CHECKING:
    from faq_core.models import SearchResult


class MarkdownFormatter:
    """Render an answer as Markdown with a bulleted source list.

    FAQ sources are listed by question and category; cited passages from
    ingested documents follow with their section and freshness date.
    """

    name = "markdown"

    def format(self, result: SearchResult) -> str:
        lines = [f"**Answer:** {result.answer}"]
        if result.sources:
            lines.append("")
            lines.append("**Sources:**")
            lines.extend(f"- {source.question} _({source.category})_" for source in result.sources)
        documents = [c for c in result.citations if c.source_type is SourceType.DOCUMENT]
        if documents:
            lines.append("")
            lines.append("**Documents:**")
            lines.extend(
                f"- {' › '.join(p for p in (c.title, c.heading) if p)} "
                f"_(updated {c.updated_at.date().isoformat()})_"
                for c in documents
            )
        return "\n".join(lines)
