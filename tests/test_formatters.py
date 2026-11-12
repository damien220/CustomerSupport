from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from faq_core.models import Citation, FAQEntry, SearchResult, SourceType
from formatters import (
    Formatter,
    FormatterRegistry,
    MarkdownFormatter,
    PlainFormatter,
    UnknownFormatterError,
    default_registry,
)


@pytest.fixture
def answer_with_sources() -> SearchResult:
    return SearchResult(
        query="How do I reset my password?",
        answer="Go to Settings > Security.",
        sources=[FAQEntry(question="Reset password?", answer="Settings.", category="account")],
        confidence=0.9,
    )


@pytest.fixture
def answer_without_sources() -> SearchResult:
    return SearchResult(query="Hi", answer="Hello.", sources=[], confidence=0.0)


def test_markdown_formatter_renders_answer_and_sources(answer_with_sources: SearchResult) -> None:
    output = MarkdownFormatter().format(answer_with_sources)

    assert "**Answer:** Go to Settings > Security." in output
    assert "**Sources:**" in output
    assert "- Reset password? _(account)_" in output


def test_markdown_formatter_omits_sources_section_when_empty(
    answer_without_sources: SearchResult,
) -> None:
    output = MarkdownFormatter().format(answer_without_sources)

    assert output == "**Answer:** Hello."


def test_plain_formatter_renders_answer_and_sources(answer_with_sources: SearchResult) -> None:
    output = PlainFormatter().format(answer_with_sources)

    assert output.startswith("Go to Settings > Security.")
    assert "Sources:" in output
    assert "- Reset password? (account)" in output


def test_formatters_declare_names() -> None:
    assert MarkdownFormatter.name == "markdown"
    assert PlainFormatter.name == "plain"


def test_default_registry_resolves_builtin_formatters() -> None:
    registry = default_registry()

    assert registry.names() == ["markdown", "plain"]
    assert isinstance(registry.get("markdown"), MarkdownFormatter)


def test_registry_unknown_name_raises_with_known_names() -> None:
    with pytest.raises(UnknownFormatterError, match="markdown, plain"):
        default_registry().get("html")


def test_registry_refuses_silent_shadowing_but_allows_explicit_replace() -> None:
    class ShoutingFormatter:
        name = "plain"

        def format(self, result: SearchResult) -> str:
            return result.answer.upper()

    registry: FormatterRegistry = default_registry()
    custom: Formatter = ShoutingFormatter()

    with pytest.raises(ValueError, match="already registered"):
        registry.register(custom)
    registry.register(custom, replace=True)

    assert registry.get("plain") is custom


def test_formatters_list_document_citations_with_freshness() -> None:
    citation = Citation(
        id=uuid.uuid4(),
        source_type=SourceType.DOCUMENT,
        source_id=uuid.uuid4(),
        title="Billing guide",
        heading="Refunds",
        snippet="Five days.",
        updated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    result = SearchResult(query="q", answer="A", confidence=0.5, sources=[], citations=[citation])

    assert (
        MarkdownFormatter()
        .format(result)
        .endswith("**Documents:**\n- Billing guide › Refunds _(updated 2026-09-01)_")
    )
    assert (
        PlainFormatter()
        .format(result)
        .endswith("Documents:\n- Billing guide > Refunds (updated 2026-09-01)")
    )
