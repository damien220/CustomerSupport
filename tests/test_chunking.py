"""Heading-aware chunker: section paths, setext headings, code fences, size limits."""

from __future__ import annotations

import pytest

from faq_core.retrieval.chunking import (
    MIN_MAX_CHARS,
    TextChunk,
    chunk_markdown,
    chunk_plain_text,
    markdown_title,
)

_GUIDE = """---
title: ignored front matter
---
# Billing guide

Intro paragraph about billing.

## Refunds

Refunds take 5 days.

### Card refunds

Card refunds go back to the card.

## Invoices
"""


def test_markdown_chunks_carry_their_heading_path() -> None:
    chunks = chunk_markdown(_GUIDE)

    assert chunks == [
        TextChunk(heading="Billing guide", content="Intro paragraph about billing."),
        TextChunk(heading="Billing guide > Refunds", content="Refunds take 5 days."),
        TextChunk(
            heading="Billing guide > Refunds > Card refunds",
            content="Card refunds go back to the card.",
        ),
    ]  # "Invoices" has no body, so no chunk; front matter is dropped


def test_text_before_the_first_heading_has_an_empty_heading() -> None:
    chunks = chunk_markdown("Preamble text.\n\n# Title\n\nBody.")

    assert chunks[0] == TextChunk(heading="", content="Preamble text.")
    assert chunks[1].heading == "Title"


def test_setext_headings_are_recognised() -> None:
    text = "Account\n=======\n\nAccount basics.\n\nSecurity\n--------\n\nUse 2FA.\n"

    chunks = chunk_markdown(text)

    assert [c.heading for c in chunks] == ["Account", "Account > Security"]
    assert markdown_title(text) == "Account"


def test_headings_inside_code_fences_are_not_split_on() -> None:
    text = "# Setup\n\n```bash\n# not a heading\necho hi\n```\n\nDone."

    [chunk] = chunk_markdown(text)

    assert chunk.heading == "Setup"
    assert "# not a heading" in chunk.content
    assert chunk.content.endswith("Done.")


def test_sibling_heading_replaces_deeper_levels() -> None:
    chunks = chunk_markdown("# A\n\n## B\n\nb\n\n### C\n\nc\n\n## D\n\nd\n")

    assert [c.heading for c in chunks] == ["A > B", "A > B > C", "A > D"]


def test_long_sections_are_packed_within_max_chars() -> None:
    sentences = " ".join(f"Sentence number {i} explains a detail." for i in range(60))
    text = f"# Long\n\n{sentences}\n\nShort tail paragraph."

    chunks = chunk_markdown(text, max_chars=200)

    assert len(chunks) > 5
    assert all(len(c.content) <= 200 for c in chunks)
    assert all(c.heading == "Long" for c in chunks)
    joined = " ".join(c.content for c in chunks)
    assert "Sentence number 59" in joined and "Short tail paragraph." in joined


def test_a_single_huge_word_is_hard_wrapped() -> None:
    [*chunks] = chunk_plain_text("x" * 450, max_chars=MIN_MAX_CHARS)

    assert [len(c.content) for c in chunks] == [100, 100, 100, 100, 50]


def test_plain_text_splits_on_blank_lines_without_headings() -> None:
    chunks = chunk_plain_text("# not markdown here\nline\n\nsecond", max_chars=MIN_MAX_CHARS)

    assert chunks == [TextChunk(heading="", content="# not markdown here\nline\n\nsecond")]


def test_markdown_title_is_none_without_a_level_one_heading() -> None:
    assert markdown_title("## Only a subheading\n\ntext") is None
    assert markdown_title("```\n# in code\n```") is None


def test_max_chars_below_minimum_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_markdown("text", max_chars=MIN_MAX_CHARS - 1)
