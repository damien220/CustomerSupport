"""Heading-aware chunking of markdown documents, plain text and FAQ answers.

A markdown document is split at ATX (``## Title``) and setext (``Title`` over
``===`` / ``---``) headings, and every chunk records its heading path
(``"Billing > Refunds"``) so a citation can point at the section. A section
longer than ``max_chars`` is packed paragraph by paragraph, then sentence by
sentence, then word by word. Lines inside fenced code blocks are never treated
as headings, and a leading YAML front-matter block is dropped. Sections with a
heading but no body produce no chunk (their heading still prefixes the
sub-sections below them).

Bump :data:`CHUNKER_VERSION` whenever the output for the same input changes: it
is part of every chunk's ``source_hash``, so the startup index sync re-chunks
the whole knowledge base after an upgrade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

CHUNKER_VERSION: Final = "heading-v1"
DEFAULT_MAX_CHARS: Final = 1200
MIN_MAX_CHARS: Final = 100
HEADING_SEPARATOR: Final = " > "

_ATX_HEADING: Final = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT_UNDERLINE: Final = re.compile(r"^ {0,3}(=+|-{2,})[ \t]*$")
_FENCE_OPEN: Final = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+")
_FRONT_MATTER_MAX_LINES: Final = 60


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One chunk of a source: its section heading path and its text."""

    heading: str
    content: str


@dataclass(slots=True)
class _Section:
    heading: str
    lines: list[str]


def chunk_markdown(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[TextChunk]:
    """Split markdown *text* into heading-scoped chunks of at most *max_chars*."""
    _check_max_chars(max_chars)
    chunks: list[TextChunk] = []
    for section in _sections(_strip_front_matter(text.splitlines())):
        chunks.extend(
            TextChunk(heading=section.heading, content=piece)
            for piece in _pack(_paragraphs(section.lines), max_chars)
        )
    return chunks


def chunk_plain_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[TextChunk]:
    """Split unstructured *text* on blank lines into chunks of at most *max_chars*."""
    _check_max_chars(max_chars)
    return [
        TextChunk(heading="", content=piece)
        for piece in _pack(_paragraphs(text.splitlines()), max_chars)
    ]


def markdown_title(text: str) -> str | None:
    """Return the first level-1 heading of markdown *text*, if it has one."""
    lines = _strip_front_matter(text.splitlines())
    fence: str | None = None
    for index, line in enumerate(lines):
        fence, in_code = _track_fence(fence, line)
        if in_code:
            continue
        atx = _ATX_HEADING.match(line)
        if atx and len(atx.group(1)) == 1 and (atx.group(2) or "").strip():
            return (atx.group(2) or "").strip()
        if _is_setext(lines, index) and lines[index + 1].strip().startswith("="):
            return line.strip()
    return None


# ── parsing ──────────────────────────────────────────────────────────────────


def _check_max_chars(max_chars: int) -> None:
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}, got {max_chars}")


def _strip_front_matter(lines: list[str]) -> list[str]:
    if not lines or lines[0].strip() != "---":
        return lines
    for index in range(1, min(len(lines), _FRONT_MATTER_MAX_LINES)):
        if lines[index].strip() in ("---", "..."):
            return lines[index + 1 :]
    return lines


def _track_fence(fence: str | None, line: str) -> tuple[str | None, bool]:
    """Return the fence state after *line* and whether *line* is code (or a fence)."""
    if fence is not None:
        closed = line.strip().startswith(fence)
        return (None if closed else fence), True
    opened = _FENCE_OPEN.match(line)
    if opened:
        return opened.group(1), True
    return None, False


def _is_setext(lines: list[str], index: int) -> bool:
    """A one-line paragraph directly followed by an ``===``/``---`` underline."""
    if index + 1 >= len(lines) or not lines[index].strip():
        return False
    if _ATX_HEADING.match(lines[index]) or _SETEXT_UNDERLINE.match(lines[index]):
        return False
    starts_paragraph = index == 0 or not lines[index - 1].strip()
    return starts_paragraph and _SETEXT_UNDERLINE.match(lines[index + 1]) is not None


def _sections(lines: list[str]) -> list[_Section]:
    stack: list[tuple[int, str]] = []
    sections = [_Section(heading="", lines=[])]
    fence: str | None = None
    skip_next = False

    def open_section(level: int, title: str) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(
            _Section(heading=HEADING_SEPARATOR.join(t for _, t in stack if t), lines=[])
        )

    for index, line in enumerate(lines):
        if skip_next:  # the setext underline of the previous line
            skip_next = False
            continue
        fence, in_code = _track_fence(fence, line)
        if not in_code:
            atx = _ATX_HEADING.match(line)
            if atx:
                open_section(len(atx.group(1)), (atx.group(2) or "").strip())
                continue
            if _is_setext(lines, index):
                level = 1 if lines[index + 1].strip().startswith("=") else 2
                open_section(level, line.strip())
                skip_next = True
                continue
        sections[-1].lines.append(line)
    return sections


def _paragraphs(lines: list[str]) -> list[str]:
    """Group *lines* into blank-line separated paragraphs, keeping code fences whole."""
    paragraphs: list[str] = []
    current: list[str] = []
    fence: str | None = None
    for line in lines:
        fence, in_code = _track_fence(fence, line)
        if not in_code and not line.strip():
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line.rstrip())
    if current:
        paragraphs.append("\n".join(current).strip())
    return [p for p in paragraphs if p]


# ── packing ──────────────────────────────────────────────────────────────────


def _pack(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedily join paragraphs (split further when needed) up to *max_chars*."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        for piece in _split_long(paragraph, max_chars):
            added = len(piece) + (2 if current else 0)
            if current and size + added > max_chars:
                chunks.append("\n\n".join(current))
                current, size, added = [], 0, len(piece)
            current.append(piece)
            size += added
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        for part in _hard_wrap(sentence, max_chars):
            if current and len(current) + 1 + len(part) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces
