"""Load knowledge from files, folders and URLs into an :class:`~faq_core.models.IngestBatch`.

Supported sources (``python app.py --mode ingest --source PATH_OR_URL``):

==================  ==========================================================
``.md``/``.markdown``  document, heading-aware chunks; title = first ``#`` heading
``.txt``            plain-text document; title = first line
``.html``/``.htm``  converted to markdown (headings kept, scripts/nav dropped)
``.csv``            FAQ entries; header must name ``question`` and ``answer``
                    columns, ``category`` is optional (default ``General``)
``.json``           FAQ entries: a list of ``{question, answer, category}``
                    (the ``data/sample_faq.json`` format)
folder              every supported file below it (hidden files, symlinks and
                    other types are skipped and reported)
``http(s)://`` URL  by response ``Content-Type`` (HTML, markdown, text, CSV, JSON)
==================  ==========================================================

URL fetching is deliberately conservative: only ``http``/``https``, no
credentials in the URL, a per-request timeout, at most three redirects (each
re-validated), and a byte cap enforced while streaming. Query strings and
fragments are dropped from stored source URIs and from every log line and error
message, because they frequently carry access tokens.

Loading only parses; :meth:`faq_core.service.SupportService.ingest` stores and
indexes the batch.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import ValidationError

from faq_core.errors import IngestError
from faq_core.models import DocumentDraft, DocumentFormat, FAQDraft, IngestBatch, SkippedSource
from faq_core.retrieval.chunking import markdown_title
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

_log = get_logger(__name__)

DEFAULT_MAX_BYTES: Final = 5_000_000
DEFAULT_TIMEOUT: Final = 15.0
MAX_REDIRECTS: Final = 3
DEFAULT_CATEGORY: Final = "General"
_TITLE_MAX_CHARS: Final = 120
_USER_AGENT: Final = "customer-support-faq-ingest/0.2"
_ACCEPT: Final = (
    "text/markdown, text/html;q=0.9, text/plain;q=0.8, text/csv;q=0.8, application/json;q=0.7"
)


class _Kind(StrEnum):
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    CSV = "csv"
    JSON = "json"


_SUFFIX_KIND: Final[dict[str, _Kind]] = {
    ".md": _Kind.MARKDOWN,
    ".markdown": _Kind.MARKDOWN,
    ".txt": _Kind.TEXT,
    ".html": _Kind.HTML,
    ".htm": _Kind.HTML,
    ".csv": _Kind.CSV,
    ".json": _Kind.JSON,
}
_CONTENT_TYPE_KIND: Final[dict[str, _Kind]] = {
    "text/markdown": _Kind.MARKDOWN,
    "text/x-markdown": _Kind.MARKDOWN,
    "text/plain": _Kind.TEXT,
    "text/html": _Kind.HTML,
    "application/xhtml+xml": _Kind.HTML,
    "text/csv": _Kind.CSV,
    "application/json": _Kind.JSON,
}


# ── URL fetching ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FetchedResource:
    """A downloaded resource (body already decoded to text)."""

    url: str
    content_type: str
    text: str
    last_modified: datetime | None = None


class UrlFetcher(Protocol):
    """Downloads one URL; raises :class:`~faq_core.errors.IngestError` on failure."""

    async def fetch(self, url: str) -> FetchedResource: ...


def safe_url(url: str) -> str:
    """Return *url* without credentials, query string or fragment (for storage and logs)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path or "/", "", ""))


def validate_url(url: str) -> str:
    """Check that *url* is an absolute http(s) URL without embedded credentials.

    Raises:
        IngestError: any other scheme, no host, or ``user:password@`` in the URL.
    """
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError as err:
        raise IngestError("malformed URL") from err
    if parts.scheme not in ("http", "https"):
        raise IngestError("only http(s) URLs can be ingested", source=safe_url(url))
    if not parts.hostname:
        raise IngestError("URL has no host")
    if parts.username is not None or parts.password is not None:
        raise IngestError("URLs with embedded credentials are not accepted", source=safe_url(url))
    return url


class HttpxUrlFetcher:
    """:class:`UrlFetcher` over ``httpx`` with timeout, redirect and size limits."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        transport: httpx.AsyncBaseTransport | None = None,
        url_policy: Callable[[str], None] | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._transport = transport
        # Extra check on the first URL and every redirect target (raises
        # IngestError to refuse) — e.g. the offline-mode network guard.
        self._url_policy = url_policy

    def _checked(self, url: str) -> str:
        url = validate_url(url)
        if self._url_policy is not None:
            self._url_policy(url)
        return url

    async def fetch(self, url: str) -> FetchedResource:
        import httpx

        current = self._checked(url)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False, transport=self._transport
            ) as client:
                for _ in range(self._max_redirects + 1):
                    async with client.stream(
                        "GET", current, headers={"Accept": _ACCEPT, "User-Agent": _USER_AGENT}
                    ) as response:
                        if response.is_redirect:
                            location = response.headers.get("location", "")
                            current = self._checked(urljoin(current, location))
                            continue
                        return await self._read(current, response)
        except httpx.TimeoutException as err:
            raise IngestError("URL fetch timed out", source=safe_url(url)) from err
        except httpx.HTTPError as err:
            raise IngestError(
                f"URL fetch failed: {type(err).__name__}", source=safe_url(url)
            ) from err
        raise IngestError("too many redirects", source=safe_url(url))

    async def _read(self, url: str, response: httpx.Response) -> FetchedResource:
        if response.status_code >= 400:
            raise IngestError(
                f"URL fetch failed: HTTP {response.status_code}", source=safe_url(url)
            )
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            raise IngestError("resource exceeds the size limit", source=safe_url(url))
        body = bytearray()
        async for piece in response.aiter_bytes():
            body.extend(piece)
            if len(body) > self._max_bytes:
                raise IngestError("resource exceeds the size limit", source=safe_url(url))
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        encoding = response.charset_encoding or "utf-8"
        try:
            text = body.decode(encoding, errors="replace")
        except LookupError:  # unknown charset label
            text = body.decode("utf-8", errors="replace")
        return FetchedResource(
            url=url,
            content_type=content_type,
            text=text,
            last_modified=_parse_http_date(response.headers.get("last-modified")),
        )


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# ── source loading ───────────────────────────────────────────────────────────


class SourceLoader:
    """Turns a path or URL into an :class:`IngestBatch` (no storage, no indexing)."""

    def __init__(self, *, fetcher: UrlFetcher, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._fetcher = fetcher
        self._max_bytes = max_bytes

    async def load(self, source: str) -> IngestBatch:
        """Load *source*: an http(s) URL, a supported file, or a folder.

        Raises:
            IngestError: missing path, unsupported file/content type, bad URL,
                failed fetch, or unparseable content.
        """
        if "://" in source:
            return await self._load_url(source)
        path, is_dir = await asyncio.to_thread(_locate, source)
        if path is None:
            raise IngestError("ingest source not found", source=source)
        if is_dir:
            return await self._load_folder(path)
        return await self._load_file(path, display=source)

    async def _load_url(self, url: str) -> IngestBatch:
        resource = await self._fetcher.fetch(url)
        stored_uri = safe_url(resource.url)
        kind = _CONTENT_TYPE_KIND.get(resource.content_type)
        if kind == _Kind.TEXT and urlsplit(resource.url).path.lower().endswith(".md"):
            kind = _Kind.MARKDOWN  # many servers send markdown as text/plain
        if kind is None:
            raise IngestError(
                "unsupported content type",
                source=stored_uri,
                content_type=resource.content_type or "unknown",
            )
        fallback_title = Path(urlsplit(resource.url).path).stem or (
            urlsplit(resource.url).hostname or "document"
        )
        _log.info("fetched ingest URL", extra={"url": stored_uri, "kind": kind})
        return _parse(
            kind,
            resource.text,
            source=stored_uri,
            fallback_title=fallback_title,
            updated_at=resource.last_modified or datetime.now(UTC),
        )

    async def _load_file(self, path: Path, *, display: str) -> IngestBatch:
        kind = _SUFFIX_KIND.get(path.suffix.lower())
        if kind is None:
            raise IngestError("unsupported file type", source=display)
        size, mtime, uri = await asyncio.to_thread(_file_info, path)
        if size > self._max_bytes:
            raise IngestError("file exceeds the size limit", source=display)
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except OSError as err:
            raise IngestError(f"cannot read file: {type(err).__name__}", source=display) from err
        return _parse(
            kind,
            text,
            source=uri,
            fallback_title=path.stem.replace("_", " ").replace("-", " "),
            updated_at=mtime,
        )

    async def _load_folder(self, folder: Path) -> IngestBatch:
        files, skipped = await asyncio.to_thread(_scan_folder, folder)
        faqs: list[FAQDraft] = []
        documents: list[DocumentDraft] = []
        for path in files:
            display = str(path.relative_to(folder))
            try:
                batch = await self._load_file(path, display=display)
            except IngestError as err:
                skipped.append(SkippedSource(source=display, reason=err.message))
                continue
            faqs.extend(batch.faqs)
            documents.extend(batch.documents)
            skipped.extend(batch.skipped)
        return IngestBatch(source=folder.as_uri(), faqs=faqs, documents=documents, skipped=skipped)


def _locate(source: str) -> tuple[Path | None, bool]:
    """Resolve *source* to an absolute path; ``(None, False)`` if it does not exist."""
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return path, True
    if path.is_file():
        return path, False
    return None, False


def _file_info(path: Path) -> tuple[int, datetime, str]:
    stat = path.stat()
    return stat.st_size, datetime.fromtimestamp(stat.st_mtime, UTC), path.resolve().as_uri()


def _scan_folder(folder: Path) -> tuple[list[Path], list[SkippedSource]]:
    files: list[Path] = []
    skipped: list[SkippedSource] = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            skipped.append(SkippedSource(source=str(relative), reason="symlinks are not followed"))
        elif path.is_file():
            if path.suffix.lower() in _SUFFIX_KIND:
                files.append(path)
            else:
                skipped.append(SkippedSource(source=str(relative), reason="unsupported file type"))
    return files, skipped


# ── parsers ──────────────────────────────────────────────────────────────────


def _parse(
    kind: _Kind, text: str, *, source: str, fallback_title: str, updated_at: datetime
) -> IngestBatch:
    if kind == _Kind.CSV:
        faqs, skipped = parse_faq_csv(text, source=source)
        return IngestBatch(source=source, faqs=faqs, skipped=skipped)
    if kind == _Kind.JSON:
        faqs, skipped = parse_faq_json(text, source=source)
        return IngestBatch(source=source, faqs=faqs, skipped=skipped)
    if kind == _Kind.HTML:
        page_title, text = html_to_markdown(text)
        title = page_title or markdown_title(text) or fallback_title
        fmt = DocumentFormat.MARKDOWN
    elif kind == _Kind.MARKDOWN:
        title = markdown_title(text) or fallback_title
        fmt = DocumentFormat.MARKDOWN
    else:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        title = first_line[:_TITLE_MAX_CHARS] or fallback_title
        fmt = DocumentFormat.TEXT
    if not text.strip():
        return IngestBatch(
            source=source, skipped=[SkippedSource(source=source, reason="empty document")]
        )
    draft = DocumentDraft(
        title=(title or "Untitled")[:_TITLE_MAX_CHARS],
        body=text,
        format=fmt,
        source_uri=source,
        updated_at=updated_at,
    )
    return IngestBatch(source=source, documents=[draft])


def parse_faq_csv(text: str, *, source: str) -> tuple[list[FAQDraft], list[SkippedSource]]:
    """Parse FAQ rows from CSV with ``question``/``answer`` (+ ``category``) headers.

    Raises:
        IngestError: no header row, or it lacks ``question`` / ``answer``.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
    if "question" not in header or "answer" not in header:
        raise IngestError("CSV needs 'question' and 'answer' columns", source=source)
    faqs: list[FAQDraft] = []
    skipped: list[SkippedSource] = []
    for line_number, row in enumerate(reader, start=2):
        question = (row.get(header["question"]) or "").strip()
        answer = (row.get(header["answer"]) or "").strip()
        category = (row.get(header.get("category", ""), "") or "").strip() or DEFAULT_CATEGORY
        if not question and not answer:
            continue  # blank line
        if not question or not answer:
            skipped.append(
                SkippedSource(
                    source=f"{source}#L{line_number}", reason="missing question or answer"
                )
            )
            continue
        faqs.append(FAQDraft(question=question, answer=answer, category=category))
    return faqs, skipped


def parse_faq_json(text: str, *, source: str) -> tuple[list[FAQDraft], list[SkippedSource]]:
    """Parse a JSON list of ``{question, answer, category}`` objects.

    Raises:
        IngestError: invalid JSON, or the top level is not a list.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as err:
        raise IngestError("invalid JSON", source=source) from err
    if not isinstance(raw, list):
        raise IngestError("JSON must be a list of FAQ objects", source=source)
    faqs: list[FAQDraft] = []
    skipped: list[SkippedSource] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            item = {"category": DEFAULT_CATEGORY, **item}
        try:
            faqs.append(FAQDraft.model_validate(item))
        except ValidationError:
            skipped.append(SkippedSource(source=f"{source}#{index}", reason="invalid FAQ object"))
    return faqs, skipped


# ── HTML → markdown ──────────────────────────────────────────────────────────

_SKIP_TAGS: Final = frozenset({"script", "style", "noscript", "template", "svg", "nav", "footer"})
_BLOCK_TAGS: Final = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "aside",
        "ul",
        "ol",
        "table",
        "tr",
        "blockquote",
        "pre",
        "dl",
        "dt",
        "dd",
        "figure",
        "form",
        "hr",
        "br",
    }
)
_HEADING_TAGS: Final = {f"h{level}": level for level in range(1, 7)}
_BLANK_RUNS: Final = re.compile(r"\n{3,}")


class _MarkdownExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _HEADING_TAGS:
            self.parts.append("\n\n" + "#" * _HEADING_TAGS[tag] + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
            self._in_pre = self._in_pre or tag == "pre"

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
        elif tag == "title":
            self._in_title = False
        elif tag in _HEADING_TAGS or tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
            if tag == "pre":
                self._in_pre = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))


def html_to_markdown(html: str) -> tuple[str, str]:
    """Convert an HTML page to ``(title, markdown_text)`` with headings preserved."""
    extractor = _MarkdownExtractor()
    extractor.feed(html)
    extractor.close()
    lines = [line.strip() for line in "".join(extractor.parts).splitlines()]
    text = _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()
    return " ".join(extractor.title.split()), text
