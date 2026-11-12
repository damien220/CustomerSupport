"""Ingestion: markdown / text / HTML / CSV / JSON files, folders, URLs, and storing a batch.

URLs are fetched from a real local ``http.server`` on 127.0.0.1 (no external
network); storage and indexing use the real SQLite repository and retrieval stack.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bootstrap import assemble_runtime, ingest_sources
from faq_core.config import AppConfig
from faq_core.errors import IngestError
from faq_core.events import EventBus
from faq_core.models import DocumentFormat
from faq_core.repository import FAQRepository
from faq_core.retrieval.engine import RetrievalAnswerEngine, RetrievalStack
from faq_core.retrieval.ingest import (
    HttpxUrlFetcher,
    SourceLoader,
    html_to_markdown,
    safe_url,
    validate_url,
)
from faq_core.service import SupportService
from tests.conftest import FakeLLMClient

_PAGES: dict[str, tuple[int, str, bytes, dict[str, str]]] = {
    "/guide.html": (
        200,
        "text/html; charset=utf-8",
        b"<html><head><title>Help Center</title><script>var x = 1;</script></head><body>"
        b"<nav>Home | Login</nav><h1>Returns</h1><p>Return items within <b>30 days</b>.</p>"
        b"<h2>Refunds</h2><ul><li>Card refunds: 5 days</li><li>PayPal: 2 days</li></ul>"
        b"<footer>Copyright</footer></body></html>",
        {"Last-Modified": "Tue, 01 Sep 2026 10:00:00 GMT"},
    ),
    "/notes.md": (200, "text/plain", b"# Notes\n\nPlain markdown served as text.", {}),
    "/faq.csv": (200, "text/csv", b"question,answer\nQ1?,A1.\n", {}),
    "/moved": (302, "text/plain", b"", {"Location": "/notes.md?token=abc"}),
    "/loop": (302, "text/plain", b"", {"Location": "/loop"}),
    "/big.md": (200, "text/markdown", b"# Big\n\n" + b"x" * 5000, {}),
    "/image.png": (200, "image/png", b"\x89PNG", {}),
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status, content_type, body, headers = _PAGES.get(
            self.path.split("?")[0], (404, "text/plain", b"missing", {})
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — stdlib signature
        return


@pytest.fixture(scope="module")
def http_base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def loader() -> SourceLoader:
    return SourceLoader(fetcher=HttpxUrlFetcher(timeout=5, max_bytes=4096), max_bytes=4096)


# ── files ────────────────────────────────────────────────────────────────────


async def test_markdown_file_becomes_a_document(tmp_path: Path, loader: SourceLoader) -> None:
    path = tmp_path / "billing_guide.md"
    path.write_text("# Billing\n\n## Refunds\n\nFive days.", encoding="utf-8")
    os.utime(path, (1_790_000_000, 1_790_000_000))

    batch = await loader.load(str(path))

    [document] = batch.documents
    assert (document.title, document.format) == ("Billing", DocumentFormat.MARKDOWN)
    assert document.source_uri == path.resolve().as_uri()
    assert document.updated_at.timestamp() == 1_790_000_000
    assert batch.faqs == []


async def test_text_and_html_files(tmp_path: Path, loader: SourceLoader) -> None:
    (tmp_path / "policy.txt").write_text("Shipping policy\n\nWe ship weekly.", encoding="utf-8")
    (tmp_path / "page.html").write_text("<h1>Hours</h1><p>9 to 5</p>", encoding="utf-8")

    text_doc = (await loader.load(str(tmp_path / "policy.txt"))).documents[0]
    html_doc = (await loader.load(str(tmp_path / "page.html"))).documents[0]

    assert (text_doc.title, text_doc.format) == ("Shipping policy", DocumentFormat.TEXT)
    assert html_doc.title == "Hours"
    assert html_doc.body == "# Hours\n\n9 to 5"


async def test_csv_rows_become_faqs_and_bad_rows_are_reported(
    tmp_path: Path, loader: SourceLoader
) -> None:
    path = tmp_path / "faq.csv"
    path.write_text(
        "﻿Question,Answer,Category\n"
        "How do I pay?,By card.,Billing\n"
        "No category?,Defaults apply.,\n"
        ",,\n"
        "Missing answer?,,X\n",
        encoding="utf-8",
    )

    batch = await loader.load(str(path))

    assert [(f.question, f.category) for f in batch.faqs] == [
        ("How do I pay?", "Billing"),
        ("No category?", "General"),
    ]
    assert [s.reason for s in batch.skipped] == ["missing question or answer"]
    assert batch.skipped[0].source.endswith("#L5")


async def test_csv_without_required_columns_is_rejected(
    tmp_path: Path, loader: SourceLoader
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("title,body\nA,B\n", encoding="utf-8")

    with pytest.raises(IngestError, match="'question' and 'answer'"):
        await loader.load(str(path))


async def test_json_faq_list(tmp_path: Path, loader: SourceLoader) -> None:
    path = tmp_path / "faq.json"
    path.write_text('[{"question": "Q?", "answer": "A."}, {"question": ""}, 3]', encoding="utf-8")

    batch = await loader.load(str(path))

    assert [(f.question, f.category) for f in batch.faqs] == [("Q?", "General")]
    assert len(batch.skipped) == 2


async def test_folder_ingests_supported_files_and_reports_the_rest(
    tmp_path: Path, loader: SourceLoader
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nAlpha.", encoding="utf-8")
    (tmp_path / "faq.csv").write_text("question,answer\nQ?,A.\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "huge.md").write_text("x" * 5000, encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# Hidden", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(tmp_path / "docs" / "a.md")

    batch = await loader.load(str(tmp_path))

    assert [d.title for d in batch.documents] == ["A"]
    assert [f.question for f in batch.faqs] == ["Q?"]
    assert {(s.source, s.reason) for s in batch.skipped} == {
        ("image.png", "unsupported file type"),
        ("huge.md", "file exceeds the size limit"),
        ("link.md", "symlinks are not followed"),
    }


async def test_missing_path_and_unsupported_file(tmp_path: Path, loader: SourceLoader) -> None:
    (tmp_path / "x.exe").write_bytes(b"MZ")

    with pytest.raises(IngestError, match="not found"):
        await loader.load(str(tmp_path / "nope.md"))
    with pytest.raises(IngestError, match="unsupported file type"):
        await loader.load(str(tmp_path / "x.exe"))


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


async def test_bundled_data_folder_loads_the_sample_faq() -> None:
    data = _DATA_DIR
    loader = SourceLoader(fetcher=HttpxUrlFetcher())

    batch = await loader.load(str(data))

    assert len(batch.faqs) == 10  # data/sample_faq.json
    reasons = {s.source: s.reason for s in batch.skipped}
    assert reasons["retrieval_benchmark.json"] == "JSON must be a list of FAQ objects"


# ── URLs ─────────────────────────────────────────────────────────────────────


async def test_html_url_is_converted_with_headings_and_freshness(
    http_base: str, loader: SourceLoader
) -> None:
    batch = await loader.load(f"{http_base}/guide.html?session=secret#top")

    [document] = batch.documents
    assert document.title == "Help Center"
    assert document.body.startswith("# Returns\n\nReturn items within 30 days.\n\n## Refunds")
    assert "- Card refunds: 5 days" in document.body
    assert "var x" not in document.body and "Home | Login" not in document.body
    assert document.source_uri == f"{http_base}/guide.html"  # query + fragment dropped
    assert document.updated_at.isoformat() == "2026-09-01T10:00:00+00:00"


async def test_redirects_are_followed_and_markdown_detected_by_path(
    http_base: str, loader: SourceLoader
) -> None:
    batch = await loader.load(f"{http_base}/moved")

    [document] = batch.documents
    assert (document.title, document.format) == ("Notes", DocumentFormat.MARKDOWN)
    assert document.source_uri == f"{http_base}/notes.md"


async def test_csv_url_becomes_faqs(http_base: str, loader: SourceLoader) -> None:
    assert [f.question for f in (await loader.load(f"{http_base}/faq.csv")).faqs] == ["Q1?"]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/big.md", "size limit"),
        ("/missing", "HTTP 404"),
        ("/image.png", "unsupported content type"),
        ("/loop", "too many redirects"),
    ],
)
async def test_url_failures_raise_ingest_errors(
    http_base: str, loader: SourceLoader, path: str, message: str
) -> None:
    with pytest.raises(IngestError, match=message):
        await loader.load(f"{http_base}{path}?api_key=hunter22")


async def test_url_errors_never_echo_query_strings(http_base: str, loader: SourceLoader) -> None:
    with pytest.raises(IngestError) as excinfo:
        await loader.load(f"{http_base}/missing?api_key=hunter22")

    assert "hunter22" not in str(excinfo.value.to_dict())


async def test_unreachable_host_is_an_ingest_error(loader: SourceLoader) -> None:
    with pytest.raises(IngestError, match="URL fetch failed"):
        await loader.load("http://127.0.0.1:9/nothing-listens-here")


@pytest.mark.parametrize(
    "url", ["ftp://example.com/a.md", "file:///etc/passwd", "http://user:pw@example.com/a"]
)
def test_only_plain_http_urls_are_accepted(url: str) -> None:
    with pytest.raises(IngestError):
        validate_url(url)


def test_safe_url_strips_credentials_query_and_fragment() -> None:
    assert (
        safe_url("https://u:p@Example.com:8443/a/b?token=x#frag") == "https://example.com:8443/a/b"
    )


def test_html_to_markdown_keeps_preformatted_text() -> None:
    title, text = html_to_markdown("<title> T </title><pre>a  b</pre><h3>H</h3>")

    assert title == "T"
    assert text == "a  b\n\n### H"


# ── storing a batch ──────────────────────────────────────────────────────────


@pytest.fixture
def ingest_service(sqlite_repo: FAQRepository, retrieval_stack: RetrievalStack) -> SupportService:
    engine = RetrievalAnswerEngine(
        retriever=retrieval_stack.retriever, indexer=retrieval_stack.indexer, llm=FakeLLMClient()
    )
    return SupportService(engine=engine, store=sqlite_repo, events=EventBus())


async def test_service_ingest_stores_indexes_and_dedupes(
    tmp_path: Path,
    loader: SourceLoader,
    ingest_service: SupportService,
    sqlite_repo: FAQRepository,
) -> None:
    (tmp_path / "faq.csv").write_text(
        "question,answer\nHow do I pay?,By card.\nhow do i PAY,Duplicate.\n", encoding="utf-8"
    )
    (tmp_path / "guide.md").write_text("# Guide\n\n## Warranty\n\nTwo years.", encoding="utf-8")

    first = await ingest_service.ingest(await loader.load(str(tmp_path)))
    again = await ingest_service.ingest(await loader.load(str(tmp_path)))
    result = await ingest_service.search("how long is the warranty", top_k=1)

    assert (first.faqs_added, first.faqs_duplicate, first.documents_added) == (1, 1, 1)
    assert (again.faqs_added, again.faqs_duplicate, again.documents_unchanged) == (0, 2, 1)
    assert await sqlite_repo.count() == 1
    assert result.citations[0].heading == "Guide > Warranty"


async def test_reingesting_a_changed_document_replaces_its_chunks(
    tmp_path: Path, loader: SourceLoader, ingest_service: SupportService
) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nOld policy: zebra.", encoding="utf-8")
    await ingest_service.ingest(await loader.load(str(path)))
    path.write_text("# Guide\n\nNew policy: giraffe.", encoding="utf-8")

    report = await ingest_service.ingest(await loader.load(str(path)))
    result = await ingest_service.search("giraffe policy", top_k=3)

    assert report.documents_updated == 1
    assert [c.snippet for c in result.citations] == ["New policy: giraffe."]


async def test_ingest_sources_runs_through_the_bootstrap_runtime(
    app_config: AppConfig,
    retrieval_stack: RetrievalStack,
    sqlite_repo: FAQRepository,
    tmp_path: Path,
) -> None:
    (tmp_path / "faq.json").write_text('[{"question": "Q?", "answer": "A."}]', encoding="utf-8")
    engine = RetrievalAnswerEngine(
        retriever=retrieval_stack.retriever, indexer=retrieval_stack.indexer, llm=FakeLLMClient()
    )
    runtime = assemble_runtime(app_config, engine, sqlite_repo)

    [report] = await ingest_sources(runtime, [str(tmp_path / "faq.json")])

    assert report.faqs_added == 1
    assert sqlite_repo._conn is None  # closed afterwards
