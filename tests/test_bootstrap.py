"""Composition root: seed loading, runtime wiring, and torch-free entry-point imports."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from bootstrap import DEFAULT_SEED_PATH, assemble_runtime, build_runtime, load_seed_drafts, serve
from faq_core.config import AppConfig
from faq_core.errors import IngestError
from faq_core.events import FAQ_ADDED
from faq_core.models import FAQEntry, IngestReport
from faq_core.repository import FAQRepository
from tests.conftest import FakeLLMClient, FakeRAGPipeline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_bundled_sample_faq_parses() -> None:
    drafts = load_seed_drafts(DEFAULT_SEED_PATH)

    assert drafts
    assert all(d.question and d.answer and d.category for d in drafts)


def test_missing_seed_file_yields_no_drafts(tmp_path: Path) -> None:
    assert load_seed_drafts(tmp_path / "absent.json") == []


async def test_serve_seeds_once_and_closes_repository(app_config: AppConfig) -> None:
    engine = FakeRAGPipeline()
    runtime = assemble_runtime(app_config, engine, FAQRepository(app_config))

    async with serve(runtime) as service:
        categories = (await service.list_categories()).categories

    assert categories
    assert len(engine.seeded) == len(load_seed_drafts())
    assert runtime.events.handlers_for(FAQ_ADDED)  # scheduler subscribed
    assert runtime.repo._conn is None
    assert len(engine.sync_calls) == 1  # index synced at startup, after seeding


async def test_serve_syncs_the_index_even_when_the_kb_is_not_empty(
    app_config: AppConfig,
) -> None:
    repo = FAQRepository(app_config)
    await repo.create(FAQEntry(question="Existing?", answer="Yes.", category="c"))
    engine = FakeRAGPipeline()
    runtime = assemble_runtime(app_config, engine, repo)

    async with serve(runtime):
        pass

    assert engine.seeded == []  # not empty → no seeding
    [(entries, documents)] = engine.sync_calls
    assert [e.question for e in entries] == ["Existing?"] and documents == []


async def test_index_is_searchable_after_a_restart(app_config: AppConfig) -> None:
    """Regression: Phase 1 only indexed while seeding an empty DB, so after a
    restart (non-empty DB, fresh in-memory vector store) search found nothing."""
    first = build_runtime(app_config, llm=FakeLLMClient())
    async with serve(first) as service:
        created = await service.add_entry("Can I pay with crypto?", "No, cards only.", "Billing")
        await first.events.drain()  # let the faq.added job start

    def _lose_index(conn: sqlite3.Connection) -> None:
        # Simulate a lost background job / old database: no chunks for one entry.
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM chunks WHERE faq_id = ?", (str(created.id),))
        conn.commit()

    probe = FAQRepository(app_config)
    await probe.run_in_connection(_lose_index)
    await probe.close()

    second = build_runtime(app_config, llm=FakeLLMClient())
    async with serve(second) as service:
        crypto = await service.search("pay with crypto", top_k=1)
        password = await service.search("reset my password", top_k=1)

    assert crypto.sources == [created]
    assert crypto.citations[0].source_id == created.id
    assert password.sources[0].question == "How do I reset my password?"


def test_entry_points_import_without_heavy_or_removed_packages() -> None:
    """``import app, server`` must stay torch-free and not pull in the dropped wheels."""
    forbidden = [
        "torch",
        "sentence_transformers",
        "sqlite_vec",  # loaded only when a repository opens its connection
        "gradio",
        "ui_controller_base",
        "cache_manager",
        "plugin_loader",
        "error_handler",
        "event_dispatcher",
        "db_accessor",
        "inout_models",
        "logger_pkg",
        "notification_manager",
        "typer",
        "streamlit",
        "api_client_base",  # dropped in Phase 3 (LLMGateway)
        "anthropic",  # imported only when an Anthropic gateway is built
    ]
    code = (
        "import sys, app, server\n"
        f"loaded = [m for m in {forbidden!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "assert server.mcp.name == 'Customer Support FAQ'\n"
    )

    completed = subprocess.run(  # noqa: S603 — fixed interpreter + literal code
        [sys.executable, "-c", code], cwd=_PROJECT_ROOT, capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_ingest_cli_prints_reports_and_maps_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import app

    async def _ok(sources: Sequence[str]) -> list[IngestReport]:
        return [IngestReport(source=s, faqs_added=1) for s in sources]

    async def _fail(sources: Sequence[str]) -> list[IngestReport]:
        raise IngestError("ingest source not found", source=sources[0])

    monkeypatch.setattr("bootstrap.run_ingest", _ok)
    assert app.run_ingest(["a.md", "b.csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["source"] for line in lines] == ["a.md", "b.csv"]

    monkeypatch.setattr("bootstrap.run_ingest", _fail)
    assert app.run_ingest(["missing.md"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "ingest_failed"
