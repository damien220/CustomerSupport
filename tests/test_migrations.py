"""Versioned migrations: fresh databases, pre-migration (old-schema) databases, failures."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from faq_core.config import AppConfig
from faq_core.migrations import (
    LATEST_VERSION,
    MIGRATIONS,
    Migration,
    MigrationError,
    apply_migrations,
    current_version,
)
from faq_core.repository import FAQRepository

# Exactly what the pre-revamp FAQRepository._init_schema created (no schema_version).
_OLD_SCHEMA = """
    CREATE TABLE IF NOT EXISTS faq_entries (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        query_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT DEFAULT '',
        helpful INTEGER DEFAULT 1,
        created_at TEXT NOT NULL
    );
"""

_EXPECTED_TABLES = {
    "faq_entries",
    "feedback",
    "chunks",
    "query_log",
    "escalations",
    "schema_version",
    "documents",
    "chunks_fts",
    "index_meta",
}


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


def _tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def test_fresh_database_gets_every_table(conn: sqlite3.Connection) -> None:
    applied = apply_migrations(conn)

    assert applied == [m.version for m in MIGRATIONS]
    assert _tables(conn) >= _EXPECTED_TABLES
    assert "kb_id" in _columns(conn, "faq_entries")
    assert current_version(conn) == LATEST_VERSION


def test_rerunning_migrations_is_a_no_op(conn: sqlite3.Connection) -> None:
    apply_migrations(conn)

    assert apply_migrations(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(MIGRATIONS)


async def test_old_schema_database_migrates_and_keeps_data(tmp_path: Path) -> None:
    db_path = tmp_path / "faq.db"
    faq_id, feedback_id, query_id = (str(uuid.uuid4()) for _ in range(3))
    legacy = sqlite3.connect(db_path)
    legacy.executescript(_OLD_SCHEMA)
    legacy.execute(
        "INSERT INTO faq_entries VALUES (?, ?, ?, ?, ?)",
        (faq_id, "Old question?", "Old answer.", "Legacy", "2026-07-01T10:00:00+00:00"),
    )
    legacy.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
        (feedback_id, query_id, 4, "ok", 1, "2026-07-01T10:05:00"),  # naive timestamp
    )
    legacy.commit()
    legacy.close()

    repo = FAQRepository(AppConfig(_env_file=None, DB_PATH=str(db_path)))  # type: ignore[call-arg]
    try:
        entry = await repo.get(faq_id)
        [feedback] = await repo.list_feedback(query_id)
        version = await repo.schema_version()
    finally:
        await repo.close()

    assert entry is not None
    assert (entry.question, entry.category, entry.kb_id) == ("Old question?", "Legacy", "default")
    assert feedback.rating == 4 and feedback.kb_id == "default"
    assert feedback.created_at.tzinfo is not None
    assert version == LATEST_VERSION
    migrated = sqlite3.connect(db_path)
    try:
        assert _tables(migrated) >= _EXPECTED_TABLES
    finally:
        migrated.close()


def test_failing_migration_rolls_back_and_keeps_previous_version(
    conn: sqlite3.Connection,
) -> None:
    apply_migrations(conn)
    broken = Migration(
        version=LATEST_VERSION + 1,
        name="broken",
        sql="CREATE TABLE half_done (id TEXT); INSERT INTO no_such_table VALUES (1);",
    )

    with pytest.raises(MigrationError, match="broken"):
        apply_migrations(conn, (*MIGRATIONS, broken))

    assert current_version(conn) == LATEST_VERSION
    assert "half_done" not in _tables(conn)


def test_migrations_must_be_unique_and_ascending(conn: sqlite3.Connection) -> None:
    duplicate = (MIGRATIONS[0], MIGRATIONS[0])

    with pytest.raises(ValueError, match="ascending"):
        apply_migrations(conn, duplicate)


def test_migration_4_adds_retrieval_columns_to_a_version_3_database(
    conn: sqlite3.Connection,
) -> None:
    apply_migrations(conn, MIGRATIONS[:3])
    faq_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO faq_entries (id, question, answer, category, created_at) "
        "VALUES (?, 'Q', 'A', 'c', '2026-07-01T10:00:00+00:00')",
        (faq_id,),
    )
    conn.execute(
        "INSERT INTO chunks (id, faq_id, ordinal, content, created_at) "
        "VALUES ('c1', ?, 0, 'old chunk', '2026-07-01T10:00:00+00:00')",
        (faq_id,),
    )
    conn.commit()

    assert apply_migrations(conn, MIGRATIONS[:4]) == [4]

    assert {"document_id", "title", "content_hash", "source_hash", "updated_at"} <= _columns(
        conn, "chunks"
    )
    row = conn.execute("SELECT title, source_hash, updated_at FROM chunks").fetchone()
    assert row == ("", "", "")  # defaults; the startup index sync re-chunks this entry
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, kb_id, title, heading, content) "
        "VALUES ('c1', 'default', 'Q', '', 'resetting passwords')"
    )
    match = conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'reset'")
    assert match.fetchall() == [("c1",)]  # porter stemming is on


def test_migration_5_adds_triage_columns_and_keeps_old_query_log_rows(
    conn: sqlite3.Connection,
) -> None:
    apply_migrations(conn, MIGRATIONS[:4])
    conn.execute(
        "INSERT INTO query_log (id, query, surface, top_k, created_at) "
        "VALUES ('q1', 'old query', 'search_faq', 3, '2026-07-01T10:00:00+00:00')"
    )
    conn.commit()

    assert apply_migrations(conn, MIGRATIONS[:5]) == [5]

    assert {"session_id", "outcome", "reasons", "citations", "answer_preview"} <= _columns(
        conn, "query_log"
    )
    assert {"session_id", "delivery", "updated_at"} <= _columns(conn, "escalations")
    row = conn.execute("SELECT outcome, reasons, citations FROM query_log").fetchone()
    assert row == ("answer", "[]", "[]")


def _indexes(connection: sqlite3.Connection, table: str) -> dict[str, bool]:
    """Index name -> unique, for *table*."""
    return {row[1]: bool(row[2]) for row in connection.execute(f"PRAGMA index_list({table})")}


def test_migration_6_indexes_metrics_reads_and_keeps_the_latest_rating_per_query(
    conn: sqlite3.Connection,
) -> None:
    apply_migrations(conn, MIGRATIONS[:5])
    conn.executemany(
        "INSERT INTO feedback (id, kb_id, query_id, rating, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            ("f1", "default", "q1", 1, "2026-07-01T10:00:00+00:00"),
            ("f2", "default", "q1", 1, "2026-07-01T10:05:00+00:00"),
            ("f3", "default", "q1", 5, "2026-07-02T09:00:00+00:00"),  # latest for q1
            ("f4", "default", "q2", 2, "2026-07-01T10:00:00+00:00"),
            ("f5", "other", "q1", 3, "2026-07-01T10:00:00+00:00"),  # other knowledge base
        ],
    )
    conn.commit()

    assert apply_migrations(conn) == [6]

    kept = conn.execute("SELECT id, rating FROM feedback ORDER BY id").fetchall()
    assert kept == [("f3", 5), ("f4", 2), ("f5", 3)]
    feedback_indexes = _indexes(conn, "feedback")
    assert feedback_indexes["idx_feedback_kb_query"] is True
    assert "idx_feedback_kb_created" in feedback_indexes
    assert "idx_escalations_kb_created" in _indexes(conn, "escalations")
    assert "idx_query_log_kb_created" in _indexes(conn, "query_log")  # from migration 3
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO feedback (id, kb_id, query_id, rating, created_at) "
            "VALUES ('f6', 'default', 'q2', 4, '2026-07-03T00:00:00+00:00')"
        )
