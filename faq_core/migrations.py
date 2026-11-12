"""Versioned SQL migrations for the SQLite support database.

Each :class:`Migration` runs at most once, in ascending ``version`` order, inside
its own transaction together with the ``schema_version`` bookkeeping row, so a
failed migration leaves the database at the previous version. Re-running
:func:`apply_migrations` is a no-op once the database is current.

Migration 1 is the pre-migration ("v2 build") schema written with
``IF NOT EXISTS``: on a database created by the old code it changes nothing and
just records the baseline, so old databases and fresh ones converge on the same
path. **Never edit a released migration — append a new one.**
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(
        version=1,
        name="baseline_faq_and_feedback",
        sql="""
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
        """,
    ),
    Migration(
        version=2,
        name="kb_id_tenancy_column",
        sql="""
        ALTER TABLE faq_entries ADD COLUMN kb_id TEXT NOT NULL DEFAULT 'default';
        ALTER TABLE feedback ADD COLUMN kb_id TEXT NOT NULL DEFAULT 'default';
        CREATE INDEX IF NOT EXISTS idx_faq_entries_kb_category
            ON faq_entries (kb_id, category);
        CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback (query_id);
        """,
    ),
    Migration(
        version=3,
        name="chunks_query_log_escalations",
        sql="""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL DEFAULT 'default',
            faq_id TEXT REFERENCES faq_entries (id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL DEFAULT 0,
            heading TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            source_uri TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE (faq_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks (kb_id);

        CREATE TABLE IF NOT EXISTS query_log (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL DEFAULT 'default',
            query TEXT NOT NULL,
            surface TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            confidence REAL,
            source_ids TEXT NOT NULL DEFAULT '[]',
            answered INTEGER NOT NULL DEFAULT 1,
            latency_ms INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_query_log_kb_created ON query_log (kb_id, created_at);

        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL DEFAULT 'default',
            query_id TEXT,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'sent', 'resolved')),
            packet TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_escalations_kb_status ON escalations (kb_id, status);
        """,
    ),
    # Phase 2 retrieval. The dense-vector table is NOT created here: it needs
    # the optional sqlite-vec extension and the embedder's dimension, so
    # faq_core/retrieval/vector_store.py creates (and re-creates) it at runtime
    # and records its embedder signature in index_meta. FTS5 ships with SQLite.
    Migration(
        version=4,
        name="documents_chunk_metadata_fts",
        sql="""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL DEFAULT 'default',
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            format TEXT NOT NULL DEFAULT 'markdown' CHECK (format IN ('markdown', 'text')),
            source_uri TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (kb_id, source_uri)
        );

        ALTER TABLE chunks ADD COLUMN document_id TEXT
            REFERENCES documents (id) ON DELETE CASCADE;
        ALTER TABLE chunks ADD COLUMN title TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunks ADD COLUMN source_hash TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunks ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
        CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_document_ordinal
            ON chunks (document_id, ordinal);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
            chunk_id UNINDEXED,
            kb_id UNINDEXED,
            title,
            heading,
            content,
            tokenize = 'porter unicode61 remove_diacritics 2'
        );

        CREATE TABLE IF NOT EXISTS index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    ),
    # Phase 4 support workflow: triage results and conversation grouping on the
    # query log (which doubles as the knowledge-gap log), delivery state on
    # escalations. Stored query text is always PII-redacted.
    Migration(
        version=5,
        name="support_workflow_triage_and_delivery",
        sql="""
        ALTER TABLE query_log ADD COLUMN session_id TEXT;
        ALTER TABLE query_log ADD COLUMN outcome TEXT NOT NULL DEFAULT 'answer';
        ALTER TABLE query_log ADD COLUMN reasons TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE query_log ADD COLUMN citations TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE query_log ADD COLUMN answer_preview TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_query_log_session
            ON query_log (kb_id, session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_query_log_outcome
            ON query_log (kb_id, outcome, created_at);

        ALTER TABLE escalations ADD COLUMN session_id TEXT;
        ALTER TABLE escalations ADD COLUMN delivery TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE escalations ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
        """,
    ),
    # Phase 6 review: bounded, indexed metrics reads and one rating per answer.
    # query_log already has idx_query_log_kb_created (migration 3). Duplicate
    # ratings written before this migration are collapsed to the latest one per
    # (kb_id, query_id) — by created_at, then insertion order — so the unique
    # index can be created; record_feedback then upserts.
    Migration(
        version=6,
        name="metrics_indexes_and_unique_feedback",
        sql="""
        CREATE INDEX IF NOT EXISTS idx_feedback_kb_created ON feedback (kb_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_escalations_kb_created
            ON escalations (kb_id, created_at);

        DELETE FROM feedback
        WHERE rowid NOT IN (
            SELECT rowid FROM (
                SELECT rowid, ROW_NUMBER() OVER (
                    PARTITION BY kb_id, query_id ORDER BY created_at DESC, rowid DESC
                ) AS position
                FROM feedback
            )
            WHERE position = 1
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_kb_query
            ON feedback (kb_id, query_id);
        """,
    ),
)

LATEST_VERSION: Final = MIGRATIONS[-1].version

_CREATE_VERSION_TABLE: Final = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at TEXT NOT NULL
    )
"""


class MigrationError(RuntimeError):
    """A migration failed; the database was rolled back to its previous version."""


def current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version (0 for an unmanaged database)."""
    conn.execute(_CREATE_VERSION_TABLE)
    conn.commit()
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(row[0])


def apply_migrations(
    conn: sqlite3.Connection, migrations: tuple[Migration, ...] = MIGRATIONS
) -> list[int]:
    """Apply every pending migration in order and return the versions applied.

    Raises:
        MigrationError: if a migration's SQL fails (that migration is rolled back).
    """
    _check_ordering(migrations)
    applied_now: list[int] = []
    start = current_version(conn)
    for migration in migrations:
        if migration.version <= start:
            continue
        _apply_one(conn, migration)
        applied_now.append(migration.version)
    return applied_now


def _apply_one(conn: sqlite3.Connection, migration: Migration) -> None:
    applied_at = datetime.now(UTC).isoformat()
    # executescript() commits any open transaction first and runs statements
    # verbatim, so the explicit BEGIN/COMMIT makes DDL + bookkeeping atomic.
    script = (
        "BEGIN;\n"
        f"{migration.sql}\n"
        "INSERT INTO schema_version (version, name, applied_at) "
        f"VALUES ({migration.version}, {_sql_literal(migration.name)}, "
        f"{_sql_literal(applied_at)});\n"
        "COMMIT;"
    )
    try:
        conn.executescript(script)
    except sqlite3.Error as err:
        if conn.in_transaction:
            conn.rollback()
        raise MigrationError(
            f"migration {migration.version} ({migration.name}) failed: {err}"
        ) from err


def _sql_literal(value: str) -> str:
    # Only ever called with migration names and ISO timestamps defined in this
    # module, never user input; quoting still escapes embedded quotes.
    return "'" + value.replace("'", "''") + "'"


def _check_ordering(migrations: tuple[Migration, ...]) -> None:
    versions = [m.version for m in migrations]
    if versions != sorted(set(versions)) or (versions and versions[0] < 1):
        raise ValueError(f"migrations must have unique ascending versions >= 1, got {versions}")
