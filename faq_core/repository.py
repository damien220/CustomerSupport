"""SQLite-backed storage for FAQ entries, documents, feedback and the support tables.

sqlite3 connections are thread-affine, so all database work runs on one
dedicated worker thread (``self._executor``) and a single connection stays open
for the repository's lifetime. Every write commits explicitly. The schema is
brought up to date by :mod:`faq_core.migrations` when the repository opens.

File databases are opened in WAL mode with a busy timeout
(:data:`BUSY_TIMEOUT_MS`): several processes (API, MCP, ingest and report CLIs)
share one file, and WAL lets readers and the writer proceed concurrently. WAL
needs a local filesystem (not NFS) and keeps ``-wal`` / ``-shm`` files next to
the database, so back it up with ``sqlite3 faq.db ".backup …"`` or
``VACUUM INTO``, never by copying the ``.db`` file alone.

The retrieval index (``faq_core.retrieval``) keeps its chunk, FTS5 and vector
SQL in its own modules but runs it on this same connection through
:meth:`FAQRepository.run_in_connection`, so connection reuse stays here. When
the connection opens, the optional ``sqlite-vec`` extension is loaded; if that
fails (package missing, or a Python built without extension loading) the
repository still works and :attr:`FAQRepository.vector_search_available` is
``False`` — retrieval then degrades to BM25 only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeVar

from faq_core.confidence import Outcome
from faq_core.migrations import apply_migrations, current_version
from faq_core.models import (
    DEFAULT_KB_ID,
    Document,
    DocumentChange,
    DocumentFormat,
    Escalation,
    EscalationActivity,
    EscalationDelivery,
    EscalationStatus,
    FAQEntry,
    FeedbackActivity,
    FeedbackEntry,
    QueryActivity,
    QueryLogRecord,
    SourceRef,
    utc_now,
)
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from faq_core.config import AppConfig
    from faq_core.models import DocumentDraft

_log = get_logger(__name__)

_T = TypeVar("_T")

_FAQ_COLUMNS = "id, kb_id, question, answer, category, created_at"
_DOCUMENT_COLUMNS = (
    "id, kb_id, title, body, format, source_uri, content_hash, updated_at, created_at"
)
_QUERY_LOG_COLUMNS = (
    "id, kb_id, query, surface, top_k, confidence, source_ids, answered, latency_ms, "
    "created_at, session_id, outcome, reasons, citations, answer_preview"
)
_ESCALATION_COLUMNS = (
    "id, kb_id, query_id, reason, status, packet, created_at, session_id, delivery, updated_at"
)

BUSY_TIMEOUT_MS: Final = 5000
"""How long a statement waits for another process's lock before ``database is locked``."""


class FAQRepository:
    """Async facade over one SQLite connection owned by a single worker thread."""

    def __init__(self, config: AppConfig) -> None:
        self._db_path = config.DB_PATH
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faq-repo-db")
        self._conn: sqlite3.Connection | None = None
        self._vector_extension_loaded = False
        # Runs before any event loop exists (bootstrap) — block until migrated.
        self._executor.submit(self._migrate).result()

    # ── connection lifecycle ─────────────────────────────────────────────────

    @property
    def vector_search_available(self) -> bool:
        """``True`` when the ``sqlite-vec`` extension is loaded on the connection."""
        return self._vector_extension_loaded

    def _connection(self) -> sqlite3.Connection:
        """Must only be called from within ``self._executor``'s worker thread."""
        if self._conn is None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=BUSY_TIMEOUT_MS / 1000)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            if not _is_memory_database(self._db_path):
                conn.execute("PRAGMA journal_mode = WAL")  # persistent: set once per file
            self._vector_extension_loaded = _load_vector_extension(conn)
            self._conn = conn
        return self._conn

    def _migrate(self) -> None:
        apply_migrations(self._connection())

    async def _run(self, fn: Callable[[], _T]) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    async def run_in_connection(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run *fn* with the shared connection on the database worker thread.

        For the retrieval index modules, which own their SQL. *fn* must commit
        its own writes (or roll back on failure) before returning.
        """
        return await self._run(lambda: fn(self._connection()))

    async def _write(self, sql: str, params: Sequence[object] = ()) -> int:
        """Execute one write statement, commit, and return the affected row count."""

        def _do() -> int:
            conn = self._connection()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount

        return await self._run(_do)

    async def _write_many(self, sql: str, rows: Sequence[Sequence[object]]) -> None:
        def _do() -> None:
            conn = self._connection()
            conn.executemany(sql, rows)
            conn.commit()

        await self._run(_do)

    async def _fetch(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        def _do() -> list[sqlite3.Row]:
            return self._connection().execute(sql, params).fetchall()

        return await self._run(_do)

    async def schema_version(self) -> int:
        """Return the applied migration version."""
        return await self._run(lambda: current_version(self._connection()))

    async def close(self) -> None:
        """Close the connection and shut down the worker thread (idempotent)."""

        def _do() -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        try:
            await self._run(_do)
        except RuntimeError:
            return  # executor already shut down by an earlier close()
        self._executor.shutdown(wait=True)

    # ── FAQ entries ──────────────────────────────────────────────────────────

    async def create(self, entry: FAQEntry) -> FAQEntry:
        await self._write(
            f"INSERT INTO faq_entries ({_FAQ_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",  # noqa: S608 — constant column list
            _faq_to_row(entry),
        )
        return entry

    async def create_many(self, entries: Sequence[FAQEntry]) -> None:
        """Insert several entries in one transaction."""
        await self._write_many(
            f"INSERT INTO faq_entries ({_FAQ_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",  # noqa: S608 — constant column list
            [_faq_to_row(entry) for entry in entries],
        )

    async def get(self, faq_id: str) -> FAQEntry | None:
        rows = await self._fetch(
            f"SELECT {_FAQ_COLUMNS} FROM faq_entries WHERE id = ?",  # noqa: S608 — constant column list
            (faq_id,),
        )
        return _row_to_faq(rows[0]) if rows else None

    async def list_categories(self, kb_id: str = DEFAULT_KB_ID) -> list[str]:
        rows = await self._fetch(
            "SELECT DISTINCT category FROM faq_entries WHERE kb_id = ? ORDER BY category",
            (kb_id,),
        )
        return [str(row["category"]) for row in rows]

    async def list_all(self, kb_id: str = DEFAULT_KB_ID) -> list[FAQEntry]:
        rows = await self._fetch(
            f"SELECT {_FAQ_COLUMNS} FROM faq_entries WHERE kb_id = ? "  # noqa: S608 — constant column list
            "ORDER BY created_at, id",
            (kb_id,),
        )
        return [_row_to_faq(row) for row in rows]

    async def count(self, kb_id: str = DEFAULT_KB_ID) -> int:
        rows = await self._fetch("SELECT COUNT(*) AS n FROM faq_entries WHERE kb_id = ?", (kb_id,))
        return int(rows[0]["n"])

    # ── feedback ─────────────────────────────────────────────────────────────

    async def save_feedback(self, feedback: FeedbackEntry) -> None:
        """Store *feedback*; a later rating of the same ``(kb_id, query_id)`` replaces it.

        The replacing row takes the new id, rating, comment, helpful flag and
        timestamp, so the metrics count one (the latest) rating per answer.
        """
        await self._write(
            "INSERT INTO feedback (id, kb_id, query_id, rating, comment, helpful, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (kb_id, query_id) DO UPDATE SET id = excluded.id, "
            "rating = excluded.rating, comment = excluded.comment, "
            "helpful = excluded.helpful, created_at = excluded.created_at",
            (
                str(feedback.id),
                feedback.kb_id,
                str(feedback.query_id),
                feedback.rating,
                feedback.comment,
                int(feedback.helpful),
                feedback.created_at.isoformat(),
            ),
        )

    async def list_feedback(self, query_id: str | None = None) -> list[FeedbackEntry]:
        sql = "SELECT * FROM feedback"
        params: tuple[object, ...] = ()
        if query_id is not None:
            sql += " WHERE query_id = ?"
            params = (query_id,)
        rows = await self._fetch(sql + " ORDER BY created_at, id", params)
        return [
            FeedbackEntry(
                id=uuid.UUID(row["id"]),
                kb_id=row["kb_id"],
                query_id=uuid.UUID(row["query_id"]),
                rating=row["rating"],
                comment=row["comment"] or "",
                helpful=bool(row["helpful"]),
                created_at=parse_timestamp(row["created_at"]),
            )
            for row in rows
        ]

    # ── documents ────────────────────────────────────────────────────────────
    # Chunks of entries and documents are written by faq_core.retrieval.

    async def upsert_document(
        self, draft: DocumentDraft, kb_id: str = DEFAULT_KB_ID
    ) -> tuple[Document, DocumentChange]:
        """Store *draft*, keyed by ``(kb_id, source_uri)``.

        Re-ingesting an unchanged source is a no-op (``UNCHANGED``); a changed
        one keeps its id and gets the new title/body/freshness (``UPDATED``).
        """
        content_hash = _document_hash(draft)

        def _do() -> tuple[Document, DocumentChange]:
            conn = self._connection()
            row = conn.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents "  # noqa: S608 — constant column list
                "WHERE kb_id = ? AND source_uri = ?",
                (kb_id, draft.source_uri),
            ).fetchone()
            if row is not None and row["content_hash"] == content_hash:
                return _row_to_document(row), DocumentChange.UNCHANGED
            if row is not None:
                document = _row_to_document(row).model_copy(
                    update={
                        "title": draft.title,
                        "body": draft.body,
                        "format": draft.format,
                        "content_hash": content_hash,
                        "updated_at": draft.updated_at,
                    }
                )
                conn.execute(
                    "UPDATE documents SET title = ?, body = ?, format = ?, content_hash = ?, "
                    "updated_at = ? WHERE id = ?",
                    (
                        document.title,
                        document.body,
                        document.format.value,
                        content_hash,
                        document.updated_at.isoformat(),
                        str(document.id),
                    ),
                )
                change = DocumentChange.UPDATED
            else:
                document = Document(
                    kb_id=kb_id,
                    title=draft.title,
                    body=draft.body,
                    format=draft.format,
                    source_uri=draft.source_uri,
                    content_hash=content_hash,
                    updated_at=draft.updated_at,
                    created_at=utc_now(),
                )
                conn.execute(
                    f"INSERT INTO documents ({_DOCUMENT_COLUMNS}) "  # noqa: S608 — constant column list
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    _document_to_row(document),
                )
                change = DocumentChange.ADDED
            conn.commit()
            return document, change

        return await self._run(_do)

    async def get_document(self, document_id: str) -> Document | None:
        rows = await self._fetch(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id = ?",  # noqa: S608 — constant column list
            (document_id,),
        )
        return _row_to_document(rows[0]) if rows else None

    async def list_documents(self, kb_id: str = DEFAULT_KB_ID) -> list[Document]:
        rows = await self._fetch(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE kb_id = ? "  # noqa: S608 — constant column list
            "ORDER BY created_at, id",
            (kb_id,),
        )
        return [_row_to_document(row) for row in rows]

    # ── query log (also the knowledge-gap log) ───────────────────────────────

    async def log_query(self, record: QueryLogRecord) -> None:
        await self._write(
            f"INSERT INTO query_log ({_QUERY_LOG_COLUMNS}) "  # noqa: S608 — constant column list
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(record.id),
                record.kb_id,
                record.query,
                record.surface,
                record.top_k,
                record.confidence,
                json.dumps([str(s) for s in record.source_ids]),
                int(record.answered),
                record.latency_ms,
                record.created_at.isoformat(),
                record.session_id,
                record.outcome.value,
                json.dumps(record.reasons),
                json.dumps([ref.model_dump(mode="json") for ref in record.citations]),
                record.answer_preview,
            ),
        )

    async def list_queries(
        self, kb_id: str = DEFAULT_KB_ID, limit: int = 100
    ) -> list[QueryLogRecord]:
        rows = await self._fetch(
            f"SELECT {_QUERY_LOG_COLUMNS} FROM query_log WHERE kb_id = ? "  # noqa: S608 — constant column list
            "ORDER BY created_at DESC, id LIMIT ?",
            (kb_id, limit),
        )
        return [_row_to_query(row) for row in rows]

    async def get_query(self, query_id: str) -> QueryLogRecord | None:
        rows = await self._fetch(
            f"SELECT {_QUERY_LOG_COLUMNS} FROM query_log WHERE id = ?",  # noqa: S608 — constant column list
            (query_id,),
        )
        return _row_to_query(rows[0]) if rows else None

    async def list_session_queries(
        self, kb_id: str, session_id: str, limit: int
    ) -> list[QueryLogRecord]:
        """The session's latest *limit* queries, oldest first."""
        rows = await self._fetch(
            f"SELECT {_QUERY_LOG_COLUMNS} FROM query_log "  # noqa: S608 — constant column list
            "WHERE kb_id = ? AND session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (kb_id, session_id, limit),
        )
        return [_row_to_query(row) for row in reversed(rows)]

    async def list_gap_queries(
        self, kb_id: str, since: datetime, limit: int
    ) -> list[QueryLogRecord]:
        """Queries since *since* not answered confidently, or rated badly; newest first."""
        rows = await self._fetch(
            f"SELECT {_QUERY_LOG_COLUMNS} FROM query_log q "  # noqa: S608 — constant column list
            "WHERE q.kb_id = ? AND q.created_at >= ? AND (q.outcome != 'answer' OR EXISTS ("
            "SELECT 1 FROM feedback f WHERE f.query_id = q.id "
            "AND (f.helpful = 0 OR f.rating <= 2))) "
            "ORDER BY q.created_at DESC, q.id LIMIT ?",
            (kb_id, since.isoformat(), limit),
        )
        return [_row_to_query(row) for row in rows]

    # ── resolution-metrics activity (Phase 6; read-only, no query text) ──────
    # Stored timestamps are ISO-8601 strings, so SQL filters on a window widened
    # by a day (robust to offset / precision differences between old and new
    # rows) and the exact half-open window [since, until) is applied after
    # parsing. Every read is capped at *limit* rows, newest kept (the
    # (kb_id, created_at) indexes serve the ORDER BY ... DESC LIMIT), and the
    # result is returned oldest first.

    async def list_query_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[QueryActivity]:
        """The newest *limit* queries logged in ``[since, until)``, oldest first."""
        rows = await self._fetch(
            "SELECT id, session_id, outcome, confidence, created_at FROM query_log "
            "WHERE kb_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (kb_id, *_widened(since, until), limit),
        )
        activity = [
            QueryActivity(
                id=uuid.UUID(row["id"]),
                session_id=row["session_id"],
                outcome=Outcome(row["outcome"]),
                confidence=row["confidence"],
                created_at=parse_timestamp(row["created_at"]),
            )
            for row in rows
        ]
        return sorted(
            (item for item in reversed(activity) if since <= item.created_at < until),
            key=lambda item: item.created_at,
        )

    async def list_escalation_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[EscalationActivity]:
        """The newest *limit* escalation records created in ``[since, until)``, oldest first."""
        rows = await self._fetch(
            "SELECT status, created_at FROM escalations "
            "WHERE kb_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (kb_id, *_widened(since, until), limit),
        )
        activity = [
            EscalationActivity(
                status=EscalationStatus(row["status"]),
                created_at=parse_timestamp(row["created_at"]),
            )
            for row in reversed(rows)
        ]
        return [item for item in activity if since <= item.created_at < until]

    async def list_feedback_activity(
        self, kb_id: str, since: datetime, until: datetime, limit: int
    ) -> list[FeedbackActivity]:
        """The newest *limit* ratings given in ``[since, until)``, oldest first.

        One row per ``(kb_id, query_id)`` (repeat ratings replace earlier ones).
        """
        rows = await self._fetch(
            "SELECT rating, helpful, created_at FROM feedback "
            "WHERE kb_id = ? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (kb_id, *_widened(since, until), limit),
        )
        activity = [
            FeedbackActivity(
                rating=row["rating"],
                helpful=bool(row["helpful"]),
                created_at=parse_timestamp(row["created_at"]),
            )
            for row in reversed(rows)
        ]
        return [item for item in activity if since <= item.created_at < until]

    # ── escalations ──────────────────────────────────────────────────────────

    async def create_escalation(self, escalation: Escalation) -> Escalation:
        await self._write(
            f"INSERT INTO escalations ({_ESCALATION_COLUMNS}) "  # noqa: S608 — constant column list
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(escalation.id),
                escalation.kb_id,
                str(escalation.query_id) if escalation.query_id else None,
                escalation.reason,
                escalation.status.value,
                json.dumps(escalation.packet),
                escalation.created_at.isoformat(),
                escalation.session_id,
                _delivery_json(escalation.delivery),
                (escalation.updated_at or escalation.created_at).isoformat(),
            ),
        )
        return escalation

    async def get_escalation(self, escalation_id: str) -> Escalation | None:
        rows = await self._fetch(
            f"SELECT {_ESCALATION_COLUMNS} FROM escalations WHERE id = ?",  # noqa: S608 — constant column list
            (escalation_id,),
        )
        return _row_to_escalation(rows[0]) if rows else None

    async def list_escalations(
        self, kb_id: str = DEFAULT_KB_ID, status: EscalationStatus | None = None
    ) -> list[Escalation]:
        sql = f"SELECT {_ESCALATION_COLUMNS} FROM escalations WHERE kb_id = ?"  # noqa: S608 — constant column list
        params: tuple[object, ...] = (kb_id,)
        if status is not None:
            sql += " AND status = ?"
            params = (kb_id, status.value)
        rows = await self._fetch(sql + " ORDER BY created_at, id", params)
        return [_row_to_escalation(row) for row in rows]

    async def set_escalation_status(self, escalation_id: str, status: EscalationStatus) -> bool:
        """Update an escalation's status; return ``False`` if the id is unknown."""
        changed = await self._write(
            "UPDATE escalations SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, utc_now().isoformat(), escalation_id),
        )
        return changed > 0

    async def update_escalation_delivery(
        self, escalation_id: str, status: EscalationStatus, delivery: EscalationDelivery
    ) -> None:
        await self._write(
            "UPDATE escalations SET status = ?, delivery = ?, updated_at = ? WHERE id = ?",
            (status.value, _delivery_json(delivery), utc_now().isoformat(), escalation_id),
        )


# ── sqlite-vec ───────────────────────────────────────────────────────────────


def _load_vector_extension(conn: sqlite3.Connection) -> bool:
    """Load ``sqlite-vec`` into *conn*; return ``False`` (and log why) if impossible."""
    try:
        import sqlite_vec  # optional native extension; imported only when a connection opens

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except (ImportError, AttributeError, sqlite3.Error) as err:
        # AttributeError: this Python's sqlite3 was built without extension loading.
        _log.warning(
            "sqlite-vec extension unavailable; dense retrieval disabled, using BM25 only",
            extra={"reason": f"{type(err).__name__}: {err}"},
        )
        return False
    return True


# ── row mapping ──────────────────────────────────────────────────────────────


def _faq_to_row(entry: FAQEntry) -> tuple[str, str, str, str, str, str]:
    return (
        str(entry.id),
        entry.kb_id,
        entry.question,
        entry.answer,
        entry.category,
        entry.created_at.isoformat(),
    )


def _row_to_faq(row: sqlite3.Row) -> FAQEntry:
    return FAQEntry(
        id=uuid.UUID(row["id"]),
        kb_id=row["kb_id"],
        question=row["question"],
        answer=row["answer"],
        category=row["category"],
        created_at=parse_timestamp(row["created_at"]),
    )


def _document_hash(draft: DocumentDraft) -> str:
    parts = (draft.title, draft.format.value, draft.updated_at.isoformat(), draft.body)
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _document_to_row(document: Document) -> tuple[str, ...]:
    return (
        str(document.id),
        document.kb_id,
        document.title,
        document.body,
        document.format.value,
        document.source_uri,
        document.content_hash,
        document.updated_at.isoformat(),
        document.created_at.isoformat(),
    )


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=uuid.UUID(row["id"]),
        kb_id=row["kb_id"],
        title=row["title"],
        body=row["body"],
        format=DocumentFormat(row["format"]),
        source_uri=row["source_uri"],
        content_hash=row["content_hash"],
        updated_at=parse_timestamp(row["updated_at"]),
        created_at=parse_timestamp(row["created_at"]),
    )


def _row_to_query(row: sqlite3.Row) -> QueryLogRecord:
    return QueryLogRecord(
        id=uuid.UUID(row["id"]),
        kb_id=row["kb_id"],
        query=row["query"],
        surface=row["surface"],
        top_k=row["top_k"],
        confidence=row["confidence"],
        source_ids=[uuid.UUID(s) for s in json.loads(row["source_ids"])],
        answered=bool(row["answered"]),
        latency_ms=row["latency_ms"],
        created_at=parse_timestamp(row["created_at"]),
        session_id=row["session_id"],
        outcome=Outcome(row["outcome"]),
        reasons=list(json.loads(row["reasons"])),
        citations=[SourceRef.model_validate(ref) for ref in json.loads(row["citations"])],
        answer_preview=row["answer_preview"],
    )


def _delivery_json(delivery: EscalationDelivery | None) -> str:
    return delivery.model_dump_json() if delivery is not None else "{}"


def _row_to_escalation(row: sqlite3.Row) -> Escalation:
    delivery: dict[str, Any] = json.loads(row["delivery"] or "{}")
    return Escalation(
        id=uuid.UUID(row["id"]),
        kb_id=row["kb_id"],
        query_id=uuid.UUID(row["query_id"]) if row["query_id"] else None,
        reason=row["reason"],
        status=EscalationStatus(row["status"]),
        packet=json.loads(row["packet"]),
        created_at=parse_timestamp(row["created_at"]),
        session_id=row["session_id"],
        delivery=EscalationDelivery.model_validate(delivery) if delivery else None,
        updated_at=parse_timestamp(row["updated_at"]) if row["updated_at"] else None,
    )


_FILTER_MARGIN: Final = timedelta(days=1)


def _is_memory_database(db_path: str) -> bool:
    """``True`` for an in-memory (``:memory:``) or temporary (``""``) database: no WAL there."""
    return db_path in {":memory:", ""}


def _widened(since: datetime, until: datetime) -> tuple[str, str]:
    """SQL bounds one day wider than ``[since, until)`` (exact filter happens after parsing)."""
    return (since - _FILTER_MARGIN).isoformat(), (until + _FILTER_MARGIN).isoformat()


def parse_timestamp(raw: str) -> datetime:
    """Parse a stored ISO-8601 timestamp as aware UTC.

    Rows written by the pre-migration code may lack an offset; they were UTC.
    """
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
