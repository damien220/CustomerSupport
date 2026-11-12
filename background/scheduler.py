"""Background re-indexing: ``faq.added`` → APScheduler job → answer engine.

The job chunks the new entry and writes its FTS5 rows and vectors into the
retrieval index (``AnswerEngine.add_document``), then publishes
``faq.indexed`` so the service drops cached answers. A job lost to a restart (the
default in-memory data store) is harmless: the startup index sync picks up
every entry that has no chunks.

``reindex_faq`` must stay a module-level function that reads its collaborators
from module globals: APScheduler stores jobs *by function reference*, and a
persistent data store (``SCHEDULER_DB_URL``) re-imports that reference after a
restart, so the job itself cannot carry live objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apscheduler import AsyncScheduler
from apscheduler.datastores.memory import MemoryDataStore
from apscheduler.eventbrokers.local import LocalEventBroker

from faq_core.events import FAQ_ADDED, build_faq_indexed_event
from observability.log import get_logger

if TYPE_CHECKING:
    from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore

    from faq_core.config import AppConfig
    from faq_core.events import Event, EventBus
    from faq_core.ports import AnswerEngine, FAQStore

_log = get_logger(__name__)

_scheduler: AsyncScheduler | None = None
_engine: AnswerEngine | None = None
_faq_repo: FAQStore | None = None
_events: EventBus | None = None


async def reindex_faq(faq_id: str) -> None:
    """Background job: fetch the FAQ entry by id and index it in the answer engine."""
    if _engine is None or _faq_repo is None:
        return
    entry = await _faq_repo.get(faq_id)
    if entry is None:
        _log.warning("reindex skipped: entry not found", extra={"faq_id": faq_id})
        return
    await _engine.add_document(entry)
    if _events is not None:
        # The new entry is now searchable: cached answers may be stale.
        await _events.publish(build_faq_indexed_event(faq_id, entry.kb_id))


async def _on_faq_added(event: Event) -> None:
    if _scheduler is None:
        return
    faq_id = event.payload.get("faq_id")
    if isinstance(faq_id, str) and faq_id:
        await _scheduler.add_job(reindex_faq, kwargs={"faq_id": faq_id})


def _create_data_store(config: AppConfig) -> MemoryDataStore | SQLAlchemyDataStore:
    """In-process ``MemoryDataStore`` by default; a persistent SQLAlchemy-backed
    store (surviving restarts) when ``AppConfig.SCHEDULER_DB_URL`` is set.
    """
    if not config.SCHEDULER_DB_URL:
        return MemoryDataStore()

    from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(config.SCHEDULER_DB_URL)
    return SQLAlchemyDataStore(engine)


def create_scheduler(
    config: AppConfig,
    engine: AnswerEngine,
    repo: FAQStore,
    events: EventBus,
) -> AsyncScheduler:
    """Create the scheduler and subscribe it to ``faq.added`` on *events*."""
    global _scheduler, _engine, _faq_repo, _events

    _engine = engine
    _faq_repo = repo
    _events = events
    _scheduler = AsyncScheduler(
        data_store=_create_data_store(config),
        event_broker=LocalEventBroker(),
    )
    events.subscribe(FAQ_ADDED, _on_faq_added)
    return _scheduler
