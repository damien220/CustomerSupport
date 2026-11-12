from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from apscheduler import AsyncScheduler
from apscheduler.datastores.memory import MemoryDataStore

import background.scheduler as scheduler_module
from background.scheduler import _on_faq_added, create_scheduler, reindex_faq
from faq_core.config import AppConfig
from faq_core.events import FAQ_ADDED, Event, EventBus
from tests.conftest import FakeFAQRepository, FakeRAGPipeline, make_entry


@pytest.fixture(autouse=True)
def _reset_scheduler_globals() -> Iterator[None]:
    yield
    scheduler_module._scheduler = None
    scheduler_module._engine = None
    scheduler_module._faq_repo = None


def _faq_added(payload: dict[str, str]) -> Event:
    return Event(kind=FAQ_ADDED, source="test", payload=payload)


async def test_reindex_faq_adds_document_when_entry_exists(
    fake_rag_pipeline: FakeRAGPipeline,
) -> None:
    entry = make_entry()
    scheduler_module._engine = fake_rag_pipeline
    scheduler_module._faq_repo = FakeFAQRepository(entries={"faq-1": entry})

    await reindex_faq("faq-1")

    assert fake_rag_pipeline.added == [entry]


async def test_reindex_faq_is_noop_when_entry_missing(fake_rag_pipeline: FakeRAGPipeline) -> None:
    scheduler_module._engine = fake_rag_pipeline
    scheduler_module._faq_repo = FakeFAQRepository(entries={})

    await reindex_faq("does-not-exist")

    assert fake_rag_pipeline.added == []


async def test_reindex_faq_is_noop_when_globals_unset() -> None:
    await reindex_faq("faq-1")  # must not raise


async def test_on_faq_added_is_noop_without_scheduler() -> None:
    await _on_faq_added(_faq_added({"faq_id": "faq-1"}))  # must not raise


async def test_on_faq_added_schedules_reindex_job(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)
    add_job = AsyncMock(return_value=None)
    monkeypatch.setattr(AsyncScheduler, "add_job", add_job)

    await _on_faq_added(_faq_added({"faq_id": "faq-1"}))

    add_job.assert_awaited_once_with(reindex_faq, kwargs={"faq_id": "faq-1"})


async def test_on_faq_added_is_noop_when_payload_missing_faq_id(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)
    add_job = AsyncMock(return_value=None)
    monkeypatch.setattr(AsyncScheduler, "add_job", add_job)

    await _on_faq_added(_faq_added({}))

    add_job.assert_not_awaited()


def test_create_scheduler_subscribes_to_faq_added(
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)

    assert event_bus.handlers_for(FAQ_ADDED) == [_on_faq_added]


def test_create_scheduler_defaults_to_memory_data_store(
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    scheduler = create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)

    assert isinstance(scheduler.data_store, MemoryDataStore)


def test_create_scheduler_uses_sqlalchemy_store_when_db_url_set(
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore

    app_config.SCHEDULER_DB_URL = "sqlite+aiosqlite:///:memory:"

    scheduler = create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)

    assert isinstance(scheduler.data_store, SQLAlchemyDataStore)
