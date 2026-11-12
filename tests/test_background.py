"""Integration: ``faq.added`` → a *running* APScheduler actually executes the re-index.

``test_scheduler.py`` covers the unit level (mocked ``add_job``). These tests
start a real ``AsyncScheduler`` and observe ``reindex_faq`` reach the answer
engine — driven directly through the subscriber, through the event bus, and
through ``SupportService.add_entry``. Lite: APScheduler + the in-process bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import pytest

import background.scheduler as scheduler_module
from background.scheduler import _on_faq_added, create_scheduler
from faq_core.config import AppConfig
from faq_core.events import EventBus, build_faq_added_event
from faq_core.service import SupportService
from tests.conftest import FakeFAQRepository, FakeRAGPipeline, make_entry


@pytest.fixture(autouse=True)
def _reset_scheduler_globals() -> Iterator[None]:
    """Prevent this module's real scheduler from leaking into other test files."""
    yield
    scheduler_module._scheduler = None
    scheduler_module._engine = None
    scheduler_module._faq_repo = None


async def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met before timeout")


async def test_subscriber_schedules_job_that_runs_reindex(
    app_config: AppConfig, fake_rag_pipeline: FakeRAGPipeline, event_bus: EventBus
) -> None:
    entry = make_entry()
    repo = FakeFAQRepository(entries={"faq-1": entry})
    scheduler = create_scheduler(app_config, fake_rag_pipeline, repo, event_bus)

    async with scheduler:
        await scheduler.start_in_background()
        await _on_faq_added(build_faq_added_event("faq-1", "Account", "default"))
        await _wait_until(lambda: bool(fake_rag_pipeline.added))

    assert fake_rag_pipeline.added == [entry]


async def test_bus_emit_drives_reindex_end_to_end(
    app_config: AppConfig, fake_rag_pipeline: FakeRAGPipeline, event_bus: EventBus
) -> None:
    entry = make_entry()
    repo = FakeFAQRepository(entries={"faq-1": entry})
    scheduler = create_scheduler(app_config, fake_rag_pipeline, repo, event_bus)

    async with scheduler:
        await scheduler.start_in_background()
        event_bus.emit(build_faq_added_event("faq-1", "Account", "default"))
        await _wait_until(lambda: bool(fake_rag_pipeline.added))

    assert fake_rag_pipeline.added == [entry]


async def test_service_add_entry_is_reindexed_in_background(
    app_config: AppConfig,
    fake_rag_pipeline: FakeRAGPipeline,
    fake_faq_repository: FakeFAQRepository,
    event_bus: EventBus,
) -> None:
    service = SupportService(engine=fake_rag_pipeline, store=fake_faq_repository, events=event_bus)
    scheduler = create_scheduler(app_config, fake_rag_pipeline, fake_faq_repository, event_bus)

    async with scheduler:
        await scheduler.start_in_background()
        entry = await service.add_entry("Q?", "A.", "Billing")
        await _wait_until(lambda: bool(fake_rag_pipeline.added))

    assert fake_rag_pipeline.added == [entry]
