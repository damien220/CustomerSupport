from __future__ import annotations

import logging

import pytest

from faq_core.events import (
    FAQ_ADDED,
    FEEDBACK_RECEIVED,
    QUERY_ANSWERED,
    Event,
    EventBus,
    build_faq_added_event,
    build_feedback_received_event,
    build_query_answered_event,
)

# ── builders ─────────────────────────────────────────────────────────────────


def test_build_faq_added_event_carries_id_category_and_kb() -> None:
    event = build_faq_added_event("faq-123", "billing", "default")

    assert event.kind == FAQ_ADDED
    assert event.source == "add_faq_entry"
    assert event.payload == {"faq_id": "faq-123", "category": "billing", "kb_id": "default"}


def test_build_query_answered_event_truncates_long_preview() -> None:
    event = build_query_answered_event("qid", "what is x?", "x" * 500, cached=False)

    assert event.kind == QUERY_ANSWERED
    assert len(str(event.payload["answer_preview"])) == 200


def test_build_query_answered_event_keeps_short_preview_intact() -> None:
    event = build_query_answered_event("qid", "q", "short answer", cached=True)

    assert event.payload["answer_preview"] == "short answer"
    assert event.payload["cached"] is True


def test_build_feedback_received_event() -> None:
    event = build_feedback_received_event("fid", "qid", 4)

    assert event.kind == FEEDBACK_RECEIVED
    assert event.payload == {"feedback_id": "fid", "query_id": "qid", "rating": 4}


# ── bus ──────────────────────────────────────────────────────────────────────


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


async def test_emit_runs_matching_handlers_in_background() -> None:
    bus = EventBus()
    recorder = Recorder()
    bus.subscribe(FAQ_ADDED, recorder)
    event = build_faq_added_event("1", "c", "default")

    bus.emit(event)
    assert recorder.events == []  # not yet run: emit never blocks the publisher
    await bus.drain()

    assert recorder.events == [event]


@pytest.mark.parametrize(
    ("pattern", "matches"), [("*", True), ("faq.*", True), ("faq.added", True), ("query.*", False)]
)
async def test_subscription_patterns(pattern: str, matches: bool) -> None:
    bus = EventBus()
    recorder = Recorder()
    bus.subscribe(pattern, recorder)

    await bus.publish(build_faq_added_event("1", "c", "default"))

    assert bool(recorder.events) is matches


async def test_duplicate_subscription_is_ignored_and_unsubscribe_removes() -> None:
    bus = EventBus()
    recorder = Recorder()
    bus.subscribe(FAQ_ADDED, recorder)
    bus.subscribe(FAQ_ADDED, recorder)
    await bus.publish(build_faq_added_event("1", "c", "default"))

    bus.unsubscribe(FAQ_ADDED, recorder)
    await bus.publish(build_faq_added_event("2", "c", "default"))

    assert len(recorder.events) == 1


async def test_failing_handler_is_logged_and_does_not_stop_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    recorder = Recorder()

    async def broken(_event: Event) -> None:
        raise RuntimeError("subscriber bug")

    bus.subscribe(FAQ_ADDED, broken)
    bus.subscribe(FAQ_ADDED, recorder)

    with caplog.at_level(logging.ERROR):
        bus.emit(build_faq_added_event("1", "c", "default"))
        await bus.drain()

    assert len(recorder.events) == 1
    assert "event handler failed" in caplog.text
