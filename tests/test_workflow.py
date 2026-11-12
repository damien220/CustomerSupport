"""Phase 4 support workflow through ``SupportService`` (boundary fakes only).

Covers outcomes in api and client mode, every escalation trigger, the
self-check, redaction before logging / the LLM, events, the escalation packet
and its delivery, knowledge-gap clustering and answer-cache invalidation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from faq_core.confidence import Outcome
from faq_core.errors import EscalationNotFoundError, InvalidRequestError, QueryNotFoundError
from faq_core.events import (
    ESCALATION_CREATED,
    FAQ_INDEXED,
    GAP_LOGGED,
    QUERY_ANSWERED,
    QUERY_ESCALATED,
    Event,
    EventBus,
    build_faq_indexed_event,
)
from faq_core.handoff import PACKET_SCHEMA
from faq_core.models import (
    EscalationDelivery,
    EscalationStatus,
    FAQDraft,
    IngestBatch,
    LLMMode,
    QueryLogRecord,
    utc_now,
)
from faq_core.pii import PIIRedactor
from faq_core.service import (
    CAVEAT_INSTRUCTION,
    ESCALATE_INSTRUCTION,
    SupportPolicy,
    SupportService,
)
from faq_core.triage import HANDOFF_MESSAGE
from tests.conftest import (
    MEDIUM_EVIDENCE,
    WEAK_EVIDENCE,
    FakeFAQRepository,
    FakeRAGPipeline,
    make_entry,
)


def _service(
    store: FakeFAQRepository,
    events: EventBus,
    engine: FakeRAGPipeline | None = None,
    **kwargs: object,
) -> SupportService:
    return SupportService(
        engine=engine or FakeRAGPipeline(sources=[make_entry()]),
        store=store,
        events=events,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def recorded(event_bus: EventBus) -> list[Event]:
    events: list[Event] = []

    async def _record(event: Event) -> None:
        events.append(event)

    event_bus.subscribe("*", _record)
    return events


class _RecordingSink:
    """In-memory ``EscalationSink`` (forwarding, always succeeds)."""

    name = "recording"
    forwards = True

    def __init__(self) -> None:
        self.delivered: list[dict[str, object]] = []

    async def deliver(self, escalation: object) -> EscalationDelivery:
        self.delivered.append(escalation.packet)  # type: ignore[attr-defined]
        return EscalationDelivery(sink=self.name, delivered=True, attempts=1)


# ── outcomes ─────────────────────────────────────────────────────────────────


async def test_confident_answer_is_generated_and_logged(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, recorded: list[Event]
) -> None:
    service = _service(fake_faq_repository, event_bus)

    result = await service.search("How do I reset my password?", session_id="s-1")
    await event_bus.drain()

    assert (result.outcome, result.escalate, result.reasons) == (Outcome.ANSWER, False, [])
    assert result.generated and result.answer.startswith("answer to")
    assert result.confidence_detail is not None
    assert result.confidence_detail.level == "high"
    assert result.confidence_detail.calibrated is False
    [logged] = fake_faq_repository.queries
    assert logged.id == result.query_id
    assert (logged.surface, logged.outcome, logged.session_id) == ("search_faq", "answer", "s-1")
    assert logged.citations and logged.answer_preview.startswith("answer to")
    assert [e.kind for e in recorded] == [QUERY_ANSWERED]


async def test_medium_confidence_asks_for_a_caveat(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, recorded: list[Event]
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], evidence=MEDIUM_EVIDENCE)
    service = _service(fake_faq_repository, event_bus, engine)

    result = await service.search("reset password?")
    await event_bus.drain()

    assert result.outcome is Outcome.ANSWER_WITH_CAVEAT and not result.escalate
    assert result.caveat and result.generated
    assert engine.generate_caveats == [True]
    assert "moderate_confidence" in result.reasons
    assert [e.kind for e in recorded] == [QUERY_ANSWERED, GAP_LOGGED]


async def test_low_confidence_escalates_without_calling_the_llm(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, recorded: list[Event]
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], evidence=WEAK_EVIDENCE)
    service = _service(fake_faq_repository, event_bus, engine)

    result = await service.answer("what is the weather tomorrow")
    await event_bus.drain()

    assert result.escalate and result.outcome is Outcome.ESCALATE
    assert result.answer == HANDOFF_MESSAGE and result.generated is False
    assert result.reasons[0] == "low_confidence"
    assert engine.search_calls == []  # no made-up LLM answer
    assert result.citations  # the passages are still returned
    assert {e.kind for e in recorded} == {QUERY_ANSWERED, QUERY_ESCALATED, GAP_LOGGED}
    assert fake_faq_repository.queries[0].answered is False


async def test_no_passages_escalates(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    service = _service(fake_faq_repository, event_bus, FakeRAGPipeline())

    result = await service.search("anything")

    assert result.escalate and result.reasons[:2] == ["low_confidence", "no_passages"]
    assert result.confidence == 0.0


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("this is ridiculous, how do I reset my password", "frustration"),
        ("reset password for jane@example.com", "pii_detected"),
        ("my lawyer says the reset failed", "legal_topic"),
        ("I got a rash from the product", "medical_topic"),
    ],
)
async def test_text_triggers_escalate_even_with_strong_retrieval(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, query: str, reason: str
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()])
    service = _service(fake_faq_repository, event_bus, engine)

    result = await service.answer(query)

    assert result.escalate and reason in result.reasons
    assert engine.search_calls == []
    assert result.confidence_detail is not None
    assert result.confidence_detail.level == "high"  # confidence itself was fine


async def test_repeated_rephrase_in_a_session_escalates(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    service = _service(fake_faq_repository, event_bus)
    session = "chat-42"

    first = await service.search("how do I reset my password", session_id=session)
    second = await service.search("password reset not working", session_id=session)
    third = await service.search("reset password not working again", session_id=session)
    other_session = await service.search("reset password not working again", session_id="x")

    assert not first.escalate and not second.escalate
    assert third.escalate and "repeated_rephrase" in third.reasons
    assert not other_session.escalate


async def test_self_check_rejection_escalates_and_drops_the_answer(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], supported=False)
    service = _service(
        fake_faq_repository, event_bus, engine, policy=SupportPolicy(self_check=True)
    )

    result = await service.answer("How do I reset my password?")

    assert engine.verify_calls == ["How do I reset my password?"]
    assert result.escalate and result.reasons[-1] == "self_check_failed"
    assert result.answer == HANDOFF_MESSAGE
    assert result.confidence_detail is not None
    assert result.confidence_detail.basis == "retrieval_signals+llm_self_check"


async def test_self_check_is_off_by_default(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], supported=False)
    service = _service(fake_faq_repository, event_bus, engine)

    result = await service.answer("How do I reset my password?")

    assert engine.verify_calls == [] and not result.escalate


def test_self_check_config_never_applies_in_client_mode(app_config: object) -> None:
    from faq_core.config import AppConfig

    assert isinstance(app_config, AppConfig)
    client = app_config.model_copy(
        update={"LLM_MODE": LLMMode.CLIENT, "CONFIDENCE_SELF_CHECK": True}
    )
    api = app_config.model_copy(update={"CONFIDENCE_SELF_CHECK": True})

    assert SupportPolicy.from_config(client).self_check is False
    assert SupportPolicy.from_config(api).self_check is True


# ── client mode ──────────────────────────────────────────────────────────────


async def test_client_mode_search_kb_escalates_with_instructions_and_no_llm(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    engine = FakeRAGPipeline(
        sources=[make_entry()], generates_answers=False, evidence=WEAK_EVIDENCE
    )
    service = _service(
        fake_faq_repository,
        event_bus,
        engine,
        llm_mode=LLMMode.CLIENT,
        policy=SupportPolicy(self_check=True),  # must still never run in client mode
    )

    kb = await service.search_kb("explain quantum physics")
    faq = await service.search("explain quantum physics")

    assert kb.escalate and kb.handling_instructions[0] == ESCALATE_INSTRUCTION
    assert faq.escalate and faq.answer == "" and faq.generated is False
    assert engine.verify_calls == [] and engine.search_calls == []
    escalation = await service.create_escalation(query_id=kb.query_id)
    assert escalation.status is EscalationStatus.OPEN  # default log sink keeps it open
    assert escalation.delivery is not None and escalation.delivery.sink == "log"


async def test_search_kb_caveat_instruction(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], evidence=MEDIUM_EVIDENCE)
    service = _service(fake_faq_repository, event_bus, engine)

    result = await service.search_kb("reset password")

    assert result.handling_instructions[0] == CAVEAT_INSTRUCTION
    assert result.outcome is Outcome.ANSWER_WITH_CAVEAT


# ── redaction ────────────────────────────────────────────────────────────────


async def test_pii_never_reaches_engine_log_or_events(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, recorded: list[Event]
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()])
    service = _service(
        fake_faq_repository,
        event_bus,
        engine,
        policy=SupportPolicy.from_config(_config(ESCALATION_TRIGGERS="low_confidence")),
    )

    result = await service.answer("reset password, card 4111 1111 1111 1111, mail a@b.io")
    await event_bus.drain()

    assert result.pii_detected == ["credit_card", "email"]
    assert not result.escalate  # the pii trigger is disabled here
    assert engine.retrieve_calls == [("reset password, card , mail", 3)]
    assert engine.search_calls[0][0] == "reset password, card [CREDIT_CARD], mail [EMAIL]"
    logged = fake_faq_repository.queries[0].query
    assert "4111" not in logged and "a@b.io" not in logged
    assert all("a@b.io" not in str(e.payload) for e in recorded)
    assert result.query.endswith("a@b.io")  # the caller's own text is echoed back


async def test_redaction_off_still_triggers_pii_escalation(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    service = _service(fake_faq_repository, event_bus, redactor=PIIRedactor(enabled=False))

    result = await service.search("my email is a@b.io")

    assert "pii_detected" in result.reasons
    assert fake_faq_repository.queries[0].query == "my email is a@b.io"


async def test_feedback_comment_is_redacted(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    query_id = (await service.search_kb("hello")).query_id

    await service.record_feedback(query_id, 1, "call me on +44 20 7946 0958", helpful=False)

    assert fake_faq_repository.saved_feedback[0].comment == "call me on [PHONE]"


# ── escalation hand-off ──────────────────────────────────────────────────────


async def test_escalation_packet_has_goal_sources_attempts_reasons_and_redacted_transcript(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus, recorded: list[Event]
) -> None:
    sink = _RecordingSink()
    service = _service(fake_faq_repository, event_bus, sink=sink)
    session = "conv-7"
    await service.search("How do I reset my password?", session_id=session)
    last = await service.search("still no reset email at jo@x.org", session_id=session)

    result = await service.create_escalation(
        query_id=last.query_id,
        reason="customer asked for a human",
        transcript=[
            {"role": "customer", "text": "my card is 4111 1111 1111 1111"},
            {"role": "assistant", "text": "I can't see card details."},
        ],
    )
    await event_bus.drain()

    packet = result.packet
    assert packet["schema"] == PACKET_SCHEMA
    assert packet["customer_goal"] == "How do I reset my password?"
    assert packet["session_id"] == session and packet["query_id"] == str(last.query_id)
    assert result.reasons[0] == "requested" and "pii_detected" in result.reasons
    assert len(packet["attempts"]) == 2  # type: ignore[arg-type]
    assert packet["sources"] and packet["sources"][0]["title"] == "Reset password?"  # type: ignore[index]
    assert packet["transcript"][0] == {"role": "customer", "text": "my card is [CREDIT_CARD]"}  # type: ignore[index]
    assert "jo@x.org" not in str(packet) and "4111" not in str(packet)
    assert set(packet["pii_detected"]) == {"credit_card", "email"}  # type: ignore[arg-type]
    assert result.status is EscalationStatus.SENT and sink.delivered == [packet]
    stored = fake_faq_repository.escalations[str(result.escalation_id)]
    assert stored.status is EscalationStatus.SENT and stored.delivery is not None
    assert ESCALATION_CREATED in [e.kind for e in recorded]
    assert (await service.get_escalation(str(result.escalation_id))).packet == packet


async def test_escalation_from_a_goal_only_and_transcript_synthesised_from_attempts(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    service = _service(fake_faq_repository, event_bus)
    await service.search("How do I reset my password?", session_id="s")

    by_goal = await service.create_escalation(customer_goal="Close my account; tel +1 555 123 4567")
    by_session = await service.create_escalation(session_id="s")

    assert by_goal.packet["customer_goal"] == "Close my account; tel [PHONE]"
    assert by_goal.packet["attempts"] == []
    roles = [turn["role"] for turn in by_session.packet["transcript"]]  # type: ignore[union-attr]
    assert roles == ["customer", "assistant"]


async def test_escalation_input_errors(service: SupportService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.create_escalation()
    with pytest.raises(QueryNotFoundError):
        await service.create_escalation(query_id=uuid.uuid4())
    with pytest.raises(InvalidRequestError):
        await service.create_escalation(session_id="bad id with spaces")
    with pytest.raises(InvalidRequestError):
        await service.get_escalation("not-a-uuid")
    with pytest.raises(EscalationNotFoundError):
        await service.get_escalation(str(uuid.uuid4()))


# ── knowledge gaps ───────────────────────────────────────────────────────────


async def test_knowledge_gaps_cluster_and_rank_unanswered_queries(
    fake_faq_repository: FakeFAQRepository, event_bus: EventBus
) -> None:
    engine = FakeRAGPipeline(sources=[make_entry()], evidence=WEAK_EVIDENCE)
    service = _service(fake_faq_repository, event_bus, engine)
    for query in (
        "do you sell gift cards",
        "can I buy a gift card",
        "gift cards for christmas",
        "student discount",
    ):
        await service.search(query)
    confident = _service(fake_faq_repository, event_bus)
    answered = await confident.search("How do I reset my password?")
    await confident.record_feedback(answered.query_id, 1, helpful=False)

    gaps = await service.list_knowledge_gaps()
    filtered = await service.list_knowledge_gaps(min_count=2)

    assert gaps.scanned == 5
    top = gaps.gaps[0]
    assert top.count == 3 and "gift" in top.representative_query
    assert top.outcomes == {"escalate": 3} and top.reasons["low_confidence"] == 3
    assert top.first_seen <= top.last_seen
    assert {g.representative_query for g in gaps.gaps[1:]} == {
        "student discount",
        "How do I reset my password?",  # negative feedback puts it in the gap log
    }
    assert [g.count for g in filtered.gaps] == [3]


async def test_knowledge_gap_parameters_are_bounded(service: SupportService) -> None:
    with pytest.raises(InvalidRequestError):
        await service.list_knowledge_gaps(limit=0)
    with pytest.raises(InvalidRequestError):
        await service.list_knowledge_gaps(since="not a date")
    since = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    result = await service.list_knowledge_gaps(since=since)
    assert result.since.tzinfo is not None


async def test_old_gaps_are_outside_the_default_window(
    service: SupportService, fake_faq_repository: FakeFAQRepository
) -> None:
    fake_faq_repository.queries.append(
        QueryLogRecord(
            query="ancient question",
            surface="search_faq",
            top_k=3,
            outcome=Outcome.ESCALATE,
            created_at=utc_now() - timedelta(days=90),
        )
    )

    assert (await service.list_knowledge_gaps()).gaps == []


# ── cache invalidation ───────────────────────────────────────────────────────


async def test_add_entry_clears_the_answer_cache(
    service: SupportService, fake_rag_pipeline: FakeRAGPipeline
) -> None:
    await service.search("hello")
    assert (await service.search("hello")).cached is True

    await service.add_entry("New?", "Yes.", "General")

    assert (await service.search("hello")).cached is False
    assert len(fake_rag_pipeline.retrieve_calls) == 2


async def test_ingest_clears_the_answer_cache(service: SupportService) -> None:
    await service.search("hello")

    await service.ingest(
        IngestBatch(source="x", faqs=[FAQDraft(question="Q?", answer="A", category="c")])
    )

    assert (await service.search("hello")).cached is False


async def test_faq_indexed_event_clears_the_answer_cache(
    service: SupportService, event_bus: EventBus
) -> None:
    await service.search("hello")

    await event_bus.publish(build_faq_indexed_event(str(uuid.uuid4()), "default"))

    assert FAQ_INDEXED == "faq.indexed"
    assert (await service.search("hello")).cached is False


def _config(**values: object) -> object:
    from faq_core.config import AppConfig

    return AppConfig(_env_file=None, **values)  # type: ignore[arg-type]
