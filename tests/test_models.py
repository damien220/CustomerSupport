from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from faq_core.models import (
    AnswerResponse,
    Chunk,
    Citation,
    FAQEntry,
    FeedbackEntry,
    QueryRequest,
    SearchResult,
)


def test_faq_entry_generates_id_timestamp_and_default_kb() -> None:
    entry = FAQEntry(
        question="How do I reset my password?", answer="Click reset.", category="account"
    )

    assert isinstance(entry.id, uuid.UUID)
    assert entry.created_at.tzinfo is not None
    assert entry.kb_id == "default"


@pytest.mark.parametrize("field", ["question", "answer", "category"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_faq_entry_rejects_blank_field(field: str, blank: str) -> None:
    kwargs = {"question": "Q", "answer": "A", "category": "billing", field: blank}

    with pytest.raises(ValidationError):
        FAQEntry.model_validate(kwargs)


def test_faq_entry_preserves_content_as_entered() -> None:
    entry = FAQEntry(question="  Q?  ", answer="A", category="c")

    assert entry.question == "  Q?  "


def test_faq_entry_accepts_uuid_string_and_rejects_naive_timestamp() -> None:
    entry_id = uuid.uuid4()

    assert (
        FAQEntry.model_validate(
            {"id": str(entry_id), "question": "Q", "answer": "A", "category": "c"}
        ).id
        == entry_id
    )
    with pytest.raises(ValidationError):
        FAQEntry(question="Q", answer="A", category="c", created_at=datetime(2026, 1, 1))


def test_faq_entry_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FAQEntry.model_validate({"question": "Q", "answer": "A", "category": "c", "x": 1})


def test_faq_entry_json_round_trip() -> None:
    entry = FAQEntry(question="Q", answer="A", category="c")

    assert FAQEntry.model_validate_json(entry.model_dump_json()) == entry


def test_query_request_defaults_top_k_to_three() -> None:
    assert QueryRequest(query="How do refunds work?").top_k == 3


@pytest.mark.parametrize("top_k", [0, 21])
def test_query_request_bounds_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="q", top_k=top_k)


def test_answer_response_defaults_confidence_to_zero() -> None:
    assert AnswerResponse(query="q", answer="a", sources=[]).confidence == 0.0


def test_search_result_contract_fields_are_stable() -> None:
    result = SearchResult(query="q", answer="a", confidence=0.5, sources=[])

    assert set(result.model_dump()) == {
        "query_id",
        "query",
        "answer",
        "confidence",
        "sources",
        "cached",
        "citations",  # additive in Phase 2
        "generated",  # additive in Phase 3
        "outcome",  # additive in Phase 4 ↓
        "escalate",
        "reasons",
        "caveat",
        "confidence_detail",
        "pii_detected",
        "session_id",
    }
    assert result.citations == []
    assert result.generated is True


def test_citation_contract_fields_are_stable() -> None:
    assert set(Citation.model_fields) == {
        "id",
        "source_type",
        "source_id",
        "title",
        "heading",
        "snippet",
        "source_uri",
        "updated_at",
        "score",
    }


def test_chunk_cannot_belong_to_both_an_entry_and_a_document() -> None:
    with pytest.raises(ValidationError, match="not both"):
        Chunk(faq_id=uuid.uuid4(), document_id=uuid.uuid4(), content="x")


def test_feedback_entry_requires_query_id() -> None:
    with pytest.raises(ValidationError):
        FeedbackEntry.model_validate({"rating": 5})


@pytest.mark.parametrize("rating", [0, 6])
def test_feedback_entry_bounds_rating(rating: int) -> None:
    with pytest.raises(ValidationError):
        FeedbackEntry(query_id=uuid.uuid4(), rating=rating)


def test_feedback_entry_accepts_valid_rating() -> None:
    feedback = FeedbackEntry(query_id=uuid.uuid4(), rating=4, comment="Helpful")

    assert feedback.helpful is True
    assert feedback.rating == 4
