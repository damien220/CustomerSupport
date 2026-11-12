"""Escalation triggers and the outcome decision (pure functions, no I/O)."""

from __future__ import annotations

import pytest

from faq_core.confidence import ConfidenceScorer, Outcome
from faq_core.pii import PIIKind
from faq_core.triage import (
    CAVEAT_TEXT,
    TriagePolicy,
    decide,
    find_terms,
    parse_terms,
    parse_trigger_groups,
    reject_by_self_check,
)
from tests.conftest import MEDIUM_EVIDENCE, STRONG_EVIDENCE, WEAK_EVIDENCE


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("This is ridiculous, I want my money back", "frustration"),
        ("let me talk to a human", "frustration"),
        ("I will contact my lawyer about this", "legal_topic"),
        ("we are going to sue you", "legal_topic"),
        ("my son had an allergic reaction to the product", "medical_topic"),
        ("I was injured by the blender", "medical_topic"),
    ],
)
def test_text_triggers(text: str, reason: str) -> None:
    assert reason in TriagePolicy().text_triggers(text, [])


def test_terms_match_whole_words_only() -> None:
    assert find_terms("the issue persists", ["sue"]) == []
    assert find_terms("I will SUE!", ["sue"]) == ["sue"]
    assert find_terms("Talk to a  human, please", ["talk to a human"]) == ["talk to a human"]


def test_pii_trigger_and_disabled_groups() -> None:
    assert TriagePolicy().text_triggers("hi", [PIIKind.EMAIL]) == ["pii_detected"]
    quiet = TriagePolicy(enabled=frozenset({"low_confidence"}))
    assert quiet.text_triggers("ridiculous lawyer", [PIIKind.EMAIL]) == []


def test_repeated_rephrase_needs_enough_similar_previous_queries() -> None:
    policy = TriagePolicy()
    previous = ["how do I reset my password", "password reset not working"]

    assert policy.is_repeated_rephrase("reset password not working again", previous)
    assert not policy.is_repeated_rephrase("reset password not working again", previous[:1])
    assert not policy.is_repeated_rephrase("what are your opening hours", previous)


def test_rephrase_only_looks_at_the_window() -> None:
    policy = TriagePolicy(rephrase_window=1)
    previous = ["reset my password", "reset my password please", "shipping times"]

    assert not policy.is_repeated_rephrase("reset password", previous)


def test_parse_settings() -> None:
    assert parse_trigger_groups("") == parse_trigger_groups("all")
    assert parse_trigger_groups("none") == frozenset()
    assert parse_trigger_groups("pii, frustration") == {"pii", "frustration"}
    with pytest.raises(ValueError, match="unknown"):
        parse_trigger_groups("pii,nonsense")
    assert parse_terms("", ("a",)) == ("a",)
    assert parse_terms(" Avocat , Abogado ,", ()) == ("avocat", "abogado")


def test_decide_follows_confidence_and_triggers_force_escalation() -> None:
    scorer, policy = ConfidenceScorer(), TriagePolicy()

    high = decide(scorer.assess(STRONG_EVIDENCE), [], policy)
    medium = decide(scorer.assess(MEDIUM_EVIDENCE), [], policy)
    low = decide(scorer.assess(WEAK_EVIDENCE), [], policy)
    frustrated = decide(scorer.assess(STRONG_EVIDENCE), ["frustration"], policy)

    assert (high.outcome, high.reasons, high.caveat) == (Outcome.ANSWER, (), "")
    assert medium.outcome is Outcome.ANSWER_WITH_CAVEAT and medium.caveat == CAVEAT_TEXT
    assert medium.reasons[0] == "moderate_confidence"
    assert low.escalate and low.reasons[0] == "low_confidence"
    assert frustrated.escalate and frustrated.reasons == ("frustration",)
    assert reject_by_self_check(high).reasons == ("self_check_failed",)


def test_low_confidence_trigger_can_be_disabled() -> None:
    policy = TriagePolicy(enabled=frozenset({"pii"}))

    decision = decide(ConfidenceScorer().assess(WEAK_EVIDENCE), [], policy)

    assert decision.outcome is Outcome.ANSWER_WITH_CAVEAT
