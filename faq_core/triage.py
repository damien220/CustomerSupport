"""Escalation triggers and the final outcome of one query (plan §4, Phase 4).

The outcome starts from the confidence level (``faq_core.confidence``) and is
raised to ``escalate`` by any enabled trigger:

====================  =========================================================
reason code           fires when
====================  =========================================================
``no_passages``       nothing was retrieved
``low_confidence``    the confidence score is below the caveat threshold
``repeated_rephrase`` at least ``ESCALATION_REPHRASE_MIN_REPEATS`` of the
                      session's last ``ESCALATION_REPHRASE_WINDOW`` queries are
                      similar (hashing-embedding cosine ≥
                      ``ESCALATION_REPHRASE_SIMILARITY``) to this one
``frustration``       a frustration phrase occurs (``ESCALATION_FRUSTRATION_TERMS``)
``pii_detected``      the query contained personal data (it was redacted)
``legal_topic``       a legal phrase occurs (``ESCALATION_LEGAL_TERMS``)
``medical_topic``     a medical phrase occurs (``ESCALATION_MEDICAL_TERMS``)
``self_check_failed`` the optional LLM self-check rejected the draft answer
====================  =========================================================

Medium confidence adds ``moderate_confidence``; confidence reasons such as
``no_keyword_match`` or ``low_similarity`` explain a non-high score. Trigger
groups are switched on/off with ``ESCALATION_TRIGGERS``. Term lists match whole
words/phrases on case-folded text; the defaults are English — set the
``*_TERMS`` settings to add other languages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from faq_core.confidence import ConfidenceLevel, Outcome
from faq_core.retrieval.embedders import HashingEmbedder
from faq_core.retrieval.ranking import normalize_text

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from faq_core.confidence import ConfidenceAssessment
    from faq_core.pii import PIIKind

TRIGGER_GROUPS: Final = frozenset(
    {"low_confidence", "repeated_rephrase", "frustration", "pii", "sensitive_topic"}
)

DEFAULT_FRUSTRATION_TERMS: Final[tuple[str, ...]] = (
    "ridiculous",
    "useless",
    "frustrated",
    "frustrating",
    "fed up",
    "angry",
    "furious",
    "annoyed",
    "unacceptable",
    "terrible service",
    "worst",
    "waste of time",
    "scam",
    "not helpful",
    "still not working",
    "speak to a human",
    "talk to a human",
    "real person",
    "speak to a manager",
)
DEFAULT_LEGAL_TERMS: Final[tuple[str, ...]] = (
    "lawyer",
    "attorney",
    "solicitor",
    "lawsuit",
    "sue",
    "suing",
    "legal action",
    "court",
    "subpoena",
    "class action",
    "liability",
    "data protection authority",
)
DEFAULT_MEDICAL_TERMS: Final[tuple[str, ...]] = (
    "injury",
    "injured",
    "hospital",
    "doctor",
    "allergic",
    "allergic reaction",
    "poisoning",
    "overdose",
    "medication",
    "bleeding",
    "rash",
    "choking",
    "burned",
    "medical emergency",
)

CAVEAT_TEXT: Final = (
    "The knowledge base only partly matches this question, so the answer may be "
    "incomplete; offer a human agent if it does not resolve the issue."
)
HANDOFF_MESSAGE: Final = (
    "I couldn't find a reliable answer to this in our knowledge base, so I'm handing it "
    "to a human support agent who can help."
)
"""The fixed ``answer`` of an escalated result in api/offline mode (never LLM-written)."""


def parse_terms(raw: str, defaults: Sequence[str]) -> tuple[str, ...]:
    """Comma-separated *raw* setting → normalised phrases; blank keeps *defaults*."""
    items = [normalize_text(part) for part in raw.split(",")] if raw.strip() else list(defaults)
    return tuple(dict.fromkeys(item for item in items if item))


def parse_trigger_groups(raw: str) -> frozenset[str]:
    """``ESCALATION_TRIGGERS`` → enabled groups (``all`` / blank = every group).

    Raises:
        ValueError: an unknown group name.
    """
    names = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not names or names == {"all"}:
        return TRIGGER_GROUPS
    if "none" in names:
        return frozenset()
    unknown = names - TRIGGER_GROUPS
    if unknown:
        raise ValueError(f"unknown ESCALATION_TRIGGERS: {sorted(unknown)}")
    return frozenset(names)


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """The phrases of *terms* that occur in *text* as whole words (case-insensitive)."""
    haystack = f" {normalize_text(text)} "
    return [term for term in terms if re.search(rf"(?<=\s){re.escape(term)}(?=\s)", haystack)]


@dataclass(frozen=True, slots=True)
class TriagePolicy:
    """Which triggers run and how (built from ``AppConfig`` in ``bootstrap``)."""

    enabled: frozenset[str] = TRIGGER_GROUPS
    frustration_terms: tuple[str, ...] = DEFAULT_FRUSTRATION_TERMS
    legal_terms: tuple[str, ...] = DEFAULT_LEGAL_TERMS
    medical_terms: tuple[str, ...] = DEFAULT_MEDICAL_TERMS
    rephrase_similarity: float = 0.5
    rephrase_min_repeats: int = 2
    rephrase_window: int = 5
    _embedder: HashingEmbedder = field(default_factory=HashingEmbedder, repr=False)

    def text_triggers(self, text: str, pii_kinds: Sequence[PIIKind]) -> list[str]:
        """Reason codes raised by the (redacted) query text itself."""
        reasons: list[str] = []
        if "frustration" in self.enabled and find_terms(text, self.frustration_terms):
            reasons.append("frustration")
        if "pii" in self.enabled and pii_kinds:
            reasons.append("pii_detected")
        if "sensitive_topic" in self.enabled:
            if find_terms(text, self.legal_terms):
                reasons.append("legal_topic")
            if find_terms(text, self.medical_terms):
                reasons.append("medical_topic")
        return reasons

    def is_repeated_rephrase(self, text: str, previous: Sequence[str]) -> bool:
        """Whether *text* rephrases enough of the session's *previous* queries."""
        if "repeated_rephrase" not in self.enabled or not previous:
            return False
        recent = previous[-self.rephrase_window :]
        similar = sum(
            1 for earlier in recent if self.similarity(text, earlier) >= self.rephrase_similarity
        )
        return similar >= self.rephrase_min_repeats

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity of two texts' hashing embeddings (lexical, dependency-free)."""
        a, b = self._embedder.embed_sync(left), self._embedder.embed_sync(right)
        return math.fsum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True, slots=True)
class TriageDecision:
    """Final outcome of one query: ``outcome``, ``reasons`` (codes) and ``caveat``."""

    outcome: Outcome
    reasons: tuple[str, ...]
    caveat: str = ""

    @property
    def escalate(self) -> bool:
        return self.outcome is Outcome.ESCALATE


def decide(
    assessment: ConfidenceAssessment, triggers: Sequence[str], policy: TriagePolicy
) -> TriageDecision:
    """Combine the confidence level with the triggers that fired."""
    reasons: list[str] = []
    outcome = assessment.outcome
    if assessment.level is ConfidenceLevel.LOW:
        if "low_confidence" in policy.enabled:
            reasons.append("low_confidence")
        else:
            outcome = Outcome.ANSWER_WITH_CAVEAT
    elif assessment.level is ConfidenceLevel.MEDIUM:
        reasons.append("moderate_confidence")
    reasons.extend(assessment.reasons)
    if triggers:
        outcome = Outcome.ESCALATE
        reasons.extend(triggers)
    caveat = CAVEAT_TEXT if outcome is Outcome.ANSWER_WITH_CAVEAT else ""
    return TriageDecision(outcome, tuple(dict.fromkeys(reasons)), caveat)


def reject_by_self_check(decision: TriageDecision) -> TriageDecision:
    """The LLM self-check rejected the draft answer: escalate instead of answering."""
    return TriageDecision(Outcome.ESCALATE, (*decision.reasons, "self_check_failed"))
