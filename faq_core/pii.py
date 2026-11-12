"""Dependency-free PII detection and redaction for customer text.

Applied to what *customers* type (queries, feedback comments, escalation
transcripts) before it is logged, stored, traced or sent to an LLM — never to
knowledge-base content, whose support e-mail addresses and phone numbers are
part of the answers.

Detectors run in a fixed priority order and each replaces its matches with a
placeholder before the next one runs, so spans never overlap and a later
detector cannot match inside a placeholder (placeholders hold no digits):

=============  ===============  ================================================
kind           placeholder      rule
=============  ===============  ================================================
secret         ``[SECRET]``     API-key / bearer-token / JWT / ``key=...`` shapes
email          ``[EMAIL]``      ``local@domain.tld``
iban           ``[IBAN]``       country code + check digits, ISO 13616 mod-97
credit_card    ``[CREDIT_CARD]`` 13-19 digits (spaces/dashes allowed), Luhn-valid
ssn            ``[SSN]``        US ``AAA-GG-SSSS`` with separators, SSA rules
ip_address     ``[IP_ADDRESS]`` IPv4 / IPv6 literals validated by :mod:`ipaddress`
phone          ``[PHONE]``      7-15 digits written like a phone number (not dates)
=============  ===============  ================================================

Redaction is irreversible by design: no mapping back to the original values
is kept anywhere (plan §4: redact before logging and before the LLM).
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class PIIKind(StrEnum):
    """A category of personal or secret data."""

    SECRET = "secret"  # noqa: S105 — a category name, not a credential
    EMAIL = "email"
    IBAN = "iban"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"
    PHONE = "phone"


PLACEHOLDERS: Final[dict[PIIKind, str]] = {kind: f"[{kind.value.upper()}]" for kind in PIIKind}
"""Replacement text per kind, e.g. ``[EMAIL]``."""

# Credential shapes. Literal key prefixes are split so this file never trips the
# repository's secrets grep.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-(?:ant|proj)-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9\-]{8,}"),
    re.compile(r"\bAI" + r"za[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    re.compile(
        r"(?i)\b(?:api[_\- ]?key|access[_\- ]?token|auth[_\- ]?token|secret|password|passwd)"
        r"\s*[:=]\s*\S{6,}"
    ),
)
_EMAIL: Final = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_IBAN: Final = re.compile(
    r"\b[A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]{4}){2,7}(?:[ ]?[A-Za-z0-9]{1,3})?\b"
)
_CARD: Final = re.compile(r"(?<![\d\-])\d(?:[ \-]?\d){12,18}(?![\d\-])")
_SSN: Final = re.compile(
    r"(?<![\d\-])(?!000|666|9\d\d)\d{3}([\- ])(?!00)\d{2}\1(?!0000)\d{4}(?![\d\-])"
)
_IPV4: Final = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)")
_IPV6: Final = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_PHONE: Final = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{1,4}\)[\s.\-]?)?\d{1,4}(?:[\s.\-]?\d{2,4}){1,4}(?!\w)"
)
_DATE: Final = re.compile(r"^\d{4}[\-/.]\d{1,2}[\-/.]\d{1,2}$|^\d{1,2}[\-/.]\d{1,2}[\-/.]\d{2,4}$")
_PHONE_CONTEXT: Final = re.compile(r"(?i)\b(?:phone|call|tel|mobile|cell|whatsapp|text me|sms)\b")
_MIN_PHONE_DIGITS: Final = 7
_MAX_PHONE_DIGITS: Final = 15
_PHONE_CONTEXT_WINDOW: Final = 24


def luhn_valid(digits: str) -> bool:
    """Whether *digits* (only ``0-9``) passes the Luhn checksum."""
    if not digits.isdigit():
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:  # noqa: PLR2004 — Luhn digit fold
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    """Whether *candidate* is a structurally valid IBAN (ISO 13616 mod-97 == 1)."""
    iban = candidate.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34 or not iban[:2].isalpha() or not iban[2:4].isdigit():  # noqa: PLR2004
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        numeric = "".join(str(int(char, 36)) for char in rearranged)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def _digits(text: str) -> str:
    return "".join(char for char in text if char.isdigit())


def _card_ok(match: re.Match[str], _text: str) -> bool:
    digits = _digits(match.group())
    return 13 <= len(digits) <= 19 and len(set(digits)) > 1 and luhn_valid(digits)  # noqa: PLR2004


def _iban_ok(match: re.Match[str], _text: str) -> bool:
    return iban_valid(match.group())


def _always(_match: re.Match[str], _text: str) -> bool:
    return True


def _ipv4_ok(match: re.Match[str], _text: str) -> bool:
    try:
        ipaddress.IPv4Address(match.group())
    except ValueError:
        return False
    return True


def _ipv6_ok(match: re.Match[str], _text: str) -> bool:
    candidate = match.group()
    if sum(char.isalnum() for char in candidate) < 3:  # noqa: PLR2004 — not a bare "::"
        return False
    try:
        ipaddress.IPv6Address(candidate)
    except ValueError:
        return False
    return True


def _phone_ok(match: re.Match[str], text: str) -> bool:
    candidate = match.group().strip()
    digits = _digits(candidate)
    if not _MIN_PHONE_DIGITS <= len(digits) <= _MAX_PHONE_DIGITS or _DATE.match(candidate):
        return False
    if candidate.startswith("+") or "(" in candidate:
        return True
    if any(sep in candidate for sep in " .-"):
        return True
    # A bare digit run is only a phone number when the text says so nearby
    # (otherwise order numbers and amounts would be redacted).
    window = text[max(0, match.start() - _PHONE_CONTEXT_WINDOW) : match.start()]
    return bool(_PHONE_CONTEXT.search(window))


type _Validator = Callable[[re.Match[str], str], bool]

_DETECTORS: Final[tuple[tuple[PIIKind, re.Pattern[str], _Validator], ...]] = (
    *((PIIKind.SECRET, pattern, _always) for pattern in _SECRET_PATTERNS),
    (PIIKind.EMAIL, _EMAIL, _always),
    (PIIKind.IBAN, _IBAN, _iban_ok),
    (PIIKind.CREDIT_CARD, _CARD, _card_ok),
    (PIIKind.SSN, _SSN, _always),
    (PIIKind.IP_ADDRESS, _IPV4, _ipv4_ok),
    (PIIKind.IP_ADDRESS, _IPV6, _ipv6_ok),
    (PIIKind.PHONE, _PHONE, _phone_ok),
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Outcome of redacting one text.

    ``text`` is the redacted text (or the original when redaction is off),
    ``kinds`` the sorted, distinct kinds found and ``count`` the number of
    redacted spans.
    """

    text: str
    kinds: tuple[PIIKind, ...] = ()
    count: int = 0

    @property
    def found(self) -> bool:
        return self.count > 0


def redact_pii(text: str) -> RedactionResult:
    """Replace every detected PII span in *text* with its placeholder."""
    kinds: set[PIIKind] = set()
    count = 0
    for kind, pattern, is_valid in _DETECTORS:
        current = text

        def _replace(
            match: re.Match[str],
            kind: PIIKind = kind,
            is_valid: _Validator = is_valid,
            current: str = current,
        ) -> str:
            nonlocal count
            if not is_valid(match, current):
                return match.group()
            kinds.add(kind)
            count += 1
            return PLACEHOLDERS[kind]

        text = pattern.sub(_replace, text)
    return RedactionResult(text=text, kinds=tuple(sorted(kinds)), count=count)


def strip_placeholders(text: str) -> str:
    """Remove redaction placeholders (for retrieval, where ``[EMAIL]`` would match "email")."""
    for placeholder in PLACEHOLDERS.values():
        text = text.replace(placeholder, " ")
    return " ".join(text.split())


class PIIRedactor:
    """Configured redaction (``PII_REDACTION=on|off``).

    When disabled, :meth:`redact` still *detects* PII (the escalation trigger
    needs it) but returns the text unchanged.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def redact(self, text: str) -> RedactionResult:
        result = redact_pii(text)
        if self._enabled:
            return result
        return RedactionResult(text=text, kinds=result.kinds, count=result.count)
