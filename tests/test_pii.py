"""PII detectors: positives, checksum-based negatives and common false-positive shapes.

Credential-shaped literals are assembled at runtime so this file never trips the
repository's secrets grep.
"""

from __future__ import annotations

import pytest

from faq_core.pii import (
    PIIKind,
    PIIRedactor,
    iban_valid,
    luhn_valid,
    redact_pii,
    strip_placeholders,
)

_PROVIDER_KEY = "sk-" + "ant-" + "api03-abcdefghijklmnop"
_BEARER = "Bear" + "er " + "abcdefghijklmnopqrstuvwxyz012345"
_GOOGLE_KEY = "AI" + "za" + "SyA1234567890abcdefghijklmnopqrstuv"


@pytest.mark.parametrize(
    ("text", "kind", "placeholder"),
    [
        ("mail me at jane.doe@example.co.uk please", PIIKind.EMAIL, "[EMAIL]"),
        ("card 4111 1111 1111 1111 was charged", PIIKind.CREDIT_CARD, "[CREDIT_CARD]"),
        ("card 5500-0000-0000-0004", PIIKind.CREDIT_CARD, "[CREDIT_CARD]"),
        ("IBAN GB82 WEST 1234 5698 7654 32 please refund", PIIKind.IBAN, "[IBAN]"),
        ("iban de89370400440532013000", PIIKind.IBAN, "[IBAN]"),
        ("my SSN is 123-45-6789", PIIKind.SSN, "[SSN]"),
        ("my IP is 192.168.1.20.", PIIKind.IP_ADDRESS, "[IP_ADDRESS]"),
        ("from 2001:db8::8a2e:370:7334 today", PIIKind.IP_ADDRESS, "[IP_ADDRESS]"),
        ("call me at +1 (555) 123-4567", PIIKind.PHONE, "[PHONE]"),
        ("my number is 555-0100", PIIKind.PHONE, "[PHONE]"),
        ("phone 5551234567", PIIKind.PHONE, "[PHONE]"),
        (f"my key is {_PROVIDER_KEY}", PIIKind.SECRET, "[SECRET]"),
        (f"header {_BEARER}", PIIKind.SECRET, "[SECRET]"),
        (f"google {_GOOGLE_KEY}", PIIKind.SECRET, "[SECRET]"),
        ("password: hunter2-long", PIIKind.SECRET, "[SECRET]"),
    ],
)
def test_detects_and_redacts_each_kind(text: str, kind: PIIKind, placeholder: str) -> None:
    result = redact_pii(text)

    assert kind in result.kinds
    assert placeholder in result.text
    assert result.found


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1112 failed",  # Luhn-invalid
        "IBAN GB82 WEST 1234 5698 7654 33",  # mod-97 invalid
        "order 12345678 has not shipped",  # bare digits without phone context
        "delivered on 2026-10-30 at 10:30:00",  # date and time
        "shipping takes 5-7 business days and costs $50",
        "upgrade to version 1.2.3",
        "ip 999.1.1.1 is not an address",
        "ssn 000-12-3456 is never issued",  # SSA rule (still caught as a phone shape)
        "How do I reset my password?",
    ],
)
def test_false_positive_shapes_are_left_alone(text: str) -> None:
    result = redact_pii(text)

    assert PIIKind.CREDIT_CARD not in result.kinds
    assert PIIKind.IBAN not in result.kinds
    assert PIIKind.SSN not in result.kinds
    if not text.lower().startswith(("ssn", "iban")):  # digit groups there look like phones
        assert result.text == text


def test_checksums() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB00 WEST 1234 5698 7654 32")


def test_multiple_kinds_are_counted_and_listed_sorted() -> None:
    result = redact_pii("a@b.io and c@d.io, card 4111111111111111")

    assert result.count == 3
    assert result.kinds == (PIIKind.CREDIT_CARD, PIIKind.EMAIL)
    assert result.text == "[EMAIL] and [EMAIL], card [CREDIT_CARD]"


def test_disabled_redactor_keeps_text_but_still_detects() -> None:
    result = PIIRedactor(enabled=False).redact("mail a@b.io")

    assert result.text == "mail a@b.io"
    assert result.kinds == (PIIKind.EMAIL,)


def test_strip_placeholders_for_retrieval() -> None:
    assert strip_placeholders("reset password for [EMAIL] now") == "reset password for now"
