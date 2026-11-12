"""Redacting log filter. Secret-shaped literals are assembled at runtime so this file
never trips the repository's secrets grep."""

from __future__ import annotations

import logging

import pytest

from observability.log import REDACTED, KeyValueFormatter, RedactingFilter, configure_logging

_ANTHROPIC_LIKE = "sk-" + "ant-" + "api03-abcdefghijklmnop"
_BEARER_LIKE = "Bear" + "er " + "eyJhbGciOiJIUzI1NiJ9.payload.sig"
_GOOGLE_LIKE = "AI" + "za" + "SyA1234567890abcdefghijklmnopqrstuv"


def _record(msg: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, msg, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


@pytest.mark.parametrize(
    "secret",
    [_ANTHROPIC_LIKE, _BEARER_LIKE, _GOOGLE_LIKE, "jane.doe@example.com", "4111 1111 1111 1111"],
)
def test_known_secret_and_pii_shapes_are_redacted(secret: str) -> None:
    record = _record("calling upstream with %s now", secret)

    RedactingFilter().filter(record)

    assert secret not in record.getMessage()
    assert REDACTED in record.getMessage()


def test_registered_secret_values_are_redacted_verbatim() -> None:
    record = _record("password is hunter2-long-value")

    RedactingFilter(secrets=["hunter2-long-value"]).filter(record)

    assert record.getMessage() == f"password is {REDACTED}"


def test_short_registered_values_are_ignored_to_avoid_over_redaction() -> None:
    record = _record("abc stays readable")

    RedactingFilter(secrets=["abc"]).filter(record)

    assert record.getMessage() == "abc stays readable"


def test_string_extras_are_redacted_too() -> None:
    record = _record("request failed", header=_BEARER_LIKE, attempts=3)

    RedactingFilter().filter(record)

    assert record.header == REDACTED  # type: ignore[attr-defined]
    assert record.attempts == 3  # type: ignore[attr-defined]


def test_formatter_appends_extras_as_key_values() -> None:
    record = _record("seeded", kb_id="default", entries=5)

    line = KeyValueFormatter().format(record)

    assert line.endswith("seeded entries=5 kb_id='default'")


def test_configure_logging_owns_the_root_handlers() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    foreign = logging.StreamHandler()  # e.g. FastMCP's basicConfig console handler
    root.addHandler(foreign)
    try:
        configure_logging("WARNING", secrets=["registered-secret-value"])
        configure_logging("WARNING")  # idempotent

        [handler] = root.handlers
        assert foreign not in root.handlers
        assert any(isinstance(f, RedactingFilter) for f in handler.filters)
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_phone_iban_and_ip_are_redacted_with_typed_placeholders() -> None:
    record = _record("customer wrote %s", "call +1 (555) 123-4567 from 10.1.2.3")

    RedactingFilter().filter(record)

    assert record.getMessage() == "customer wrote call [PHONE] from [IP_ADDRESS]"


def test_jwts_and_query_string_tokens_are_redacted() -> None:
    jwt_like = "ey" + "JhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJlLXZhbHVl"
    line = f"GET /mcp?access_token=abcdef123456&x=1 HTTP/1.1 token {jwt_like}"
    redacting = RedactingFilter()

    redacted = redacting.redact(line)

    assert jwt_like not in redacted
    assert "abcdef123456" not in redacted
    assert "x=1" in redacted


def test_configured_static_api_keys_are_redacted_individually() -> None:
    import secrets

    from faq_core.config import AppConfig

    first, second = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    config = AppConfig(_env_file=None, API_KEYS=f"crm:{first},bot:{second}")  # type: ignore[call-arg, arg-type]
    redacting = RedactingFilter(config.secret_values())

    redacted = redacting.redact(f"saw {first} and {second}")

    assert first not in redacted and second not in redacted
