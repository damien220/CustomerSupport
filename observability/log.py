"""Stdlib logging setup with secret/PII redaction.

Use one ``_log = get_logger(__name__)`` per module and pass structured context
through ``extra={...}``; the formatter appends extras as ``key=value`` pairs.
:class:`RedactingFilter` scrubs the rendered message *and* string extras of:

* credential-shaped tokens (provider API keys, bearer tokens, JWTs, Langfuse
  keys, ``?access_token=`` / ``?api_key=`` style query parameters),
* the exact values of every secret registered via :func:`configure_logging`
  (``AppConfig`` ``SecretStr`` fields),
* e-mail addresses and payment-card-like digit runs (basic PII), then every
  other kind :mod:`faq_core.pii` detects (phone, IBAN, SSN, IP address,
  more credential shapes) as typed placeholders such as ``[PHONE]``.

Handlers write to stderr, which keeps the MCP stdio transport's stdout clean.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import Final

from faq_core.pii import redact_pii

REDACTED: Final = "[REDACTED]"

# Patterns are assembled so this file never contains a literal key prefix that
# the repository's secrets grep would (rightly) flag.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-(?:ant|proj)-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9\-]{8,}"),
    re.compile(r"\bAI" + r"za[0-9A-Za-z_\-]{30,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    # A JWT anywhere (header.payload.signature, base64url): OAuth access tokens.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*"),
    # Credentials in a URL query string (e.g. an uvicorn access-log line).
    re.compile(
        r"(?i)(?<=[?&;])(?:access_token|id_token|refresh_token|token|api_key|apikey|key)"
        r"=[^&\s\"']+"
    ),
)
_PII_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"),
    re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"),
)
_MIN_SECRET_LEN: Final = 6

_STANDARD_RECORD_ATTRS: Final = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
    | {"message", "asctime", "color_message"}  # uvicorn's ANSI copy of the message
)


class RedactingFilter(logging.Filter):
    """Replace secrets and PII in log records before any handler formats them."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets: set[str] = set()
        self.add_secrets(secrets)

    def add_secrets(self, secrets: Iterable[str]) -> None:
        # Very short values would redact ordinary words; config secrets are long.
        self._secrets.update(s for s in secrets if len(s) >= _MIN_SECRET_LEN)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for pattern in (*_SECRET_PATTERNS, *_PII_PATTERNS):
            text = pattern.sub(REDACTED, text)
        return redact_pii(text).text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redact(record.getMessage())
        record.args = None
        for key, value in list(vars(record).items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, self.redact(value))
        return True


class KeyValueFormatter(logging.Formatter):
    """``time level logger message key=value ...`` with extras appended."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in vars(record).items()
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return base
        return base + " " + " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))


_redacting_filter = RedactingFilter()
_handler: logging.Handler | None = None


def configure_logging(level: str = "INFO", *, secrets: Iterable[str] = ()) -> None:
    """Make the redacting stderr handler the root logger's only handler (idempotent).

    Any handler already on the root logger is removed: libraries such as
    FastMCP install their own console handler at construction time, which would
    duplicate every line *and* bypass redaction.

    Args:
        level: Root log level name, e.g. ``"INFO"``.
        secrets: Raw secret values to scrub verbatim from every record.
    """
    global _handler

    _redacting_filter.add_secrets(secrets)
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _handler is None:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(KeyValueFormatter())
        _handler.addFilter(_redacting_filter)
    for existing in list(root.handlers):
        if existing is not _handler:
            root.removeHandler(existing)
    if _handler not in root.handlers:
        root.addHandler(_handler)


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for *name* (use ``__name__``)."""
    return logging.getLogger(name)
