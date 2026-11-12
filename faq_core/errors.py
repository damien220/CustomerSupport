"""Domain exceptions raised by :class:`faq_core.service.SupportService`.

Every error carries a stable machine-readable ``code`` and an HTTP-equivalent
``status_code`` so each surface can map it without knowing the concrete class:
the REST adapter turns it into a JSON error body with that status, and the MCP
adapter turns it into a structured tool error. Messages must never contain
secrets (they are returned to callers verbatim).
"""

from __future__ import annotations

from typing import Any, ClassVar


class SupportError(Exception):
    """Base class for all expected (non-bug) failures of the support core."""

    code: ClassVar[str] = "internal_error"
    status_code: ClassVar[int] = 500

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Return the stable error envelope shared by every surface."""
        return {"error": {"code": self.code, "message": self.message, **self.details}}


class NotFoundError(SupportError):
    """A requested resource does not exist."""

    code = "not_found"
    status_code = 404


class FAQNotFoundError(NotFoundError):
    """No FAQ entry exists for the requested id."""

    code = "faq_not_found"

    def __init__(self, faq_id: str) -> None:
        super().__init__(f"FAQ entry {faq_id!r} not found", faq_id=faq_id)


class QueryNotFoundError(NotFoundError):
    """No logged query exists for the requested ``query_id``."""

    code = "query_not_found"

    def __init__(self, query_id: str) -> None:
        super().__init__(f"query {query_id!r} not found", query_id=query_id)


class EscalationNotFoundError(NotFoundError):
    """No escalation exists for the requested id."""

    code = "escalation_not_found"

    def __init__(self, escalation_id: str) -> None:
        super().__init__(f"escalation {escalation_id!r} not found", escalation_id=escalation_id)


class InvalidRequestError(SupportError):
    """Caller-supplied input failed a business-rule check."""

    code = "invalid_request"
    status_code = 422


class IngestError(SupportError):
    """An ingest source could not be read, fetched or parsed."""

    code = "ingest_failed"
    status_code = 422


class AnswerGenerationUnavailableError(SupportError):
    """This server does not generate answers in its ``LLM_MODE`` (``client``).

    In ``client`` mode the calling agent's own model writes the reply from the
    passages returned by ``search_kb``.
    """

    code = "answer_generation_unavailable"
    status_code = 501

    def __init__(self, llm_mode: str) -> None:
        super().__init__(
            f"answer generation is not available in LLM_MODE={llm_mode}: "
            "call search_kb and write the reply from its passages",
            llm_mode=llm_mode,
            use_instead="search_kb",
        )


class LLMUnavailableError(SupportError):
    """The configured LLM provider failed, refused, or its circuit breaker is open.

    The message names only the provider and the failure kind (an HTTP status,
    ``refusal``, ...) — never request headers or keys.
    """

    code = "llm_unavailable"
    status_code = 502
