"""Security errors.

Every message here is a fixed, operator-written string: no token, key, claim
value or URL credential ever reaches an exception message (they end up in logs
and HTTP responses).
"""

from __future__ import annotations

from typing import Final


class SecurityConfigError(ValueError):
    """Remote-access security settings are missing, invalid or unsafe.

    Raised at startup so a misconfigured server refuses to run instead of
    running open.
    """


class AuthError(Exception):
    """A request failed authentication (401) or authorization (403).

    Attributes:
        error: RFC 6750 error code (``invalid_token``, ``insufficient_scope``),
            or ``None`` when the request carried no credentials at all (RFC 6750
            §3.1: no error code then).
        description: Safe, human-readable text for the response. For
            ``invalid_token`` it is always :data:`INVALID_TOKEN_DESCRIPTION`,
            so a caller cannot tell a malformed, unknown, expired or
            wrong-audience token apart.
        reason: Short fixed code for the server log only (e.g. ``expired``,
            ``wrong_audience``); never sent to the client.
        status_code: HTTP status for the response.
        required_scopes: Scopes to advertise in the challenge (403 only).
    """

    def __init__(
        self,
        description: str,
        *,
        error: str | None = "invalid_token",
        status_code: int = 401,
        required_scopes: tuple[str, ...] = (),
        reason: str | None = None,
    ) -> None:
        super().__init__(description)
        self.description = description
        self.error = error
        self.status_code = status_code
        self.required_scopes = required_scopes
        self.reason = reason or error or "missing_token"


INVALID_TOKEN_DESCRIPTION: Final = "the access token is invalid or expired"  # noqa: S105


def invalid_token(reason: str) -> AuthError:
    """A generic 401 ``invalid_token`` error; *reason* goes to the log only."""
    return AuthError(INVALID_TOKEN_DESCRIPTION, reason=reason)


class AuthUnavailableError(AuthError):
    """Tokens cannot be verified right now (e.g. the issuer's key set is unreachable).

    Fails closed with 503 instead of 401, so clients retry rather than
    re-authorize.
    """

    def __init__(self, description: str = "token verification is temporarily unavailable") -> None:
        super().__init__(description, error="temporarily_unavailable", status_code=503)
