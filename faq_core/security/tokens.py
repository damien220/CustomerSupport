"""Principals, the token-verifier interface and static bearer keys.

A :class:`TokenVerifier` turns the raw bearer token of one request into a
:class:`Principal` or raises :class:`~faq_core.security.errors.AuthError`. The
token itself is never stored on the principal, so nothing downstream can
forward it (no token passthrough).

Static keys (``AUTH_MODE=token``) are configured as ``name:token`` pairs; only
the SHA-256 digest of each token is kept, and a presented token is compared
against every digest with :func:`hmac.compare_digest` (constant time, no early
exit), so timing reveals neither the token nor which key matched.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol

from faq_core.security.errors import SecurityConfigError, invalid_token

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_STATIC_TOKEN_LENGTH: Final = 32
MAX_TOKEN_LENGTH: Final = 8192
_KEY_NAME: Final = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
# RFC 6750 b64token: the only characters a bearer token may contain.
_B64TOKEN: Final = re.compile(r"^[A-Za-z0-9\-._~+/]+=*$")

AuthMethod = Literal["token", "oauth", "none"]


@dataclass(frozen=True, slots=True)
class Principal:
    """Who made an authenticated request.

    Attributes:
        key_id: Stable identifier for rate limits and logs, e.g. ``key:crm`` or
            ``oauth:<subject>``; never a secret.
        method: How the caller authenticated.
        client_id: The OAuth client (or the static key id).
        subject: The resource owner (``sub``) for OAuth tokens.
        issuer: The token issuer for OAuth tokens.
        scopes: Granted scopes.
    """

    key_id: str
    method: AuthMethod
    client_id: str
    subject: str | None = None
    issuer: str | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)


class TokenVerifier(Protocol):
    """Verifies one bearer token."""

    async def verify(self, token: str) -> Principal:
        """Return the principal for *token*.

        Raises:
            AuthError: the token is invalid, expired, for another audience or
                lacks a required scope.
        """
        ...


def is_well_formed_token(token: str) -> bool:
    """Whether *token* is a syntactically valid, reasonably sized bearer token."""
    return 0 < len(token) <= MAX_TOKEN_LENGTH and _B64TOKEN.fullmatch(token) is not None


@dataclass(frozen=True, slots=True)
class ApiKey:
    """One configured static key: its public name and its token's SHA-256 digest."""

    name: str
    digest: bytes


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def parse_api_keys(raw: str) -> list[ApiKey]:
    """Parse ``API_KEYS`` (comma-separated ``name:token`` pairs).

    Raises:
        SecurityConfigError: a malformed pair, a bad name, a short or
            malformed token, or a duplicate name/token. Messages name the
            entry position or key name, never the token.
    """
    keys: list[ApiKey] = []
    names: set[str] = set()
    digests: set[bytes] = set()
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    for position, entry in enumerate(entries, 1):
        name, separator, token = entry.partition(":")
        name, token = name.strip(), token.strip()
        if not separator or not name or not token:
            raise SecurityConfigError(f"API_KEYS entry {position} must look like name:token")
        if not _KEY_NAME.fullmatch(name):
            raise SecurityConfigError(
                f"API_KEYS entry {position}: the name may use letters, digits, '_', '.', '-' "
                "(1-64 characters)"
            )
        if len(token) < MIN_STATIC_TOKEN_LENGTH or not is_well_formed_token(token):
            raise SecurityConfigError(
                f"API_KEYS key {name!r}: the token must be at least {MIN_STATIC_TOKEN_LENGTH} "
                'URL-safe characters (python -c "import secrets; print(secrets.token_urlsafe(32))")'
            )
        digest = _digest(token)
        if name in names or digest in digests:
            raise SecurityConfigError(f"API_KEYS key {name!r} is a duplicate name or token")
        names.add(name)
        digests.add(digest)
        keys.append(ApiKey(name=name, digest=digest))
    return keys


class StaticTokenVerifier:
    """Accepts exactly the configured static keys."""

    def __init__(self, keys: Sequence[ApiKey]) -> None:
        if not keys:
            raise SecurityConfigError("static token verification needs at least one API key")
        self._keys = tuple(keys)

    def matches(self, token: str) -> ApiKey | None:
        """The key whose token is *token*, compared in constant time against every key."""
        presented = _digest(token)
        match: ApiKey | None = None
        for key in self._keys:
            if hmac.compare_digest(key.digest, presented):
                match = key
        return match

    async def verify(self, token: str) -> Principal:
        key = self.matches(token)
        if key is None:
            raise invalid_token("unknown_static_key")
        return Principal(key_id=f"key:{key.name}", method="token", client_id=f"key:{key.name}")


class ChainedVerifier:
    """``AUTH_MODE=both``: a static key first, otherwise an OAuth access token."""

    def __init__(self, static: StaticTokenVerifier, oauth: TokenVerifier) -> None:
        self._static = static
        self._oauth = oauth

    async def verify(self, token: str) -> Principal:
        if self._static.matches(token) is not None:
            return await self._static.verify(token)
        return await self._oauth.verify(token)
