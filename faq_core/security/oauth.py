"""OAuth 2.1 resource-server mode: JWT access-token verification.

This server is only the *resource server* (MCP authorization spec, revision
2026-07-28): an external authorization server runs the authorization-code +
PKCE flow and issues JWT access tokens; this module verifies them.

A token is accepted only when all of these hold (anything else fails closed):

* it is a well-formed JWS of at most 8 KiB;
* its header ``alg`` is in the configured allowlist, which may contain only
  asymmetric algorithms — ``none`` and ``HS*`` are refused, so a public key can
  never be used as an HMAC secret (alg confusion);
* the signing key comes from the configured JWKS, matched by ``kid``, and its
  key type / curve fits ``alg``;
* ``exp``, ``iss`` and ``aud`` are present; ``iss`` equals ``OAUTH_ISSUER``;
  ``aud`` contains this server's canonical resource URI (RFC 8707) — a token
  minted for any other resource is refused, which is what stops tokens from
  being passed through to or from other services;
* every scope in ``OAUTH_REQUIRED_SCOPES`` is granted (``scope`` or ``scp``),
  else 403 ``insufficient_scope``.

Token introspection (RFC 7662) is not implemented: opaque access tokens are not
supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import jwt

from faq_core.security.errors import AuthError, invalid_token
from faq_core.security.tokens import Principal, is_well_formed_token

if TYPE_CHECKING:
    from faq_core.security.jwks import JWKSCache

ALLOWED_JWT_ALGORITHMS: Final = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"}
)
_REQUIRED_CLAIMS: Final = ["exp", "iss", "aud"]
_SAFE_ID: Final = re.compile(r"[^A-Za-z0-9_.@:\-]")
_MAX_ID_LENGTH: Final = 128


@dataclass(frozen=True, slots=True)
class OAuthSettings:
    """Validated resource-server settings (built by the security factory)."""

    issuer: str
    resource: str
    algorithms: frozenset[str]
    required_scopes: tuple[str, ...] = ()
    leeway: int = 60

    @property
    def audiences(self) -> list[str]:
        """The resource URI with and without a trailing slash (RFC 8707 spelling drift)."""
        base = self.resource.rstrip("/")
        return [base, base + "/"]


class JWTVerifier:
    """Verifies JWT access tokens issued for this resource."""

    def __init__(self, settings: OAuthSettings, keys: JWKSCache) -> None:
        self._settings = settings
        self._keys = keys

    async def verify(self, token: str) -> Principal:
        if not is_well_formed_token(token) or token.count(".") != 2:
            raise invalid_token("malformed_jwt")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as err:
            raise invalid_token("malformed_jwt") from err
        alg = header.get("alg")
        if not isinstance(alg, str) or alg not in self._settings.algorithms:
            raise invalid_token("algorithm_not_allowed")
        if "crit" in header:
            # RFC 7515 §4.1.11: unknown critical extensions must be rejected; none are supported.
            raise invalid_token("unsupported_crit_header")
        kid = header.get("kid")
        if kid is not None and not isinstance(kid, str):
            raise invalid_token("malformed_jwt")
        key = await self._keys.signing_key(kid, alg)
        claims = self._decode(token, key, alg)
        return self._principal(claims)

    def _decode(self, token: str, key: Any, alg: str) -> dict[str, Any]:  # noqa: ANN401
        try:
            return jwt.decode(
                token,
                key=key,
                algorithms=[alg],
                audience=self._settings.audiences,
                issuer=self._settings.issuer,
                leeway=self._settings.leeway,
                options={"require": _REQUIRED_CLAIMS, "strict_aud": False},
            )
        except jwt.ExpiredSignatureError as err:
            raise invalid_token("expired") from err
        except jwt.InvalidAudienceError as err:
            raise invalid_token("wrong_audience") from err
        except jwt.InvalidIssuerError as err:
            raise invalid_token("wrong_issuer") from err
        except jwt.InvalidSignatureError as err:
            raise invalid_token("bad_signature") from err
        except jwt.MissingRequiredClaimError as err:
            raise invalid_token("missing_claim") from err
        except jwt.PyJWTError as err:
            raise invalid_token("invalid_jwt") from err

    def _principal(self, claims: dict[str, Any]) -> Principal:
        scopes = _scopes(claims)
        missing = [scope for scope in self._settings.required_scopes if scope not in scopes]
        if missing:
            raise AuthError(
                "access token lacks a required scope",
                error="insufficient_scope",
                status_code=403,
                required_scopes=self._settings.required_scopes,
                reason="missing_scope",
            )
        subject = _claim_str(claims, "sub")
        client_id = _claim_str(claims, "client_id") or _claim_str(claims, "azp") or ""
        identity = _safe_id(subject or client_id or "anonymous")
        return Principal(
            key_id=f"oauth:{identity}",
            method="oauth",
            client_id=client_id,
            subject=subject,
            issuer=self._settings.issuer,
            scopes=scopes,
        )


def _scopes(claims: dict[str, Any]) -> frozenset[str]:
    scope = claims.get("scope")
    if isinstance(scope, str):
        return frozenset(scope.split())
    scp = claims.get("scp")
    if isinstance(scp, list):
        return frozenset(item for item in scp if isinstance(item, str))
    if isinstance(scp, str):
        return frozenset(scp.split())
    return frozenset()


def _claim_str(claims: dict[str, Any], name: str) -> str | None:
    value = claims.get(name)
    return value if isinstance(value, str) and value else None


def _safe_id(value: str) -> str:
    """A log- and key-safe rendering of a subject / client id."""
    return _SAFE_ID.sub("_", value)[:_MAX_ID_LENGTH]
