"""The authorization server's signing keys (JWKS), fetched and cached.

The JWKS URL comes from configuration only (``OAUTH_JWKS_URL``), never from a
token header (``jku`` / ``x5u`` are ignored), so a token cannot make this server
fetch an attacker's URL. Fetches have a timeout and a size cap, do not follow
redirects, and are rate limited: a token with an unknown ``kid`` triggers at
most one refresh per ``min_refresh_interval`` seconds. When the cached set has
expired and cannot be refreshed, verification fails closed
(:class:`~faq_core.security.errors.AuthUnavailableError`, HTTP 503).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from faq_core.network_guard import host_of
from faq_core.security.errors import AuthUnavailableError, invalid_token
from observability.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

_log = get_logger(__name__)

DEFAULT_MAX_JWKS_BYTES: Final = 256_000
DEFAULT_MIN_REFRESH_INTERVAL: Final = 30.0
MIN_RSA_KEY_BITS: Final = 2048

# Which JWK key type / curve each allowed algorithm needs (alg-confusion guard).
_RSA_ALGS: Final = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512"})
_EC_CURVES: Final[dict[str, str]] = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}
_EDDSA_CURVES: Final = frozenset({"Ed25519", "Ed448"})

JWK = dict[str, Any]


class JWKSFetchError(RuntimeError):
    """The key set could not be fetched or parsed (message names the host only)."""


class JWKSFetcher(Protocol):
    """Fetches the current JSON Web Key Set."""

    async def fetch(self) -> list[JWK]:
        """Return the set's ``keys``.

        Raises:
            JWKSFetchError: transport error, non-200 status, oversized or
                malformed document.
        """
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


class HttpxJWKSFetcher:
    """GETs the JWKS document over httpx (no redirects, size-capped)."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int = DEFAULT_MAX_JWKS_BYTES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._host = host_of(url) or "invalid URL"
        self._max_bytes = max_bytes
        self._client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    async def fetch(self) -> list[JWK]:
        body = bytearray()
        try:
            async with self._client.stream(
                "GET", self._url, headers={"Accept": "application/json"}
            ) as response:
                if response.status_code != httpx.codes.OK:
                    raise JWKSFetchError(
                        f"JWKS fetch from {self._host} returned HTTP {response.status_code}"
                    )
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        raise JWKSFetchError(f"JWKS document from {self._host} is too large")
        except httpx.HTTPError as err:
            raise JWKSFetchError(
                f"JWKS fetch from {self._host} failed: {type(err).__name__}"
            ) from err
        return _parse_key_set(bytes(body), self._host)

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_key_set(body: bytes, host: str) -> list[JWK]:
    try:
        document = json.loads(body)
    except ValueError as err:
        raise JWKSFetchError(f"JWKS document from {host} is not JSON") from err
    keys = document.get("keys") if isinstance(document, dict) else None
    if not isinstance(keys, list):
        raise JWKSFetchError(f"JWKS document from {host} has no 'keys' list")
    return [key for key in keys if isinstance(key, dict) and isinstance(key.get("kty"), str)]


class JWKSCache:
    """Caches the key set for *ttl* seconds; refreshes on expiry or an unknown ``kid``."""

    def __init__(
        self,
        fetcher: JWKSFetcher,
        *,
        ttl: float,
        min_refresh_interval: float = DEFAULT_MIN_REFRESH_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._ttl = ttl
        self._min_refresh_interval = min_refresh_interval
        self._clock = clock
        self._keys: list[JWK] = []
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._lock = asyncio.Lock()

    @property
    def fetch_attempts_allowed(self) -> bool:
        """Whether a refresh may run now (the rate limit on refreshes)."""
        last = self._last_attempt
        return last is None or self._clock() - last >= self._min_refresh_interval

    def _is_fresh(self) -> bool:
        return self._fetched_at is not None and self._clock() - self._fetched_at < self._ttl

    async def signing_key(self, kid: str | None, alg: str) -> Any:  # noqa: ANN401 — PyJWT key object
        """The verification key for a token header's *kid* and *alg*.

        Raises:
            AuthUnavailableError: no fresh key set and none could be fetched.
            AuthError: no matching key, or the key does not fit *alg*.
        """
        if not self._is_fresh():
            await self._refresh()
            if not self._is_fresh():
                raise AuthUnavailableError()
        jwk_data = self._select(kid)
        if jwk_data is None and self.fetch_attempts_allowed:
            await self._refresh()  # the issuer may have rotated its keys
            jwk_data = self._select(kid)
        if jwk_data is None:
            raise invalid_token("unknown_kid")
        return key_for_algorithm(jwk_data, alg)

    def _select(self, kid: str | None) -> JWK | None:
        if kid is not None:
            return next((key for key in self._keys if key.get("kid") == kid), None)
        return self._keys[0] if len(self._keys) == 1 else None

    async def _refresh(self) -> None:
        async with self._lock:
            if not self.fetch_attempts_allowed:
                return  # another request refreshed (or failed to) moments ago
            self._last_attempt = self._clock()
            try:
                keys = await self._fetcher.fetch()
            except JWKSFetchError as err:
                _log.warning("jwks refresh failed", extra={"reason": str(err)})
                return
            self._keys = keys
            self._fetched_at = self._clock()
            _log.info("jwks refreshed", extra={"key_count": len(keys)})

    async def aclose(self) -> None:
        await self._fetcher.aclose()


def key_for_algorithm(jwk_data: JWK, alg: str) -> Any:  # noqa: ANN401 — PyJWT key object
    """Build the public key for *alg* from *jwk_data*, refusing mismatches.

    Raises:
        AuthError: the key is not a public signing key, declares another
            ``alg``, has the wrong key type / curve for *alg*, or is a
            too-small RSA key.
    """
    use = jwk_data.get("use")
    declared_alg = jwk_data.get("alg")
    if (use is not None and use != "sig") or (declared_alg is not None and declared_alg != alg):
        raise invalid_token("key_alg_mismatch")
    if "d" in jwk_data or not _key_type_fits(jwk_data, alg):
        # "d" marks a private key; a JWKS must publish public keys only.
        raise invalid_token("key_alg_mismatch")
    try:
        key = jwt.PyJWK(jwk_data, algorithm=alg).key
    except (jwt.PyJWTError, ValueError, TypeError) as err:
        raise invalid_token("unusable_key") from err
    if isinstance(key, RSAPublicKey) and key.key_size < MIN_RSA_KEY_BITS:
        raise invalid_token("weak_key")
    return key


def _key_type_fits(jwk_data: JWK, alg: str) -> bool:
    kty, crv = jwk_data.get("kty"), jwk_data.get("crv")
    if alg in _RSA_ALGS:
        return kty == "RSA"
    if alg in _EC_CURVES:
        return kty == "EC" and crv == _EC_CURVES[alg]
    if alg == "EdDSA":
        return kty == "OKP" and crv in _EDDSA_CURVES
    return False
