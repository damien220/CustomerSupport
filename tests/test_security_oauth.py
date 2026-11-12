"""OAuth 2.1 resource-server mode: JWT access tokens checked against a fake JWKS.

Keys are generated per run (RSA 2048 / EC P-256); no network is used — the
JWKS comes from an in-memory fetcher or an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from faq_core.security.errors import (
    INVALID_TOKEN_DESCRIPTION,
    AuthError,
    AuthUnavailableError,
)
from faq_core.security.jwks import HttpxJWKSFetcher, JWKSCache, JWKSFetchError
from faq_core.security.oauth import JWTVerifier, OAuthSettings

ISSUER = "https://auth.example.com"
RESOURCE = "https://mcp.example.com/mcp"
JWKS_URL = "https://auth.example.com/.well-known/jwks.json"


def _rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


RSA_KEY = _rsa_key()
OTHER_RSA_KEY = _rsa_key()
EC_KEY = ec.generate_private_key(ec.SECP256R1())


def _jwk(key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey, kid: str) -> dict[str, Any]:
    public = key.public_key()
    if isinstance(public, rsa.RSAPublicKey):
        data: dict[str, Any] = RSAAlgorithm.to_jwk(public, as_dict=True)
    else:
        data = ECAlgorithm.to_jwk(public, as_dict=True)
    return {**data, "kid": kid, "use": "sig"}


class FakeFetcher:
    """In-memory JWKS source that counts fetches."""

    def __init__(self, keys: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.keys = keys
        self.fail = fail
        self.calls = 0

    async def fetch(self) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail:
            raise JWKSFetchError("JWKS fetch from auth.example.com failed: ConnectError")
        return list(self.keys)

    async def aclose(self) -> None:
        return None


def _claims(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": RESOURCE,
        "sub": "user-123",
        "client_id": "https://claude.ai/oauth/client.json",
        "exp": now + 300,
        "iat": now,
        "scope": "support.read support.write",
    }
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def _sign(
    claims: dict[str, Any],
    *,
    key: Any = RSA_KEY,  # noqa: ANN401
    alg: str = "RS256",
    kid: str | None = "rsa-1",
) -> str:
    headers = {"kid": kid} if kid is not None else {}
    return jwt.encode(claims, key, algorithm=alg, headers=headers)


def _verifier(
    fetcher: FakeFetcher | None = None, *, scopes: tuple[str, ...] = ()
) -> tuple[JWTVerifier, FakeFetcher]:
    fetcher = fetcher or FakeFetcher([_jwk(RSA_KEY, "rsa-1"), _jwk(EC_KEY, "ec-1")])
    settings = OAuthSettings(
        issuer=ISSUER,
        resource=RESOURCE,
        algorithms=frozenset({"RS256", "ES256"}),
        required_scopes=scopes,
        leeway=0,
    )
    return JWTVerifier(settings, JWKSCache(fetcher, ttl=3600)), fetcher


def _refused(verifier: JWTVerifier, token: str) -> AuthError:
    with pytest.raises(AuthError) as caught:
        asyncio.run(verifier.verify(token))
    return caught.value


# ── accepted ─────────────────────────────────────────────────────────────────


def test_a_valid_rs256_token_yields_a_principal_without_the_token() -> None:
    verifier, _ = _verifier()
    token = _sign(_claims())

    principal = asyncio.run(verifier.verify(token))

    assert principal.key_id == "oauth:user-123"
    assert principal.method == "oauth"
    assert principal.issuer == ISSUER
    assert principal.scopes == {"support.read", "support.write"}
    assert token not in repr(principal)


def test_a_valid_es256_token_and_audience_lists_and_trailing_slash() -> None:
    verifier, _ = _verifier()
    token = _sign(
        _claims(aud=["https://other.example.com", RESOURCE + "/"]),
        key=EC_KEY,
        alg="ES256",
        kid="ec-1",
    )

    assert asyncio.run(verifier.verify(token)).subject == "user-123"


def test_scp_list_claim_and_required_scopes() -> None:
    verifier, _ = _verifier(scopes=("support.read",))
    token = _sign(_claims(scope=None, scp=["support.read"]))

    assert asyncio.run(verifier.verify(token)).scopes == {"support.read"}


# ── refused (all with the same generic description) ─────────────────────────


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        ({"aud": "https://other-service.example.com"}, "wrong_audience"),
        ({"aud": "https://mcp.example.com"}, "wrong_audience"),  # other resource on host
        ({"iss": "https://evil.example.com"}, "wrong_issuer"),
        ({"exp": int(time.time()) - 10}, "expired"),
        ({"nbf": int(time.time()) + 600}, "invalid_jwt"),
        ({"exp": None}, "missing_claim"),
        ({"aud": None}, "missing_claim"),
        ({"iss": None}, "missing_claim"),
    ],
)
def test_bad_claims_are_refused(claims: dict[str, Any], reason: str) -> None:
    verifier, _ = _verifier()

    error = _refused(verifier, _sign(_claims(**claims)))

    assert (error.status_code, error.error, error.reason) == (401, "invalid_token", reason)
    assert error.description == INVALID_TOKEN_DESCRIPTION


def test_a_token_signed_by_another_key_is_refused() -> None:
    verifier, _ = _verifier()

    error = _refused(verifier, _sign(_claims(), key=OTHER_RSA_KEY))

    assert error.reason == "bad_signature"


def test_alg_none_is_refused() -> None:
    verifier, _ = _verifier()
    token = jwt.encode(_claims(), key=None, algorithm="none", headers={"kid": "rsa-1"})

    assert _refused(verifier, token).reason == "algorithm_not_allowed"


def test_hs256_signed_with_the_public_key_is_refused() -> None:
    """Algorithm confusion: the RSA public key used as an HMAC secret."""
    verifier, _ = _verifier()
    public_pem = RSA_KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = {"alg": "HS256", "typ": "JWT", "kid": "rsa-1"}
    token = _forge(header, _claims(), public_pem)

    assert _refused(verifier, token).reason == "algorithm_not_allowed"


def _forge(header: dict[str, Any], claims: dict[str, Any], secret: bytes) -> str:
    """An HS256 JWS built by hand (PyJWT refuses PEM keys as HMAC secrets)."""
    import base64
    import hashlib
    import hmac

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f"{b64(json.dumps(header).encode())}.{b64(json.dumps(claims).encode())}"
    signature = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64(signature)}"


def test_an_rs256_header_pointing_at_an_ec_key_is_refused() -> None:
    verifier, _ = _verifier()

    error = _refused(verifier, _sign(_claims(), kid="ec-1"))

    assert error.reason == "key_alg_mismatch"


def test_a_jwks_entry_carrying_a_private_key_is_never_used() -> None:
    private_jwk = json.loads(RSAAlgorithm.to_jwk(RSA_KEY))
    verifier, _ = _verifier(FakeFetcher([{**private_jwk, "kid": "rsa-1"}]))

    assert _refused(verifier, _sign(_claims())).reason == "key_alg_mismatch"


def test_crit_header_is_refused() -> None:
    verifier, _ = _verifier()
    token = jwt.encode(
        _claims(), RSA_KEY, algorithm="RS256", headers={"kid": "rsa-1", "crit": ["exp"]}
    )

    assert _refused(verifier, token).reason in {"unsupported_crit_header", "malformed_jwt"}


@pytest.mark.parametrize("token", ["not-a-jwt", "a.b.c", "eyJhbGciOi.x.y"])
def test_garbage_tokens_are_refused_generically(token: str) -> None:
    verifier, _ = _verifier()

    error = _refused(verifier, token)

    assert error.status_code == 401
    assert error.description == INVALID_TOKEN_DESCRIPTION


def test_missing_scope_is_403_insufficient_scope() -> None:
    verifier, _ = _verifier(scopes=("support.admin",))

    error = _refused(verifier, _sign(_claims()))

    assert (error.status_code, error.error) == (403, "insufficient_scope")
    assert error.required_scopes == ("support.admin",)


# ── JWKS cache ───────────────────────────────────────────────────────────────


def test_unknown_kid_refreshes_at_most_once_per_interval() -> None:
    verifier, fetcher = _verifier()
    asyncio.run(verifier.verify(_sign(_claims())))  # first fetch

    for _ in range(5):
        _refused(verifier, _sign(_claims(), kid="rotated"))

    assert fetcher.calls == 1  # within min_refresh_interval: no fetch storm


def test_rotated_key_is_picked_up_by_a_refresh() -> None:
    clock = [0.0]
    fetcher = FakeFetcher([_jwk(RSA_KEY, "rsa-1")])
    cache = JWKSCache(fetcher, ttl=3600, min_refresh_interval=30, clock=lambda: clock[0])
    settings = OAuthSettings(ISSUER, RESOURCE, frozenset({"RS256"}), leeway=0)
    verifier = JWTVerifier(settings, cache)
    asyncio.run(verifier.verify(_sign(_claims())))

    fetcher.keys = [_jwk(OTHER_RSA_KEY, "rsa-2")]
    clock[0] = 31.0
    principal = asyncio.run(verifier.verify(_sign(_claims(), key=OTHER_RSA_KEY, kid="rsa-2")))

    assert principal.subject == "user-123"
    assert fetcher.calls == 2


def test_unreachable_jwks_fails_closed_with_503() -> None:
    verifier, _ = _verifier(FakeFetcher([], fail=True))

    error = _refused(verifier, _sign(_claims()))

    assert isinstance(error, AuthUnavailableError)
    assert error.status_code == 503


# ── HTTP fetcher: no redirects, size cap, and no token passthrough ───────────


def _fetch(handler: httpx.MockTransport, *, max_bytes: int = 256_000) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        client = httpx.AsyncClient(transport=handler, follow_redirects=False)
        fetcher = HttpxJWKSFetcher(JWKS_URL, timeout=1.0, max_bytes=max_bytes, client=client)
        try:
            return await fetcher.fetch()
        finally:
            await fetcher.aclose()

    return asyncio.run(run())


def test_jwks_fetch_carries_no_client_token() -> None:
    """No token passthrough: the only outbound request (the JWKS GET) has no credentials."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"keys": [_jwk(RSA_KEY, "rsa-1")]})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cache = JWKSCache(HttpxJWKSFetcher(JWKS_URL, timeout=1.0, client=client), ttl=3600)
        settings = OAuthSettings(ISSUER, RESOURCE, frozenset({"RS256"}), leeway=0)
        await JWTVerifier(settings, cache).verify(token)
        await cache.aclose()

    token = _sign(_claims())
    asyncio.run(run())

    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == JWKS_URL
    assert "authorization" not in request.headers
    assert token not in str(request.url)
    assert all(token not in value for value in request.headers.values())


def test_jwks_redirects_are_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest"})

    with pytest.raises(JWKSFetchError, match="HTTP 302"):
        _fetch(httpx.MockTransport(handler))


def test_oversized_jwks_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{" + b" " * 5000 + b"}")

    with pytest.raises(JWKSFetchError, match="too large"):
        _fetch(httpx.MockTransport(handler), max_bytes=1000)


@pytest.mark.parametrize("body", [b"not json", b'{"no": "keys"}', b"[]"])
def test_malformed_jwks_is_refused(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(JWKSFetchError):
        _fetch(httpx.MockTransport(handler))
