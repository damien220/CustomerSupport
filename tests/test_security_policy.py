"""Static keys, rate limiting, Origin/Host policy and the fail-closed policy factory."""

from __future__ import annotations

import asyncio
import logging

import pytest

from faq_core.network_guard import NetworkPolicyError
from faq_core.security.errors import INVALID_TOKEN_DESCRIPTION, AuthError, SecurityConfigError
from faq_core.security.origins import HostOriginPolicy, normalize_origin, parse_allowed_origins
from faq_core.security.policy import AuthMode, build_security
from faq_core.security.ratelimit import SlidingWindowRateLimiter
from faq_core.security.tokens import StaticTokenVerifier, parse_api_keys
from tests.conftest import make_api_token, security_config

# ── static keys ──────────────────────────────────────────────────────────────


def test_static_keys_are_matched_by_name_and_never_stored_raw() -> None:
    crm, bot = make_api_token(), make_api_token()
    keys = parse_api_keys(f"crm:{crm}, bot:{bot}")
    verifier = StaticTokenVerifier(keys)

    principal = asyncio.run(verifier.verify(bot))

    assert principal.key_id == "key:bot"
    assert principal.method == "token"
    assert all(crm not in repr(key) and bot not in repr(key) for key in keys)


def test_an_unknown_static_token_gets_the_generic_invalid_token_error() -> None:
    verifier = StaticTokenVerifier(parse_api_keys(f"crm:{make_api_token()}"))

    with pytest.raises(AuthError) as caught:
        asyncio.run(verifier.verify(make_api_token()))

    assert caught.value.status_code == 401
    assert caught.value.error == "invalid_token"
    assert caught.value.description == INVALID_TOKEN_DESCRIPTION


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("crm:short", "at least 32"),
        ("crm:", "name:token"),
        ("no-separator", "name:token"),
        ("bad name:{token}", "name may use"),
        ("crm:{token},crm:{other}", "duplicate"),
        ("a:{token},b:{token}", "duplicate"),
        ("crm:{token}%%", "at least 32"),
    ],
)
def test_bad_api_keys_are_refused_without_echoing_the_token(raw: str, match: str) -> None:
    token, other = make_api_token(), make_api_token()
    entry = raw.format(token=token, other=other)

    with pytest.raises(SecurityConfigError, match=match) as caught:
        parse_api_keys(entry)

    assert token not in str(caught.value)
    assert other not in str(caught.value)


def test_static_verifier_needs_at_least_one_key() -> None:
    with pytest.raises(SecurityConfigError):
        StaticTokenVerifier([])


# ── rate limiter ─────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_sliding_window_refuses_over_the_limit_and_frees_slots_later() -> None:
    clock = _Clock()
    limiter = SlidingWindowRateLimiter(2, 60.0, clock=clock)

    first, second, third = (limiter.check("key:a") for _ in range(3))
    other = limiter.check("key:b")
    clock.now += 61
    later = limiter.check("key:a")

    assert first.allowed and second.allowed
    assert not third.allowed and third.retry_after == 60
    assert other.allowed  # limits are per key
    assert later.allowed


def test_rate_limiter_bounds_the_number_of_tracked_keys() -> None:
    limiter = SlidingWindowRateLimiter(1, 60.0, max_keys=2)

    for key in ("a", "b", "c"):
        limiter.check(key)

    assert limiter.check("a").allowed  # "a" was evicted (least recently used)
    assert not limiter.check("c").allowed


# ── Origin / Host ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://App.Example.com", "https://app.example.com"),
        ("https://app.example.com:443", "https://app.example.com"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://[::1]:80", "http://[::1]"),
        ("null", None),
        ("file://", None),
        ("https://user:pw@app.example.com", None),
        ("https://app.example.com/path", None),
        ("https://app.example.com:99999", None),
    ],
)
def test_origins_are_normalised(raw: str, expected: str | None) -> None:
    assert normalize_origin(raw) == expected


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (None, True),  # non-browser client
        ("https://app.example.com", True),
        ("HTTPS://APP.EXAMPLE.COM:443", True),
        ("https://app.example.com:8443", False),  # other port
        ("http://app.example.com", False),  # other scheme
        ("https://evil.example.com", False),
        ("https://app.example.com.evil.net", False),
        ("null", False),
        ("", False),
    ],
)
def test_origin_allowlist(origin: str | None, allowed: bool) -> None:
    policy = HostOriginPolicy(parse_allowed_origins("https://app.example.com"), None)

    assert policy.origin_allowed(origin) is allowed


def test_wildcard_origins_are_refused() -> None:
    with pytest.raises(SecurityConfigError, match="wildcards"):
        parse_allowed_origins("https://*.example.com")


def test_loopback_bind_checks_the_host_header() -> None:
    policy = build_security(
        security_config(ALLOWED_HOSTS="", API_KEYS=f"k:{make_api_token()}"),
        bind_host="127.0.0.1",
    )

    assert policy.origins.host_allowed("localhost:8001")
    assert policy.origins.host_allowed("127.0.0.1")
    assert policy.origins.host_allowed("[::1]:8001")
    assert not policy.origins.host_allowed("attacker.example:8001")  # DNS rebinding
    assert not policy.origins.host_allowed(None)
    assert not policy.origins.host_allowed("localhost@evil")


# ── policy factory: fail closed ──────────────────────────────────────────────


def test_token_mode_is_the_default_and_requires_keys() -> None:
    config = security_config(AUTH_MODE="token")

    assert AuthMode(config.AUTH_MODE) is AuthMode.TOKEN
    with pytest.raises(SecurityConfigError, match="API_KEYS"):
        build_security(config, bind_host="127.0.0.1")


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "10.0.0.5", "mcp.example.com", "::"])  # noqa: S104
def test_auth_none_is_refused_on_a_non_loopback_bind(bind_host: str) -> None:
    with pytest.raises(SecurityConfigError, match="loopback"):
        build_security(security_config(AUTH_MODE="none"), bind_host=bind_host)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ALLOWED_HOSTS": "mcp.example.com"},
        {"OAUTH_RESOURCE_URL": "https://mcp.example.com/mcp"},
    ],
)
def test_auth_none_is_refused_with_public_hosting_settings(overrides: dict[str, str]) -> None:
    config = security_config(AUTH_MODE="none", **overrides)

    with pytest.raises(SecurityConfigError, match="unauthenticated"):
        build_security(config, bind_host="127.0.0.1")


def test_auth_none_on_loopback_starts_with_a_loud_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        policy = build_security(
            security_config(AUTH_MODE="none", ALLOWED_HOSTS=""), bind_host="127.0.0.1"
        )

    assert not policy.requires_auth
    assert policy.verifier is None
    assert "UNAUTHENTICATED" in caplog.text


def test_unknown_auth_mode_is_refused() -> None:
    with pytest.raises(SecurityConfigError, match="AUTH_MODE"):
        build_security(security_config(AUTH_MODE="open"), bind_host="127.0.0.1")


_OAUTH = {
    "AUTH_MODE": "oauth",
    "OAUTH_ISSUER": "https://auth.example.com",
    "OAUTH_RESOURCE_URL": "https://mcp.example.com/mcp",
    "OAUTH_JWKS_URL": "https://auth.example.com/.well-known/jwks.json",
}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"OAUTH_ISSUER": ""}, "OAUTH_ISSUER is required"),
        ({"OAUTH_RESOURCE_URL": ""}, "OAUTH_RESOURCE_URL is required"),
        ({"OAUTH_JWKS_URL": ""}, "OAUTH_JWKS_URL is required"),
        ({"OAUTH_JWKS_URL": "http://auth.example.com/jwks"}, "https"),
        ({"OAUTH_ISSUER": "https://u:p@auth.example.com"}, "credentials"),
        ({"OAUTH_ALGORITHMS": "HS256"}, "asymmetric"),
        ({"OAUTH_ALGORITHMS": "RS256,none"}, "asymmetric"),
        ({"OAUTH_ALGORITHMS": ""}, "asymmetric"),
        ({"OAUTH_CLOCK_SKEW": 3600}, "CLOCK_SKEW"),
        ({"OAUTH_REQUIRED_SCOPES": 'a"b'}, "scope"),
    ],
)
def test_oauth_settings_are_validated(overrides: dict[str, object], match: str) -> None:
    config = security_config(**{**_OAUTH, **overrides})

    with pytest.raises(SecurityConfigError, match=match):
        build_security(config, bind_host="127.0.0.1")


def test_both_mode_needs_static_keys_too() -> None:
    with pytest.raises(SecurityConfigError, match="API_KEYS"):
        build_security(security_config(**{**_OAUTH, "AUTH_MODE": "both"}), bind_host="127.0.0.1")


def test_oauth_policy_advertises_rfc9728_metadata() -> None:
    policy = build_security(
        security_config(**_OAUTH, OAUTH_REQUIRED_SCOPES="support.read"), bind_host="127.0.0.1"
    )

    assert policy.resource_metadata_url == (
        "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
    )
    assert policy.resource_metadata_paths == (
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
    )
    assert policy.protected_resource_metadata() == {
        "resource": "https://mcp.example.com/mcp",
        "authorization_servers": ["https://auth.example.com"],
        "bearer_methods_supported": ["header"],
        "resource_name": "Customer Support Assistant",
        "scopes_supported": ["support.read"],
    }


def test_offline_mode_refuses_remote_hosting_before_building_a_policy() -> None:
    config = security_config(
        LLM_MODE="offline", LLM_MODEL="llama3", API_KEYS=f"k:{make_api_token()}"
    )

    with pytest.raises(NetworkPolicyError, match="cannot be hosted remotely"):
        build_security(config, bind_host="0.0.0.0")  # noqa: S104
