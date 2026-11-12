"""The security policy of one HTTP surface, built from configuration.

:func:`build_security` is the only place the auth settings are interpreted
(``bootstrap.build_security`` calls it). It fails closed: a mode without its
settings, an unsafe URL or an unauthenticated public bind raises
:class:`~faq_core.security.errors.SecurityConfigError` at startup.

=============  ===============================================================
``AUTH_MODE``  requirements
=============  ===============================================================
``token``      ``API_KEYS`` with at least one ``name:token`` (default mode)
``oauth``      ``OAUTH_ISSUER``, ``OAUTH_RESOURCE_URL``, ``OAUTH_JWKS_URL``
               (https unless local), an asymmetric ``OAUTH_ALGORITHMS`` list
``both``       everything ``token`` and ``oauth`` need
``none``       a loopback bind, loopback-only ``ALLOWED_HOSTS``, no public
               ``OAUTH_RESOURCE_URL``; logs a warning on every start
=============  ===============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from faq_core.network_guard import enforce_hosting_policy, is_local_url, is_loopback_host
from faq_core.security.errors import SecurityConfigError
from faq_core.security.jwks import HttpxJWKSFetcher, JWKSCache, JWKSFetcher
from faq_core.security.oauth import ALLOWED_JWT_ALGORITHMS, JWTVerifier, OAuthSettings
from faq_core.security.origins import (
    LOOPBACK_HOST_NAMES,
    HostOriginPolicy,
    host_name,
    parse_allowed_hosts,
    parse_allowed_origins,
)
from faq_core.security.ratelimit import SlidingWindowRateLimiter
from faq_core.security.tokens import ChainedVerifier, StaticTokenVerifier, parse_api_keys
from observability.log import get_logger

if TYPE_CHECKING:
    from faq_core.config import AppConfig
    from faq_core.security.tokens import TokenVerifier

_log = get_logger(__name__)

WELL_KNOWN_RESOURCE_METADATA: Final = "/.well-known/oauth-protected-resource"
REALM: Final = "customer-support"
RESOURCE_NAME: Final = "Customer Support Assistant"


class AuthMode(StrEnum):
    TOKEN = "token"  # noqa: S105 — a mode name, not a credential
    OAUTH = "oauth"
    BOTH = "both"
    NONE = "none"

    @property
    def uses_static_keys(self) -> bool:
        return self in (AuthMode.TOKEN, AuthMode.BOTH)

    @property
    def uses_oauth(self) -> bool:
        return self in (AuthMode.OAUTH, AuthMode.BOTH)


@dataclass(slots=True)
class SecurityPolicy:
    """Everything the security middleware enforces for one surface.

    ``verifier`` is ``None`` only in ``AUTH_MODE=none``; a limiter is ``None``
    when its limit is 0 (disabled).
    """

    mode: AuthMode
    verifier: TokenVerifier | None
    origins: HostOriginPolicy
    key_limiter: SlidingWindowRateLimiter | None
    anon_limiter: SlidingWindowRateLimiter | None
    oauth: OAuthSettings | None = None
    keys: JWKSCache | None = None
    _closed: bool = field(default=False, repr=False)

    @property
    def requires_auth(self) -> bool:
        return self.mode is not AuthMode.NONE

    @property
    def resource_metadata_url(self) -> str | None:
        """RFC 9728 §3.1 metadata URL for the resource (path inserted after the host)."""
        if self.oauth is None:
            return None
        parts = urlsplit(self.oauth.resource)
        path = parts.path.rstrip("/")
        return f"{parts.scheme}://{parts.netloc}{WELL_KNOWN_RESOURCE_METADATA}{path}"

    @property
    def resource_metadata_paths(self) -> tuple[str, ...]:
        """Paths serving the metadata: path-inserted first, then the root fallback."""
        if self.oauth is None:
            return ()
        path = urlsplit(self.oauth.resource).path.rstrip("/")
        paths = [WELL_KNOWN_RESOURCE_METADATA + path] if path else []
        return (*paths, WELL_KNOWN_RESOURCE_METADATA)

    def protected_resource_metadata(self) -> dict[str, Any] | None:
        """The RFC 9728 document advertised to MCP clients (``None`` without OAuth)."""
        if self.oauth is None:
            return None
        document: dict[str, Any] = {
            "resource": self.oauth.resource,
            "authorization_servers": [self.oauth.issuer],
            "bearer_methods_supported": ["header"],
            "resource_name": RESOURCE_NAME,
        }
        if self.oauth.required_scopes:
            document["scopes_supported"] = list(self.oauth.required_scopes)
        return document

    async def aclose(self) -> None:
        """Close the JWKS client (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self.keys is not None:
            await self.keys.aclose()


def parse_auth_mode(raw: str) -> AuthMode:
    """Raises: SecurityConfigError: an unknown ``AUTH_MODE``."""
    try:
        return AuthMode(raw.strip().lower())
    except ValueError as err:
        valid = ", ".join(mode.value for mode in AuthMode)
        raise SecurityConfigError(f"AUTH_MODE must be one of {valid}") from err


def build_security(
    config: AppConfig, *, bind_host: str, jwks_fetcher: JWKSFetcher | None = None
) -> SecurityPolicy:
    """Build and validate the security policy for a surface bound to *bind_host*.

    *jwks_fetcher* replaces the HTTP key-set fetcher (tests).

    Raises:
        SecurityConfigError: missing, invalid or unsafe settings (fail closed).
        NetworkPolicyError: ``LLM_MODE=offline`` with remote hosting settings.
    """
    enforce_hosting_policy(config, bind_host)
    mode = parse_auth_mode(config.AUTH_MODE)
    origins = _origin_policy(config, bind_host)
    key_limiter, anon_limiter = _limiters(config)
    if mode is AuthMode.NONE:
        _check_unauthenticated_is_local(config, bind_host)
        _log.warning(
            "AUTH_MODE=none: the HTTP surface is UNAUTHENTICATED — anyone who can reach "
            "this loopback port can read and change the knowledge base. Never expose it "
            "through a tunnel or proxy.",
            extra={"bind_host": bind_host},
        )
        return SecurityPolicy(mode, None, origins, key_limiter, anon_limiter)

    static = None
    if mode.uses_static_keys:
        keys = parse_api_keys(config.API_KEYS.get_secret_value())
        if not keys:
            raise SecurityConfigError(
                f"AUTH_MODE={mode.value} needs API_KEYS (comma-separated name:token pairs); "
                "refusing to start without authentication"
            )
        static = StaticTokenVerifier(keys)
    oauth_settings: OAuthSettings | None = None
    cache: JWKSCache | None = None
    verifier: TokenVerifier | None = static
    if mode.uses_oauth:
        oauth_settings = _oauth_settings(config)
        fetcher = jwks_fetcher or HttpxJWKSFetcher(
            config.OAUTH_JWKS_URL.strip(), timeout=config.OAUTH_JWKS_TIMEOUT
        )
        cache = JWKSCache(fetcher, ttl=config.OAUTH_JWKS_CACHE_TTL)
        jwt_verifier = JWTVerifier(oauth_settings, cache)
        verifier = jwt_verifier if static is None else ChainedVerifier(static, jwt_verifier)
    if not is_loopback_host(bind_host) and origins.allowed_hosts is None:
        _log.warning(
            "ALLOWED_HOSTS is not set for a non-loopback bind; the Host header is not checked",
            extra={"bind_host": bind_host},
        )
    return SecurityPolicy(
        mode, verifier, origins, key_limiter, anon_limiter, oauth=oauth_settings, keys=cache
    )


def _origin_policy(config: AppConfig, bind_host: str) -> HostOriginPolicy:
    origins = parse_allowed_origins(config.ALLOWED_ORIGINS)
    hosts = parse_allowed_hosts(config.ALLOWED_HOSTS)
    if is_loopback_host(bind_host):
        return HostOriginPolicy(origins, hosts | LOOPBACK_HOST_NAMES)
    return HostOriginPolicy(origins, hosts or None)


def _limiters(
    config: AppConfig,
) -> tuple[SlidingWindowRateLimiter | None, SlidingWindowRateLimiter | None]:
    limits = {
        "RATE_LIMIT_REQUESTS": config.RATE_LIMIT_REQUESTS,
        "RATE_LIMIT_ANON_REQUESTS": config.RATE_LIMIT_ANON_REQUESTS,
    }
    if any(value < 0 for value in limits.values()) or config.RATE_LIMIT_WINDOW <= 0:
        raise SecurityConfigError(
            "RATE_LIMIT_REQUESTS / RATE_LIMIT_ANON_REQUESTS must be >= 0 (0 disables) and "
            "RATE_LIMIT_WINDOW > 0"
        )
    window = config.RATE_LIMIT_WINDOW
    key_limit, anon_limit = config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_ANON_REQUESTS
    return (
        SlidingWindowRateLimiter(key_limit, window) if key_limit > 0 else None,
        SlidingWindowRateLimiter(anon_limit, window) if anon_limit > 0 else None,
    )


def _check_unauthenticated_is_local(config: AppConfig, bind_host: str) -> None:
    if not is_loopback_host(bind_host):
        raise SecurityConfigError(
            f"AUTH_MODE=none is only allowed on a loopback bind (127.0.0.1 / localhost / ::1), "
            f"not {bind_host[:80]!r}"
        )
    public_hosts = [
        entry
        for entry in (part.strip() for part in config.ALLOWED_HOSTS.split(","))
        if entry and not is_loopback_host(host_name(entry))
    ]
    resource = config.OAUTH_RESOURCE_URL.strip()
    if public_hosts or (resource and not is_local_url(resource)):
        raise SecurityConfigError(
            "AUTH_MODE=none cannot be combined with public ALLOWED_HOSTS or "
            "OAUTH_RESOURCE_URL (that would publish an unauthenticated server)"
        )


def _oauth_settings(config: AppConfig) -> OAuthSettings:
    issuer = _checked_url("OAUTH_ISSUER", config.OAUTH_ISSUER)
    resource = _checked_url("OAUTH_RESOURCE_URL", config.OAUTH_RESOURCE_URL)
    _checked_url("OAUTH_JWKS_URL", config.OAUTH_JWKS_URL)
    algorithms = frozenset(
        part.strip() for part in config.OAUTH_ALGORITHMS.split(",") if part.strip()
    )
    unsupported = sorted(algorithms - ALLOWED_JWT_ALGORITHMS)
    if not algorithms or unsupported:
        raise SecurityConfigError(
            "OAUTH_ALGORITHMS must list asymmetric JWS algorithms from "
            f"{', '.join(sorted(ALLOWED_JWT_ALGORITHMS))}; refused: {', '.join(unsupported)}"
        )
    if config.OAUTH_CLOCK_SKEW < 0 or config.OAUTH_CLOCK_SKEW > 300:
        raise SecurityConfigError("OAUTH_CLOCK_SKEW must be between 0 and 300 seconds")
    if config.OAUTH_JWKS_CACHE_TTL <= 0 or config.OAUTH_JWKS_TIMEOUT <= 0:
        raise SecurityConfigError("OAUTH_JWKS_CACHE_TTL and OAUTH_JWKS_TIMEOUT must be > 0")
    scopes = tuple(
        dict.fromkeys(
            part.strip() for part in config.OAUTH_REQUIRED_SCOPES.split(",") if part.strip()
        )
    )
    if any('"' in scope or "\\" in scope or " " in scope for scope in scopes):
        raise SecurityConfigError("OAUTH_REQUIRED_SCOPES entries must be plain scope tokens")
    return OAuthSettings(
        issuer=issuer,
        # Kept exactly as configured: the metadata's "resource" must equal the URL
        # the user enters in the client (audience matching tolerates a trailing "/").
        resource=resource,
        algorithms=algorithms,
        required_scopes=scopes,
        leeway=config.OAUTH_CLOCK_SKEW,
    )


def _checked_url(setting: str, raw: str) -> str:
    """An absolute https URL (http only for local hosts), no credentials or fragment.

    Raises:
        SecurityConfigError: missing or unsafe URL (message names the setting only).
    """
    url = raw.strip()
    if not url:
        raise SecurityConfigError(f"{setting} is required when AUTH_MODE uses oauth")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 — raises ValueError on a malformed port
    except ValueError as err:
        raise SecurityConfigError(f"{setting} is not a valid URL") from err
    scheme = parts.scheme.lower()
    if scheme not in ("https", "http") or not parts.hostname:
        raise SecurityConfigError(f"{setting} must be an absolute https URL")
    if parts.username or parts.password or parts.fragment:
        raise SecurityConfigError(f"{setting} must not contain credentials or a fragment")
    if scheme == "http" and not is_local_url(url):
        raise SecurityConfigError(f"{setting} must use https for a non-local host")
    return url
