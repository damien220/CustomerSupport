"""Origin and Host header policy for the HTTP surfaces.

* **Origin** (MCP Streamable HTTP: "Servers MUST validate the Origin header";
  an invalid one gets 403): a request *with* an ``Origin`` header must name an
  allowlisted origin; requests without one (agents, curl, server-to-server) are
  allowed. ``null`` and malformed origins are refused.
* **Host** (DNS-rebinding protection): on a loopback bind only loopback names
  and the configured ``ALLOWED_HOSTS`` are accepted, so a public DNS name that
  rebinds to 127.0.0.1 is refused (421). On other binds the Host check applies
  when ``ALLOWED_HOSTS`` is set.

Comparison is on normalised values: lower-case scheme and host, default ports
(80/443) dropped, IPv6 in brackets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from faq_core.security.errors import SecurityConfigError

LOOPBACK_HOST_NAMES: Final = frozenset({"localhost", "127.0.0.1", "[::1]"})
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}


def normalize_origin(value: str) -> str | None:
    """``scheme://host[:port]`` in canonical form, or ``None`` when not a valid origin."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host or parts.username or parts.password:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def split_host_header(value: str) -> tuple[str, int | None] | None:
    """``(host, port)`` from a Host header (``example.com:8000``, ``[::1]:80``)."""
    value = value.strip().lower()
    if not value or any(char in value for char in "/@?#\\ "):
        return None
    try:
        parts = urlsplit(f"//{value}")
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname or ""
    if not host:
        return None
    return (f"[{host}]" if ":" in host else host), port


@dataclass(frozen=True, slots=True)
class HostOriginPolicy:
    """The allowlists; ``allowed_hosts=None`` disables the Host check."""

    allowed_origins: frozenset[str]
    allowed_hosts: frozenset[str] | None

    def origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return True
        normalized = normalize_origin(origin)
        return normalized is not None and normalized in self.allowed_origins

    def host_allowed(self, host_header: str | None) -> bool:
        if self.allowed_hosts is None:
            return True
        if host_header is None:
            return False
        parsed = split_host_header(host_header)
        if parsed is None:
            return False
        host, port = parsed
        if host in self.allowed_hosts:
            return True
        return port is not None and f"{host}:{port}" in self.allowed_hosts


def parse_allowed_origins(raw: str) -> frozenset[str]:
    """Parse ``ALLOWED_ORIGINS``.

    Raises:
        SecurityConfigError: a wildcard or an entry that is not an http(s) origin.
    """
    origins: set[str] = set()
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        normalized = normalize_origin(entry) if "*" not in entry else None
        if normalized is None:
            raise SecurityConfigError(
                f"ALLOWED_ORIGINS entry {entry[:80]!r} is not an origin like "
                "https://app.example.com (wildcards are not allowed)"
            )
        origins.add(normalized)
    return frozenset(origins)


def parse_allowed_hosts(raw: str) -> frozenset[str]:
    """Parse ``ALLOWED_HOSTS`` (``host`` = any port, ``host:port`` = that port).

    Raises:
        SecurityConfigError: a wildcard or malformed entry.
    """
    hosts: set[str] = set()
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        parsed = split_host_header(entry) if "*" not in entry else None
        if parsed is None:
            raise SecurityConfigError(
                f"ALLOWED_HOSTS entry {entry[:80]!r} is not a host or host:port "
                "(wildcards are not allowed)"
            )
        host, port = parsed
        hosts.add(host if port is None else f"{host}:{port}")
    return frozenset(hosts)


def host_name(entry: str) -> str:
    """The bare host of an ``ALLOWED_HOSTS`` entry (brackets and port removed)."""
    parsed = split_host_header(entry)
    host = parsed[0] if parsed else entry
    return host.strip("[]")
