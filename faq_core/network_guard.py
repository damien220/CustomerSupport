"""Offline-mode network guard (``LLM_MODE=offline``).

At startup :func:`enforce_network_policy` lists every outbound endpoint the
configuration would contact — the LLM base URL, a remote embedder, the Cohere
reranker, Langfuse, the escalation webhook, the OAuth issuer and its JWKS — and
refuses to start if any of them is not local. Local
means: a unix-socket URL, ``localhost`` / ``*.localhost``, a loopback IP, or a
host named in ``OFFLINE_ALLOWED_HOSTS`` (IP literals there must be private).
Hostnames are judged by name only — no DNS lookups.

It also sets ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` (unless already set)
so a local sentence-transformers embedder or BGE reranker loads only from the
local model cache instead of downloading, and :func:`offline_url_policy`
restricts CLI ingestion of URLs (including redirect hops) to local hosts.

Offline mode is incompatible with remote hosting by definition (plan §2):
:func:`enforce_hosting_policy` refuses an HTTP surface bound to a non-loopback
interface (unless it is a private IP listed in ``OFFLINE_ALLOWED_HOSTS``), a
public ``OAUTH_RESOURCE_URL`` and public ``ALLOWED_HOSTS`` names.

Messages name the setting and the host only — never a full URL, which could
carry credentials.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from faq_core.errors import IngestError
from faq_core.models import LLMMode
from faq_core.security.origins import host_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from faq_core.config import AppConfig

_UNIX_SCHEMES: Final = frozenset({"unix", "http+unix", "https+unix"})
_HF_OFFLINE_ENV: Final = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class NetworkPolicyError(RuntimeError):
    """The configuration contacts a non-local endpoint while ``LLM_MODE=offline``."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One outbound endpoint and the setting it comes from."""

    setting: str
    url: str


def parse_allowed_hosts(raw: str) -> frozenset[str]:
    """Parse ``OFFLINE_ALLOWED_HOSTS`` (comma-separated, case-insensitive).

    Raises:
        NetworkPolicyError: an entry is a public (non-private) IP address.
    """
    hosts = {part.strip().lower().strip("[]") for part in raw.split(",") if part.strip()}
    for host in hosts:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            continue  # a hostname — trusted as the operator wrote it
        if not (address.is_private or address.is_loopback or address.is_link_local):
            raise NetworkPolicyError(
                f"OFFLINE_ALLOWED_HOSTS may only list private addresses, not {host}"
            )
    return frozenset(hosts)


def host_of(url: str) -> str:
    """The lower-cased host of *url* (``""`` when there is none)."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def is_local_url(url: str, allowed_hosts: frozenset[str] = frozenset()) -> bool:
    """Whether *url* stays on this machine or on an explicitly allowed host."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    if parts.scheme.lower() in _UNIX_SCHEMES:
        return True
    host = (parts.hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost") or host in allowed_hosts:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def outbound_endpoints(config: AppConfig) -> list[Endpoint]:
    """Every remote endpoint the configured LLM, embedder, reranker, tracing,
    escalation webhook and OAuth key set use.

    Raises:
        ValueError: an unknown ``LLM_PROVIDER`` or incomplete LLM settings.
    """
    from faq_core.llm.presets import resolve_llm_settings
    from faq_core.retrieval.rerankers import DEFAULT_COHERE_URL

    endpoints: list[Endpoint] = []
    if config.LLM_MODE.generates_answers:
        endpoints.append(Endpoint("LLM_BASE_URL", resolve_llm_settings(config).base_url))
    embedder = config.EMBED_PROVIDER.strip().lower()
    if embedder == "openai":
        endpoints.append(Endpoint("EMBED_BASE_URL", config.EMBED_BASE_URL))
    elif embedder == "ollama":
        endpoints.append(Endpoint("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL))
    if config.RERANKER.strip().lower() == "cohere":
        endpoints.append(Endpoint("RERANKER=cohere", DEFAULT_COHERE_URL))
    if (
        config.LANGFUSE_SECRET_KEY.get_secret_value()
        and config.LANGFUSE_PUBLIC_KEY.get_secret_value()
    ):
        endpoints.append(Endpoint("LANGFUSE_HOST", config.LANGFUSE_HOST))
    # Listed whenever a URL is set (not only with ESCALATION_SINK=webhook), so a
    # later switch of the sink cannot reach a remote host unchecked.
    if config.ESCALATION_WEBHOOK_URL.strip():
        endpoints.append(Endpoint("ESCALATION_WEBHOOK_URL", config.ESCALATION_WEBHOOK_URL.strip()))
    # OAuth resource-server mode fetches the issuer's keys; listed whenever set.
    for setting in ("OAUTH_ISSUER", "OAUTH_JWKS_URL"):
        value = str(getattr(config, setting)).strip()
        if value:
            endpoints.append(Endpoint(setting, value))
    return endpoints


def enforce_network_policy(config: AppConfig) -> None:
    """Fail fast when ``LLM_MODE=offline`` and any endpoint is not local.

    Does nothing in the other modes.

    Raises:
        NetworkPolicyError: offline mode with a non-local endpoint; the message
            lists every offending setting and host.
    """
    if config.LLM_MODE is not LLMMode.OFFLINE:
        return
    allowed = parse_allowed_hosts(config.OFFLINE_ALLOWED_HOSTS)
    try:
        endpoints = outbound_endpoints(config)
    except ValueError as err:
        raise NetworkPolicyError(f"LLM_MODE=offline: {err}") from err
    refused = [
        f"{endpoint.setting} -> {host_of(endpoint.url) or 'invalid URL'}"
        for endpoint in endpoints
        if not is_local_url(endpoint.url, allowed)
    ]
    if refused:
        raise NetworkPolicyError(
            "LLM_MODE=offline refuses non-local endpoints: "
            + "; ".join(refused)
            + ". Use a local runtime (LLM_PROVIDER=ollama|vllm|llamacpp|lmstudio), a local "
            "embedder/reranker, or add a private host to OFFLINE_ALLOWED_HOSTS."
        )
    for name in _HF_OFFLINE_ENV:
        os.environ.setdefault(name, "1")


def is_loopback_host(host: str) -> bool:
    """Whether binding *host* listens on this machine only (``localhost`` or a loopback IP)."""
    host = host.strip().lower().strip("[]")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def enforce_hosting_policy(config: AppConfig, bind_host: str) -> None:
    """Refuse remote hosting of an HTTP surface when ``LLM_MODE=offline``.

    Allowed binds: loopback, or a private IP literal listed in
    ``OFFLINE_ALLOWED_HOSTS`` (an air-gapped LAN). ``0.0.0.0`` / ``::`` and
    public addresses are refused, as are a non-local ``OAUTH_RESOURCE_URL`` and
    non-local ``ALLOWED_HOSTS`` names (both mean "reachable from elsewhere").
    Does nothing in the other modes.

    Raises:
        NetworkPolicyError: offline mode with remote hosting settings.
    """
    if config.LLM_MODE is not LLMMode.OFFLINE:
        return
    allowed = parse_allowed_hosts(config.OFFLINE_ALLOWED_HOSTS)
    refused: list[str] = []
    bind = bind_host.strip().lower().strip("[]")
    if not (is_loopback_host(bind) or (bind in allowed and _is_private_ip(bind))):
        refused.append(f"bind host {bind or 'empty'}")
    resource = config.OAUTH_RESOURCE_URL.strip()
    if resource and not is_local_url(resource, allowed):
        refused.append(f"OAUTH_RESOURCE_URL -> {host_of(resource) or 'invalid URL'}")
    for entry in (part.strip() for part in config.ALLOWED_HOSTS.split(",")):
        name = host_name(entry).lower()
        if entry and not (is_loopback_host(name) or name in allowed):
            refused.append(f"ALLOWED_HOSTS -> {name[:80]}")
    if refused:
        raise NetworkPolicyError(
            "LLM_MODE=offline cannot be hosted remotely: "
            + "; ".join(refused)
            + ". Bind to 127.0.0.1 (or a private IP listed in OFFLINE_ALLOWED_HOSTS)."
        )


def _is_private_ip(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private and not address.is_unspecified


def offline_url_policy(config: AppConfig) -> Callable[[str], None] | None:
    """A URL check for the ingest fetcher in offline mode (``None`` otherwise).

    The returned callable raises :class:`~faq_core.errors.IngestError` for a
    non-local URL; the fetcher calls it for the first URL and every redirect.
    """
    if config.LLM_MODE is not LLMMode.OFFLINE:
        return None
    allowed = parse_allowed_hosts(config.OFFLINE_ALLOWED_HOSTS)

    def _check(url: str) -> None:
        if not is_local_url(url, allowed):
            raise IngestError(
                "LLM_MODE=offline only ingests from local hosts",
                host=host_of(url) or "invalid URL",
            )

    return _check
