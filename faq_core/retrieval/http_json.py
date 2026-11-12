"""Minimal JSON-over-HTTP helper shared by the API embedder and the Cohere reranker.

Every call has a timeout, retries transport errors / 429 / 5xx with exponential
backoff, and raises :class:`RemoteServiceError` whose message carries only the
service name and HTTP status — never request headers, so API keys cannot leak
into logs or error envelopes. ``httpx`` (already a core dependency) is imported
lazily so importing the retrieval modules stays cheap.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})
_RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
_BACKOFF_SECONDS: Final = 0.5


class RemoteServiceError(RuntimeError):
    """A remote embedding/rerank call failed after its retries."""


def require_tls(url: str, *, setting: str) -> None:
    """Reject plaintext ``http://`` *url* unless it points at the local machine.

    Raises:
        ValueError: unsupported scheme, or plain HTTP to a non-local host.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{setting} must be an http(s) URL")
    if parts.scheme == "http" and parts.hostname not in LOCAL_HOSTS:
        raise ValueError(f"{setting} must use https for non-local hosts")


async def post_json(
    url: str,
    payload: Mapping[str, object],
    *,
    service: str,
    headers: Mapping[str, str] | None = None,
    request_timeout: float = 30.0,
    max_retries: int = 2,
    transport: httpx.AsyncBaseTransport | None = None,
) -> object:
    """POST *payload* as JSON and return the decoded JSON response.

    Raises:
        RemoteServiceError: non-2xx after retries, a transport failure after
            retries, or a body that is not JSON.
    """
    import httpx

    attempt = 0
    async with httpx.AsyncClient(timeout=request_timeout, transport=transport) as client:
        while True:
            try:
                response = await client.post(url, json=dict(payload), headers=dict(headers or {}))
            except httpx.TransportError as err:
                if attempt >= max_retries:
                    raise RemoteServiceError(
                        f"{service} request failed: {type(err).__name__}"
                    ) from err
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as err:
                        raise RemoteServiceError(f"{service} returned a non-JSON body") from err
                if response.status_code not in _RETRY_STATUSES or attempt >= max_retries:
                    raise RemoteServiceError(f"{service} returned HTTP {response.status_code}")
            await asyncio.sleep(_BACKOFF_SECONDS * 2**attempt)
            attempt += 1
