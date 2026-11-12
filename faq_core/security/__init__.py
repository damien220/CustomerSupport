"""Remote-access security for the HTTP surfaces (REST/WebSocket and MCP over HTTP).

Framework-free building blocks — token verification (static keys, OAuth JWT),
Origin/Host policy, rate limiting — assembled by :mod:`faq_core.security.policy`
and enforced by the ASGI middleware in :mod:`api.security`. MCP stdio does not
use any of it. Submodules are imported directly (this package imports nothing,
which keeps ``faq_core.network_guard`` free of import cycles).
"""
