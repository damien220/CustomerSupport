"""CLI entry point — ``python app.py --mode mcp|api|ui|ingest|report``.

``--mode mcp`` serves MCP over stdio (default; unauthenticated — the client
launched the process) or, with ``--transport http``, over Streamable HTTP at
``/mcp`` with authentication (:mod:`remote_mcp`). ``--mode api`` serves REST +
WebSocket with authentication on every route but ``/health``. Both HTTP
surfaces bind ``--host`` / ``--port`` (defaults: ``BIND_HOST``, ``PORT`` or
``MCP_HTTP_PORT``) and refuse to start (JSON error on stderr, exit 2) when the
``AUTH_MODE`` settings are missing or unsafe.

Each mode is a thin surface over the same :class:`~faq_core.service.SupportService`
built by :mod:`bootstrap`. Surface modules are imported lazily so ``import app``
stays cheap.

``--mode ingest --source PATH_OR_URL [--source ...]`` loads markdown / text /
HTML / CSV / FAQ-JSON files, folders or http(s) URLs into the knowledge base and
prints one JSON report per source on stdout.

``--mode report [--window-days N] [--limit N] [--output FILE]`` prints the
weekly knowledge-gap report (resolution metrics, ranked gaps, suggested FAQ
topics) as Markdown on stdout, or writes it to FILE and prints a one-line JSON
summary. Query text in it is the PII-redacted text from the query log.

Every mode first runs the offline network guard (``LLM_MODE=offline``): a
non-local endpoint prints a JSON error on stderr and exits with status 2 before
any server starts. ``bootstrap.build_runtime`` enforces it again. A malformed
setting (``.env`` value of the wrong type, inconsistent workflow settings) is
reported the same way (``config_error``, exit 2) — the message names the
setting, never its value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence


def run_mcp() -> None:
    """Serve the MCP tools over stdio (services start in the server lifespan)."""
    from server import mcp

    mcp.run()


def run_mcp_http(host: str, port: int) -> None:
    """Serve the MCP tools over authenticated Streamable HTTP at ``/mcp``."""
    import uvicorn

    from bootstrap import build_security, production_service
    from faq_core.config import get_config
    from observability.log import configure_logging
    from remote_mcp import create_mcp_http_app

    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    security = build_security(config, bind_host=host)
    app = create_mcp_http_app(
        production_service,
        security=security,
        llm_mode=config.LLM_MODE,
        operator_tools=config.MCP_OPERATOR_TOOLS,
    )
    # log_config=None: uvicorn's loggers propagate to the redacting root handler.
    uvicorn.run(app, host=host, port=port, log_config=None)


def run_api(host: str, port: int) -> None:
    """Serve the FastAPI REST + WebSocket app (services start in the app lifespan)."""
    import uvicorn

    from api.app import create_app
    from bootstrap import build_security, production_service
    from faq_core.config import get_config
    from observability.log import configure_logging

    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    security = build_security(config, bind_host=host)
    app = create_app(production_service, security=security)
    # log_config=None: uvicorn's loggers propagate to the redacting root handler.
    uvicorn.run(app, host=host, port=port, log_config=None)


def run_ui() -> None:
    """Serve the Gradio dev UI (search only, so no background scheduler)."""
    from bootstrap import build_runtime, prepare_knowledge_base
    from faq_core.config import get_config
    from observability.log import configure_logging
    from ui.gradio_app import build_demo

    config = get_config()
    configure_logging(config.LOG_LEVEL, secrets=config.secret_values())
    runtime = build_runtime(config)
    asyncio.run(prepare_knowledge_base(runtime))
    demo = build_demo(runtime.service)
    demo.launch(server_name="0.0.0.0", server_port=config.GRADIO_PORT)  # noqa: S104


def run_ingest(sources: Sequence[str]) -> int:
    """Ingest *sources*; print one JSON report per source; return the exit code."""
    from bootstrap import run_ingest as ingest
    from faq_core.errors import IngestError

    try:
        reports = asyncio.run(ingest(sources))
    except IngestError as err:
        print(json.dumps(err.to_dict()), file=sys.stderr)  # noqa: T201 — CLI output
        return 1
    for report in reports:
        print(report.model_dump_json())  # noqa: T201 — CLI output
    return 0


def run_report(window_days: int, limit: int, output: str | None) -> int:
    """Print (or write to *output*) the Markdown gap report; return the exit code.

    Exit codes: 0 ok, 1 invalid ``--window-days`` / ``--limit`` (checked before
    anything is built) or an unwritable *output*, 2 configuration / startup
    error (JSON on stderr, like the server modes).
    """
    from pathlib import Path

    from bootstrap import run_report as report
    from faq_core.errors import InvalidRequestError
    from faq_core.network_guard import NetworkPolicyError

    try:
        result = asyncio.run(report(window_days, limit))
    except InvalidRequestError as err:
        print(json.dumps(err.to_dict()), file=sys.stderr)  # noqa: T201 — CLI output
        return 1
    except NetworkPolicyError as err:
        return _fail("network_policy_violation", str(err))
    except ValueError as err:  # includes pydantic ValidationError and SecurityConfigError
        return _fail("config_error", config_error_message(err))
    if output is None:
        sys.stdout.write(result.markdown)
        return 0
    try:
        Path(output).write_text(result.markdown, encoding="utf-8")
    except OSError as err:
        return _fail_with("report_write_failed", f"cannot write {output}: {err.strerror}", 1)
    summary = {
        "output": output,
        "window_days": result.window_days,
        "total_queries": result.metrics.total_queries,
        "knowledge_gaps": result.metrics.knowledge_gaps,
        "suggested_topics": len(result.suggested_topics),
    }
    print(json.dumps(summary))  # noqa: T201 — CLI output
    return 0


def preflight(bind_host: str | None = None) -> int:
    """Refuse to start when ``LLM_MODE=offline`` meets a non-local endpoint.

    *bind_host* (HTTP surfaces only) is also checked: offline mode cannot be
    hosted remotely. Returns 0 when startup may proceed, 2 (after printing a
    JSON error on stderr) when the network policy is violated.
    """
    from faq_core.config import get_config
    from faq_core.network_guard import (
        NetworkPolicyError,
        enforce_hosting_policy,
        enforce_network_policy,
    )

    try:
        config = get_config()
        enforce_network_policy(config)
        if bind_host is not None:
            enforce_hosting_policy(config, bind_host)
    except NetworkPolicyError as err:
        return _fail("network_policy_violation", str(err))
    except ValueError as err:  # pydantic ValidationError: a malformed .env value
        return _fail("config_error", config_error_message(err))
    return 0


def config_error_message(err: ValueError) -> str:
    """Describe a configuration error without echoing the offending value.

    pydantic's ``ValidationError`` text includes ``input_value``; only the
    setting names and problems are kept.
    """
    from pydantic import ValidationError

    if isinstance(err, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in err.errors()
        )
        return f"invalid settings: {problems}"
    return str(err)


def _fail(code: str, message: str) -> int:
    return _fail_with(code, message, 2)


def _fail_with(code: str, message: str, status: int) -> int:
    print(json.dumps({"error": {"code": code, "message": message}}), file=sys.stderr)  # noqa: T201
    return status


def serve_http(mode: str, host: str, port: int) -> int:
    """Run an HTTP surface; return 2 (JSON error on stderr) on unsafe security settings."""
    from faq_core.network_guard import NetworkPolicyError
    from faq_core.security.errors import SecurityConfigError

    try:
        if mode == "api":
            run_api(host, port)
        else:
            run_mcp_http(host, port)
    except SecurityConfigError as err:
        return _fail("security_config_error", str(err))
    except NetworkPolicyError as err:
        return _fail("network_policy_violation", str(err))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Support FAQ Assistant")
    parser.add_argument(
        "--mode",
        choices=["mcp", "api", "ui", "ingest", "report"],
        default="mcp",
        help="Server mode, 'ingest' to load --source content, or 'report' for the weekly "
        "knowledge-gap report (default: mcp)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="ingest mode: file, folder or http(s) URL (repeatable)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="mcp mode: stdio (default, local client) or http (remote, authenticated)",
    )
    parser.add_argument(
        "--window-days", type=int, help="report mode: days covered, 1-365 (default: 7)"
    )
    parser.add_argument(
        "--limit", type=int, help="report mode: maximum gap clusters listed, 1-100 (default: 10)"
    )
    parser.add_argument("--output", help="report mode: write the Markdown report to this file")
    parser.add_argument("--host", help="api / mcp-http bind address (default: BIND_HOST)")
    parser.add_argument(
        "--port", type=int, help="api / mcp-http port (default: PORT / MCP_HTTP_PORT)"
    )
    args = parser.parse_args()
    if args.mode == "ingest" and not args.source:
        parser.error("--mode ingest needs at least one --source")
    report_options = (args.window_days, args.limit, args.output)
    if args.mode != "report" and any(option is not None for option in report_options):
        parser.error("--window-days / --limit / --output apply to --mode report")
    http_mode = args.mode == "api" or (args.mode == "mcp" and args.transport == "http")
    if not http_mode and (args.host or args.port):
        parser.error("--host / --port apply to --mode api and --mode mcp --transport http")
    host, port = _bind_address(args.mode, args.host, args.port) if http_mode else (None, None)
    if (status := preflight(host)) != 0:
        sys.exit(status)

    if host is not None and port is not None:
        sys.exit(serve_http(args.mode, host, port))
    elif args.mode == "mcp":
        run_mcp()
    elif args.mode == "ingest":
        sys.exit(run_ingest(args.source))
    elif args.mode == "report":
        from faq_core.models import DEFAULT_METRICS_WINDOW_DAYS, DEFAULT_REPORT_GAPS

        window_days = DEFAULT_METRICS_WINDOW_DAYS if args.window_days is None else args.window_days
        limit = DEFAULT_REPORT_GAPS if args.limit is None else args.limit
        sys.exit(run_report(window_days, limit, args.output))
    else:
        run_ui()


def _bind_address(mode: str, host: str | None, port: int | None) -> tuple[str, int]:
    from faq_core.config import get_config

    config = get_config()
    default_port = config.PORT if mode == "api" else config.MCP_HTTP_PORT
    return (host or config.BIND_HOST), (port if port is not None else default_port)


if __name__ == "__main__":
    main()
