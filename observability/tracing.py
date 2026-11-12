from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from observability.log import get_logger

if TYPE_CHECKING:
    from faq_core.config import AppConfig

_log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Try to import Langfuse; if unavailable we fall back to a no-op decorator so
# all @observe call sites work without branching. `langfuse.decorators` was
# removed in the v4 SDK redesign — `observe` now lives at the package root.
try:
    from langfuse import get_client as _langfuse_get_client
    from langfuse import observe as _langfuse_observe

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

_tracing_enabled = False


def init_langfuse(config: AppConfig) -> None:
    """Initialise Langfuse from AppConfig. No-op when keys are not set."""
    global _tracing_enabled

    secret = config.LANGFUSE_SECRET_KEY.get_secret_value()
    public = config.LANGFUSE_PUBLIC_KEY.get_secret_value()

    if not (_LANGFUSE_AVAILABLE and secret and public):
        return

    import os

    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public)
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)
    _tracing_enabled = True


def observe(name: str) -> Callable[[F], F]:
    """Decorator that traces a function via Langfuse when tracing is enabled.

    Dispatch is decided once, at decoration time — call sites (in
    faq_core/llm/*.py, faq_core/retrieval/*.py) are only decorated after
    init_langfuse() has run during bootstrap (see server.py / app.py), so
    `_tracing_enabled` correctly reflects whether real keys were provided.
    Without this gate, langfuse.observe() logs an "Authentication error" on
    every call when unconfigured instead of staying quiet as documented in
    .env.example ("leave blank to disable tracing in dev").
    """

    def decorator(fn: F) -> F:
        if _LANGFUSE_AVAILABLE and _tracing_enabled:
            return _langfuse_observe(name=name)(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def push_evaluation_scores(scores: dict[str, float], *, trace_id: str | None = None) -> None:
    """Push named evaluation scores (e.g. RAGAS metrics) to Langfuse.

    No-op when tracing is disabled — same gate as :func:`observe`, so calling
    this from ``tests/test_ragas.py`` is safe whether or not Langfuse keys are
    configured.
    """
    if not (_LANGFUSE_AVAILABLE and _tracing_enabled):
        return

    client = _langfuse_get_client()
    for name, value in scores.items():
        client.create_score(name=name, value=value, trace_id=trace_id)


type MetadataValue = str | int | float | bool | None


def push_metric_scores(
    scores: Mapping[str, float | None],
    *,
    name: str = "support.resolution_metrics",
    metadata: Mapping[str, MetadataValue] | None = None,
) -> int:
    """Push aggregate metric scores (e.g. resolution metrics) to Langfuse.

    Opens one ``evaluator`` observation named *name* (with *metadata*: plain
    scalars only — never query text or credentials) and attaches each score to
    its trace. ``None`` values (rates with no denominator) are skipped. Same
    gate as :func:`observe`: a no-op returning 0 when tracing is disabled.
    Telemetry never breaks the caller — any SDK failure is logged (exception
    type only) and reported as 0.

    Returns:
        The number of scores pushed.
    """
    if not (_LANGFUSE_AVAILABLE and _tracing_enabled):
        return 0
    values = {key: float(value) for key, value in scores.items() if value is not None}
    if not values:
        return 0
    try:
        span = _langfuse_get_client().start_observation(
            name=name, as_type="evaluator", metadata=dict(metadata or {})
        )
        try:
            for key, value in values.items():
                span.score_trace(name=key, value=value)
        finally:
            span.end()
    except Exception as err:  # noqa: BLE001 — telemetry must never fail a request
        _log.warning("langfuse metric push failed", extra={"error_type": type(err).__name__})
        return 0
    return len(values)


def flush_tracing() -> None:
    """Send buffered Langfuse events now (short-lived CLI runs); no-op when disabled."""
    if not (_LANGFUSE_AVAILABLE and _tracing_enabled):
        return
    try:
        _langfuse_get_client().flush()
    except Exception as err:  # noqa: BLE001 — telemetry must never fail the command
        _log.warning("langfuse flush failed", extra={"error_type": type(err).__name__})
