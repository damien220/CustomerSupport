from __future__ import annotations

from faq_core.config import AppConfig
from observability.tracing import init_langfuse, observe


async def test_observe_decorator_passes_through_result() -> None:
    @observe(name="test.op")
    async def add(a: int, b: int) -> int:
        return a + b

    result = await add(2, 3)

    assert result == 5


async def test_observe_decorator_propagates_exceptions() -> None:
    @observe(name="test.op")
    async def boom() -> None:
        raise ValueError("boom")

    try:
        await boom()
    except ValueError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected ValueError to propagate")


def test_init_langfuse_is_noop_with_blank_keys(app_config: AppConfig) -> None:
    init_langfuse(app_config)  # must not raise, regardless of keys
