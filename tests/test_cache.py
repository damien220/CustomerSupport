from __future__ import annotations

import asyncio

import pytest

from faq_core.cache import AsyncStampedeCache, MemoryTTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def cache() -> AsyncStampedeCache[str]:
    return AsyncStampedeCache(MemoryTTLCache(max_entries=100), ttl=300)


# ── AsyncStampedeCache ───────────────────────────────────────────────────────


async def test_get_or_compute_returns_computed_value_on_miss(
    cache: AsyncStampedeCache[str],
) -> None:
    async def compute() -> str:
        return "computed"

    assert await cache.get_or_compute("k", compute) == "computed"


async def test_get_or_compute_caches_after_first_miss(cache: AsyncStampedeCache[str]) -> None:
    calls = 0

    async def compute() -> str:
        nonlocal calls
        calls += 1
        return "value"

    first = await cache.get_or_compute("k", compute)
    second = await cache.get_or_compute("k", compute)

    assert first == second == "value"
    assert calls == 1


async def test_distinct_keys_compute_independently(cache: AsyncStampedeCache[str]) -> None:
    await cache.get_or_compute("a", lambda: _const("A"))

    assert await cache.get_or_compute("b", lambda: _const("B")) == "B"


async def test_concurrent_misses_compute_once(cache: AsyncStampedeCache[str]) -> None:
    calls = 0
    release = asyncio.Event()

    async def compute() -> str:
        nonlocal calls
        calls += 1
        await release.wait()
        return "shared"

    tasks = [asyncio.create_task(cache.get_or_compute("k", compute)) for _ in range(5)]
    await asyncio.sleep(0.01)  # let all tasks reach the lock / cache check
    release.set()
    results = await asyncio.gather(*tasks)

    assert results == ["shared"] * 5
    assert calls == 1
    assert cache.in_flight_keys == 0  # per-key locks are released afterwards


async def test_failed_compute_is_not_cached_and_can_be_retried(
    cache: AsyncStampedeCache[str],
) -> None:
    async def boom() -> str:
        raise RuntimeError("upstream down")

    with pytest.raises(RuntimeError):
        await cache.get_or_compute("k", boom)

    assert await cache.get_or_compute("k", lambda: _const("ok")) == "ok"
    assert cache.in_flight_keys == 0


async def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache: AsyncStampedeCache[str] = AsyncStampedeCache(MemoryTTLCache(clock=clock), ttl=10)
    await cache.get_or_compute("k", lambda: _const("old"))

    clock.now += 11

    assert await cache.get_or_compute("k", lambda: _const("new")) == "new"


def test_make_key_is_deterministic_and_order_sensitive() -> None:
    assert AsyncStampedeCache.make_key("search", "q", 3) == AsyncStampedeCache.make_key(
        "search", "q", 3
    )
    assert AsyncStampedeCache.make_key("search", "q", 3) != AsyncStampedeCache.make_key(
        "search", "q", 5
    )


# ── MemoryTTLCache ───────────────────────────────────────────────────────────


def test_memory_cache_evicts_least_recently_used() -> None:
    store: MemoryTTLCache[int] = MemoryTTLCache(max_entries=2)
    store.set("a", 1)
    store.set("b", 2)
    store.get("a")  # touch "a" so "b" becomes least recently used

    store.set("c", 3)

    assert (store.get("a"), store.get("b"), store.get("c")) == (1, None, 3)
    assert len(store) == 2


def test_memory_cache_without_ttl_never_expires() -> None:
    clock = FakeClock()
    store: MemoryTTLCache[str] = MemoryTTLCache(clock=clock)
    store.set("k", "v")

    clock.now += 10**9

    assert store.get("k") == "v"


def test_memory_cache_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        MemoryTTLCache(max_entries=0)


async def _const(value: str) -> str:
    return value
