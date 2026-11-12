"""In-process TTL cache with async, per-key stampede protection.

:class:`MemoryTTLCache` is a small bounded TTL map (LRU eviction). Its ``get``/
``set`` are synchronous on purpose: they touch only in-process memory, so there
is nothing to await. :class:`AsyncStampedeCache` layers per-key
:class:`asyncio.Lock` serialisation on top, so concurrent misses for the same
key run the expensive coroutine exactly once.

Any object with the :class:`CacheBackend` shape can replace the memory backend
(e.g. a future shared cache) without touching callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol


class CacheBackend[V](Protocol):
    """Structural shape of a synchronous key/value store with per-entry TTL."""

    def get(self, key: str) -> V | None: ...

    def set(self, key: str, value: V, ttl: float | None = None) -> None: ...

    def clear(self) -> None: ...


class MemoryTTLCache[V]:
    """Bounded in-memory TTL cache with least-recently-used eviction.

    Args:
        max_entries: Entry count above which the least-recently-used entry is evicted.
        clock: Monotonic time source in seconds (injectable for tests).
    """

    def __init__(
        self, max_entries: int = 1024, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float | None, V]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> V | None:
        """Return the live value for *key*, or ``None`` when absent or expired."""
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at is not None and self._clock() >= expires_at:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: V, ttl: float | None = None) -> None:
        """Store *value* under *key*; ``ttl=None`` means no expiry."""
        expires_at = None if ttl is None else self._clock() + ttl
        self._data[key] = (expires_at, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class AsyncStampedeCache[V]:
    """Serialise async cache misses per key over a :class:`CacheBackend`.

    Args:
        backend: The backing synchronous store.
        ttl: Time-to-live in seconds applied to every stored value.
    """

    def __init__(self, backend: CacheBackend[V], ttl: float) -> None:
        self._backend = backend
        self._ttl = ttl
        self._locks: dict[str, _KeyLock] = {}

    @staticmethod
    def make_key(*parts: object) -> str:
        """Build a short, collision-resistant key from arbitrary parts."""
        raw = "|".join(str(part) for part in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get_or_compute(self, key: str, compute: Callable[[], Awaitable[V]]) -> V:
        """Return the cached value for *key*, computing and storing it on a miss.

        Concurrent callers that miss the same key share one lock, so *compute*
        is awaited exactly once and the others read the freshly cached value.
        A failing *compute* caches nothing; the error propagates to its caller
        and the next queued waiter retries.
        """
        cached = self._backend.get(key)
        if cached is not None:
            return cached

        # Single event-loop thread: there is no await between the lookup and
        # the insert below, so the lock map needs no lock of its own.
        key_lock = self._locks.get(key)
        if key_lock is None:
            key_lock = self._locks[key] = _KeyLock()
        key_lock.users += 1
        try:
            async with key_lock.lock:
                # Re-check under the lock: a racer may have filled it meanwhile.
                cached = self._backend.get(key)
                if cached is not None:
                    return cached
                value = await compute()
                self._backend.set(key, value, ttl=self._ttl)
                return value
        finally:
            # Forget the lock once nobody holds or awaits it, so the map stays
            # bounded by the number of in-flight keys.
            key_lock.users -= 1
            if key_lock.users == 0:
                del self._locks[key]

    def clear(self) -> None:
        """Drop every cached value (the knowledge base changed).

        A computation already in flight still stores its (possibly stale)
        result; that only happens while an add/ingest races a query.
        """
        self._backend.clear()

    @property
    def in_flight_keys(self) -> int:
        """Number of keys that currently have a caller holding or awaiting a lock."""
        return len(self._locks)


@dataclass(slots=True)
class _KeyLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
