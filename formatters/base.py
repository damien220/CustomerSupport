"""``Formatter`` — the interface for rendering a search result as text.

Formatters are stateless and synchronous: they only assemble strings from an
already-resolved :class:`~faq_core.models.SearchResult`, with no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from faq_core.models import SearchResult


class Formatter(Protocol):
    """Renders a :class:`SearchResult` into a surface-specific string."""

    @property
    def name(self) -> str:
        """Registry key, e.g. ``"markdown"``."""
        ...

    def format(self, result: SearchResult) -> str: ...
