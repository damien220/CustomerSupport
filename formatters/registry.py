"""Name → formatter registry (replaces the former ``plugin_loader`` discovery).

Formatters are registered explicitly — there are two of them, so directory
scanning and a plugin lifecycle bought nothing but import-time cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatters.markdown_formatter import MarkdownFormatter
from formatters.plain_formatter import PlainFormatter

if TYPE_CHECKING:
    from formatters.base import Formatter


class UnknownFormatterError(KeyError):
    """No formatter is registered under the requested name."""


class FormatterRegistry:
    """Holds formatters by their ``name``."""

    def __init__(self) -> None:
        self._formatters: dict[str, Formatter] = {}

    def register(self, formatter: Formatter, *, replace: bool = False) -> None:
        """Add *formatter*; refuse to shadow an existing name unless *replace*."""
        if formatter.name in self._formatters and not replace:
            raise ValueError(f"formatter {formatter.name!r} is already registered")
        self._formatters[formatter.name] = formatter

    def get(self, name: str) -> Formatter:
        try:
            return self._formatters[name]
        except KeyError:
            known = ", ".join(sorted(self._formatters)) or "none"
            raise UnknownFormatterError(f"unknown formatter {name!r} (known: {known})") from None

    def names(self) -> list[str]:
        return sorted(self._formatters)


def default_registry() -> FormatterRegistry:
    """Return a registry holding the built-in Markdown and plain formatters."""
    registry = FormatterRegistry()
    registry.register(MarkdownFormatter())
    registry.register(PlainFormatter())
    return registry
