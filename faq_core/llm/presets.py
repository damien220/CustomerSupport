"""LLM provider presets selected by ``LLM_PROVIDER``.

A preset supplies the base URL, default model, wire protocol, whether an API
key is required and whether the endpoint is local. Explicit settings win over
the preset: ``LLM_BASE_URL`` > the provider's legacy base-URL setting
(``OLLAMA_BASE_URL`` / ``VLLM_BASE_URL`` / ``LMSTUDIO_BASE_URL``) > preset;
``LLM_MODEL`` > preset; ``LLM_MAX_TOKENS`` (> 0) > preset.

Verification (2026-10-30, against the vendors' current docs / live endpoints):

* anthropic — ``claude-opus-5-5`` from the Claude API reference (cached
  2026-10-06); base URL is the SDK default.
* openai — ``gpt-6-astra`` listed as the "not sure where to start" model at
  developers.openai.com/api/docs/models.
* gemini — base URL and ``gemini-3.8-flash`` from
  ai.google.dev/gemini-api/docs/openai.
* groq — base URL from console.groq.com/docs/openai; ``llama-3.3-70b-versatile``
  listed as a production model at console.groq.com/docs/models.
* openrouter — base URL from openrouter.ai/docs/quickstart; ``openrouter/auto``
  present in the live ``GET /api/v1/models`` list.
* ollama — ``http://localhost:11434/v1`` from Ollama's OpenAI-compatibility doc.
* llamacpp — ``llama-server`` listens on 127.0.0.1:9931 by default (current
  llama.cpp server README; older builds used 8080).
* lmstudio — ``http://localhost:1234/v1`` from lmstudio.ai OpenAI-compat docs.
* vllm — ``http://localhost:8000/v1`` is vLLM's long-standing default; NOT
  re-verified this session (the docs page did not state it).
* mistral / cohere / together — kept from the pre-Phase-3 table so existing
  ``.env`` files keep working; default models NOT re-verified.

Local runtimes have no default model: ``LLM_MODEL`` must name the model the
runtime has loaded (``ollama pull <model>``, ``vllm serve <model>``, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import SecretStr

    from faq_core.config import AppConfig


class WireProtocol(StrEnum):
    """The HTTP API a provider speaks — decides which gateway adapter is built."""

    ANTHROPIC = "anthropic"  # Messages API via the official SDK
    OPENAI = "openai"  # OpenAI-compatible POST {base_url}/chat/completions


ANTHROPIC_DEFAULT_MAX_TOKENS: Final = 16_000
"""Room for adaptive thinking plus the answer (non-streaming request)."""

OPENAI_COMPATIBLE_DEFAULT_MAX_TOKENS: Final = 1024
"""Conservative: local runtimes reject ``max_tokens`` beyond their context."""


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Defaults for one ``LLM_PROVIDER`` value."""

    name: str
    protocol: WireProtocol
    base_url: str  # "" = LLM_BASE_URL must be set
    default_model: str  # "" = LLM_MODEL must be set
    key_required: bool
    local: bool  # runs on the operator's own machine/network
    default_max_tokens: int = OPENAI_COMPATIBLE_DEFAULT_MAX_TOKENS
    base_url_setting: str | None = None  # legacy AppConfig field holding the base URL


def _hosted(name: str, base_url: str, default_model: str) -> ProviderPreset:
    return ProviderPreset(
        name=name,
        protocol=WireProtocol.OPENAI,
        base_url=base_url,
        default_model=default_model,
        key_required=True,
        local=False,
    )


def _local(name: str, base_url: str, base_url_setting: str | None = None) -> ProviderPreset:
    return ProviderPreset(
        name=name,
        protocol=WireProtocol.OPENAI,
        base_url=base_url,
        default_model="",
        key_required=False,
        local=True,
        base_url_setting=base_url_setting,
    )


PRESETS: Final[Mapping[str, ProviderPreset]] = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            ProviderPreset(
                name="anthropic",
                protocol=WireProtocol.ANTHROPIC,
                base_url="https://api.anthropic.com",
                default_model="claude-opus-5-5",
                key_required=True,
                local=False,
                default_max_tokens=ANTHROPIC_DEFAULT_MAX_TOKENS,
            ),
            _hosted("openai", "https://api.openai.com/v1", "gpt-6-astra"),
            _hosted(
                "gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "gemini-3.8-flash",
            ),
            _hosted("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            _hosted("openrouter", "https://openrouter.ai/api/v1", "openrouter/auto"),
            # Pre-Phase-3 providers, kept for existing .env files (models unverified).
            _hosted("mistral", "https://api.mistral.ai/v1", "mistral-large-latest"),
            _hosted("cohere", "https://api.cohere.com/compatibility/v1", "command-r-plus"),
            _hosted(
                "together", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            ),
            _local("ollama", "http://localhost:11434/v1", "OLLAMA_BASE_URL"),
            _local("vllm", "http://localhost:8000/v1", "VLLM_BASE_URL"),
            _local("llamacpp", "http://127.0.0.1:9931/v1"),
            _local("lmstudio", "http://localhost:1234/v1", "LMSTUDIO_BASE_URL"),
            # Any other OpenAI-compatible server: set LLM_BASE_URL and LLM_MODEL.
            ProviderPreset(
                name="custom",
                protocol=WireProtocol.OPENAI,
                base_url="",
                default_model="",
                key_required=False,
                local=False,
            ),
        )
    }
)

_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "claude": "anthropic",
        "google": "gemini",
        "llama.cpp": "llamacpp",
        "llama-cpp": "llamacpp",
        "llama_cpp": "llamacpp",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "openai-compatible": "custom",
        "openai_compatible": "custom",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedLLM:
    """The effective LLM settings after applying overrides. Holds no secrets."""

    provider: str
    protocol: WireProtocol
    base_url: str
    model: str
    max_tokens: int
    key_required: bool
    local: bool


def get_preset(provider: str) -> ProviderPreset:
    """Return the preset for an ``LLM_PROVIDER`` value (case-insensitive, aliases ok).

    Raises:
        ValueError: unknown provider.
    """
    key = provider.strip().lower()
    preset = PRESETS.get(_ALIASES.get(key, key))
    if preset is None:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown LLM_PROVIDER: {key!r} (known: {known})")
    return preset


def resolve_llm_settings(config: AppConfig) -> ResolvedLLM:
    """Apply ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_MAX_TOKENS`` over the preset.

    Raises:
        ValueError: unknown provider, or no base URL / model after overrides.
    """
    preset = get_preset(config.LLM_PROVIDER)
    base_url = config.LLM_BASE_URL.strip()
    if not base_url and preset.base_url_setting is not None:
        base_url = str(getattr(config, preset.base_url_setting)).strip()
    base_url = base_url or preset.base_url
    if not base_url:
        raise ValueError(f"LLM_PROVIDER={preset.name} needs LLM_BASE_URL")
    model = config.LLM_MODEL.strip() or preset.default_model
    if not model:
        raise ValueError(
            f"LLM_PROVIDER={preset.name} needs LLM_MODEL (the model the server has loaded)"
        )
    max_tokens = config.LLM_MAX_TOKENS if config.LLM_MAX_TOKENS > 0 else preset.default_max_tokens
    return ResolvedLLM(
        provider=preset.name,
        protocol=preset.protocol,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        key_required=preset.key_required,
        local=preset.local,
    )


def api_key_for(config: AppConfig, resolved: ResolvedLLM) -> SecretStr:
    """The key setting a provider uses: ``ANTHROPIC_API_KEY`` or ``LLM_API_KEY``.

    Raises:
        ValueError: the provider requires a key and none is configured (the
            message names the setting, never a value).
    """
    if resolved.protocol is WireProtocol.ANTHROPIC:
        setting, key = "ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY
    else:
        setting, key = "LLM_API_KEY", config.LLM_API_KEY
    if resolved.key_required and not key.get_secret_value():
        raise ValueError(f"LLM_PROVIDER={resolved.provider} requires {setting}")
    return key
