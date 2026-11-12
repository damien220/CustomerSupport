"""Provider presets: defaults per LLM_PROVIDER and the override order."""

from __future__ import annotations

import pytest

from faq_core.config import AppConfig
from faq_core.llm.presets import (
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    OPENAI_COMPATIBLE_DEFAULT_MAX_TOKENS,
    PRESETS,
    WireProtocol,
    api_key_for,
    get_preset,
    resolve_llm_settings,
)


def _config(**values: object) -> AppConfig:
    return AppConfig(_env_file=None, **values)  # type: ignore[arg-type]


def test_every_required_provider_has_a_preset() -> None:
    required = {
        "anthropic",
        "openai",
        "gemini",
        "groq",
        "openrouter",
        "ollama",
        "vllm",
        "llamacpp",
        "lmstudio",
    }

    assert required <= set(PRESETS)


@pytest.mark.parametrize(
    ("provider", "base_url", "model"),
    [
        ("anthropic", "https://api.anthropic.com", "claude-opus-5-5"),
        ("openai", "https://api.openai.com/v1", "gpt-6-astra"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
        ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("openrouter", "https://openrouter.ai/api/v1", "openrouter/auto"),
        ("mistral", "https://api.mistral.ai/v1", "mistral-large-latest"),
        ("cohere", "https://api.cohere.com/compatibility/v1", "command-r-plus"),
        ("together", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ],
)
def test_hosted_presets_resolve_base_url_and_default_model(
    provider: str, base_url: str, model: str
) -> None:
    resolved = resolve_llm_settings(_config(LLM_PROVIDER=provider))

    assert (resolved.base_url, resolved.model) == (base_url, model)
    assert resolved.key_required
    assert not resolved.local


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("ollama", "http://localhost:11434/v1"),
        ("vllm", "http://localhost:8000/v1"),
        ("llamacpp", "http://127.0.0.1:9931/v1"),
        ("lmstudio", "http://localhost:1234/v1"),
    ],
)
def test_local_presets_need_an_explicit_model(provider: str, base_url: str) -> None:
    resolved = resolve_llm_settings(_config(LLM_PROVIDER=provider, LLM_MODEL="qwen3:8b"))

    assert (resolved.base_url, resolved.model) == (base_url, "qwen3:8b")
    assert resolved.local
    assert not resolved.key_required
    assert resolved.protocol is WireProtocol.OPENAI
    with pytest.raises(ValueError, match="needs LLM_MODEL"):
        resolve_llm_settings(_config(LLM_PROVIDER=provider))


def test_llm_base_url_and_model_override_the_preset() -> None:
    config = _config(
        LLM_PROVIDER="openai", LLM_BASE_URL="https://proxy.example.com/v1", LLM_MODEL="my-model"
    )

    resolved = resolve_llm_settings(config)

    assert (resolved.base_url, resolved.model) == ("https://proxy.example.com/v1", "my-model")


@pytest.mark.parametrize(
    ("provider", "setting"),
    [("ollama", "OLLAMA_BASE_URL"), ("vllm", "VLLM_BASE_URL"), ("lmstudio", "LMSTUDIO_BASE_URL")],
)
def test_legacy_per_provider_base_url_is_honoured(provider: str, setting: str) -> None:
    config = _config(LLM_PROVIDER=provider, LLM_MODEL="m", **{setting: "http://gpu-box:9000/v1"})

    assert resolve_llm_settings(config).base_url == "http://gpu-box:9000/v1"


def test_llm_base_url_wins_over_the_legacy_setting() -> None:
    config = _config(
        LLM_PROVIDER="ollama",
        LLM_MODEL="m",
        OLLAMA_BASE_URL="http://gpu-box:11434/v1",
        LLM_BASE_URL="http://127.0.0.1:5000/v1",
    )

    assert resolve_llm_settings(config).base_url == "http://127.0.0.1:5000/v1"


def test_max_tokens_default_per_protocol_and_override() -> None:
    assert (
        resolve_llm_settings(_config(LLM_PROVIDER="anthropic")).max_tokens
        == ANTHROPIC_DEFAULT_MAX_TOKENS
    )
    assert (
        resolve_llm_settings(_config(LLM_PROVIDER="groq")).max_tokens
        == OPENAI_COMPATIBLE_DEFAULT_MAX_TOKENS
    )
    assert resolve_llm_settings(_config(LLM_PROVIDER="groq", LLM_MAX_TOKENS=300)).max_tokens == 300


@pytest.mark.parametrize(
    ("alias", "name"),
    [("llama.cpp", "llamacpp"), ("LM-Studio", "lmstudio"), (" Anthropic ", "anthropic")],
)
def test_provider_names_are_case_insensitive_with_aliases(alias: str, name: str) -> None:
    assert get_preset(alias).name == name


def test_unknown_provider_is_rejected_with_the_known_list() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER.*anthropic"):
        resolve_llm_settings(_config(LLM_PROVIDER="not-a-real-provider"))


def test_custom_preset_requires_a_base_url() -> None:
    with pytest.raises(ValueError, match="needs LLM_BASE_URL"):
        resolve_llm_settings(_config(LLM_PROVIDER="custom", LLM_MODEL="m"))


def test_api_key_comes_from_the_protocol_specific_setting() -> None:
    config = _config(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="a-key", LLM_API_KEY="o-key")

    assert api_key_for(config, resolve_llm_settings(config)).get_secret_value() == "a-key"
    config = _config(LLM_PROVIDER="groq", ANTHROPIC_API_KEY="a-key", LLM_API_KEY="o-key")
    assert api_key_for(config, resolve_llm_settings(config)).get_secret_value() == "o-key"


def test_missing_required_key_names_the_setting_not_a_value() -> None:
    config = _config(LLM_PROVIDER="openai")

    with pytest.raises(ValueError, match="requires LLM_API_KEY"):
        api_key_for(config, resolve_llm_settings(config))


def test_local_runtime_needs_no_key() -> None:
    config = _config(LLM_PROVIDER="ollama", LLM_MODEL="m")

    assert api_key_for(config, resolve_llm_settings(config)).get_secret_value() == ""
