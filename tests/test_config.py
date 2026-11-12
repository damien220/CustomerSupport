from __future__ import annotations

import pytest
from pydantic import ValidationError

from faq_core.config import AppConfig, get_config
from faq_core.llm.presets import resolve_llm_settings
from faq_core.models import LLMMode


def test_app_config_defaults_to_api_mode_and_anthropic_provider() -> None:
    config = AppConfig(_env_file=None, ANTHROPIC_API_KEY="k")

    assert config.LLM_MODE is LLMMode.API
    assert config.LLM_PROVIDER == "anthropic"
    assert config.LLM_MODEL == ""  # blank = the preset's default model
    assert resolve_llm_settings(config).model == "claude-opus-5-5"


@pytest.mark.parametrize("raw", ["client", "CLIENT", " Client "])
def test_llm_mode_is_parsed_case_insensitively(raw: str) -> None:
    config = AppConfig(_env_file=None, LLM_MODE=raw)

    assert config.LLM_MODE is LLMMode.CLIENT


def test_unknown_llm_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, LLM_MODE="hybrid")


def test_offline_mode_defaults_to_local_llm_embedder_and_reranker() -> None:
    config = AppConfig(_env_file=None, LLM_MODE="offline")

    assert (config.LLM_PROVIDER, config.EMBED_PROVIDER, config.RERANKER) == (
        "ollama",
        "local",
        "bge",
    )


def test_offline_mode_keeps_explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_PROVIDER", "hashing")

    config = AppConfig(_env_file=None, LLM_MODE="offline", LLM_PROVIDER="vllm", RERANKER="none")

    assert (config.LLM_PROVIDER, config.EMBED_PROVIDER, config.RERANKER) == (
        "vllm",
        "hashing",
        "none",
    )


def test_other_modes_keep_lite_retrieval_defaults() -> None:
    config = AppConfig(_env_file=None, LLM_MODE="client")

    assert (config.LLM_PROVIDER, config.EMBED_PROVIDER, config.RERANKER) == (
        "anthropic",
        "hashing",
        "none",
    )


def test_app_config_reads_offline_base_urls_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")

    config = AppConfig(ANTHROPIC_API_KEY="k")

    assert config.OLLAMA_BASE_URL == "http://gpu-box:11434/v1"


def test_app_config_secrets_are_not_plain_strings() -> None:
    config = AppConfig(_env_file=None, ANTHROPIC_API_KEY="super-secret")

    assert "super-secret" not in repr(config.ANTHROPIC_API_KEY)
    assert config.ANTHROPIC_API_KEY.get_secret_value() == "super-secret"


def test_get_config_is_a_cached_singleton() -> None:
    get_config.cache_clear()
    try:
        first = get_config()
        second = get_config()
        assert first is second
    finally:
        get_config.cache_clear()


def test_secret_values_lists_only_non_empty_secrets() -> None:
    config = AppConfig(_env_file=None, ANTHROPIC_API_KEY="super-secret", LLM_API_KEY="")

    assert config.secret_values() == ["super-secret"]


def test_retrieval_defaults_are_lite_and_torch_free() -> None:
    config = AppConfig(_env_file=None)  # type: ignore[call-arg]

    assert (config.EMBED_PROVIDER, config.VECTOR_STORE, config.RERANKER) == (
        "hashing",
        "sqlite-vec",
        "none",
    )
    assert config.RETRIEVAL_RRF_K == 60
    assert config.RERANK_API_KEY.get_secret_value() == ""
