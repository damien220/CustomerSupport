"""Offline network guard: local endpoints pass, remote ones are refused at startup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bootstrap import build_runtime
from faq_core.config import AppConfig
from faq_core.errors import IngestError
from faq_core.network_guard import (
    NetworkPolicyError,
    enforce_network_policy,
    is_local_url,
    offline_url_policy,
    outbound_endpoints,
    parse_allowed_hosts,
)
from faq_core.retrieval.ingest import HttpxUrlFetcher
from tests.conftest import FakeLLMClient


def _offline(tmp_path: Path, **values: object) -> AppConfig:
    settings: dict[str, object] = {
        "LLM_MODE": "offline",
        "LLM_MODEL": "llama3.2",
        "DB_PATH": str(tmp_path / "offline.db"),
        "EMBED_PROVIDER": "hashing",
        "RERANKER": "none",
    }
    settings.update(values)
    return AppConfig(_env_file=None, **settings)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _restore_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch records the original state and removes any
    # value the guard sets during the test.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:9931/v1",
        "http://127.8.0.1:8000",
        "http://[::1]:8000/v1",
        "http://ollama.localhost:11434/v1",
        "http+unix://%2Frun%2Fllm.sock/v1",
        "unix:///run/llm.sock",
    ],
)
def test_loopback_localhost_and_unix_socket_urls_are_local(url: str) -> None:
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com",
        "http://10.0.0.5:11434/v1",  # private, but not allowlisted
        "http://host.docker.internal:11434/v1",
        "http://localhost.evil.com/v1",
        "not a url",
        "",
    ],
)
def test_other_urls_are_not_local(url: str) -> None:
    assert not is_local_url(url)


def test_allowlisted_private_hosts_are_local() -> None:
    allowed = parse_allowed_hosts(" host.docker.internal , 10.0.0.5,GPU-BOX.lan ")

    assert is_local_url("http://host.docker.internal:11434/v1", allowed)
    assert is_local_url("http://10.0.0.5:11434/v1", allowed)
    assert is_local_url("http://gpu-box.lan:8000/v1", allowed)
    assert not is_local_url("http://10.0.0.6:11434/v1", allowed)


def test_public_ip_in_the_allowlist_is_rejected() -> None:
    with pytest.raises(NetworkPolicyError, match="private"):
        parse_allowed_hosts("8.8.8.8")


def test_local_fake_endpoint_passes_and_hf_hub_goes_offline(tmp_path: Path) -> None:
    config = _offline(tmp_path, LLM_BASE_URL="http://127.0.0.1:9931/v1")

    enforce_network_policy(config)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_remote_llm_endpoint_is_refused_naming_setting_and_host(tmp_path: Path) -> None:
    config = _offline(tmp_path, LLM_BASE_URL="https://user:pw@llm.example.com/v1?key=abc")

    with pytest.raises(NetworkPolicyError) as excinfo:
        enforce_network_policy(config)

    message = str(excinfo.value)
    assert "LLM_BASE_URL -> llm.example.com" in message
    assert "pw" not in message and "abc" not in message


def test_hosted_provider_preset_is_refused_in_offline_mode(tmp_path: Path) -> None:
    config = _offline(tmp_path, LLM_PROVIDER="anthropic", LLM_MODEL="")

    with pytest.raises(NetworkPolicyError, match="api.anthropic.com"):
        enforce_network_policy(config)


def test_remote_embedder_reranker_and_tracing_are_all_listed(tmp_path: Path) -> None:
    config = _offline(
        tmp_path,
        EMBED_PROVIDER="openai",
        EMBED_BASE_URL="https://api.openai.com/v1",
        RERANKER="cohere",
        LANGFUSE_SECRET_KEY="sk",  # noqa: S106 — fake value, only presence matters
        LANGFUSE_PUBLIC_KEY="pk",
    )

    with pytest.raises(NetworkPolicyError) as excinfo:
        enforce_network_policy(config)

    message = str(excinfo.value)
    assert "EMBED_BASE_URL -> api.openai.com" in message
    assert "RERANKER=cohere -> api.cohere.com" in message
    assert "LANGFUSE_HOST -> cloud.langfuse.com" in message


def test_local_ollama_embedder_is_allowed(tmp_path: Path) -> None:
    config = _offline(tmp_path, EMBED_PROVIDER="ollama")

    assert [e.setting for e in outbound_endpoints(config)] == ["LLM_BASE_URL", "OLLAMA_BASE_URL"]
    enforce_network_policy(config)


def test_allowlisted_lan_runtime_passes(tmp_path: Path) -> None:
    config = _offline(
        tmp_path, OLLAMA_BASE_URL="http://gpu-box.lan:11434/v1", OFFLINE_ALLOWED_HOSTS="gpu-box.lan"
    )

    enforce_network_policy(config)


@pytest.mark.parametrize("mode", ["api", "client"])
def test_guard_is_a_no_op_outside_offline_mode(mode: str) -> None:
    config = AppConfig(_env_file=None, LLM_MODE=mode, RERANKER="cohere")  # type: ignore[call-arg]

    enforce_network_policy(config)

    assert "HF_HUB_OFFLINE" not in os.environ


def test_build_runtime_refuses_before_opening_the_database(tmp_path: Path) -> None:
    config = _offline(tmp_path, LLM_BASE_URL="https://llm.example.com/v1")

    with pytest.raises(NetworkPolicyError):
        build_runtime(config, llm=FakeLLMClient())

    assert not (tmp_path / "offline.db").exists()


async def test_offline_ingest_refuses_remote_urls_before_fetching(tmp_path: Path) -> None:
    policy = offline_url_policy(_offline(tmp_path))
    assert policy is not None
    fetcher = HttpxUrlFetcher(url_policy=policy)

    with pytest.raises(IngestError) as excinfo:
        await fetcher.fetch("https://docs.example.com/faq")

    assert excinfo.value.details["host"] == "docs.example.com"


async def test_offline_ingest_refuses_a_redirect_to_a_remote_host(tmp_path: Path) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://cdn.example.com/faq.md"})

    policy = offline_url_policy(_offline(tmp_path))
    fetcher = HttpxUrlFetcher(url_policy=policy, transport=httpx.MockTransport(handler))

    with pytest.raises(IngestError, match="local hosts"):
        await fetcher.fetch("http://localhost:8080/faq.md")


def test_no_ingest_policy_outside_offline_mode() -> None:
    assert offline_url_policy(AppConfig(_env_file=None)) is None  # type: ignore[call-arg]


def test_cli_preflight_exits_2_with_a_json_error_for_a_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import json

    import app

    config = _offline(tmp_path, LLM_BASE_URL="https://llm.example.com/v1")
    monkeypatch.setattr("faq_core.config.get_config", lambda: config)

    assert app.preflight() == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "network_policy_violation"
    assert "llm.example.com" in error["message"]


def test_cli_preflight_passes_for_local_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import app

    config = _offline(tmp_path)
    monkeypatch.setattr("faq_core.config.get_config", lambda: config)

    assert app.preflight() == 0


# ── escalation webhook (Phase 4) ─────────────────────────────────────────────


def test_remote_escalation_webhook_is_refused_in_offline_mode(tmp_path: Path) -> None:
    config = _offline(
        tmp_path,
        ESCALATION_SINK="webhook",
        ESCALATION_WEBHOOK_URL="https://hooks.example.com/support?token=abc",
        ESCALATION_WEBHOOK_SECRET="signing-secret",  # noqa: S106 — fake value
    )

    with pytest.raises(NetworkPolicyError) as excinfo:
        build_runtime(config, llm=FakeLLMClient())

    message = str(excinfo.value)
    assert "ESCALATION_WEBHOOK_URL -> hooks.example.com" in message
    assert "abc" not in message and "signing-secret" not in message
    assert not (tmp_path / "offline.db").exists()


def test_webhook_url_is_checked_even_when_the_sink_is_log(tmp_path: Path) -> None:
    config = _offline(tmp_path, ESCALATION_WEBHOOK_URL="https://hooks.example.com/x")

    assert "ESCALATION_WEBHOOK_URL" in [e.setting for e in outbound_endpoints(config)]
    with pytest.raises(NetworkPolicyError, match="hooks.example.com"):
        enforce_network_policy(config)


def test_local_escalation_webhook_passes_offline(tmp_path: Path) -> None:
    config = _offline(
        tmp_path,
        ESCALATION_SINK="webhook",
        ESCALATION_WEBHOOK_URL="http://127.0.0.1:9999/hook",
        ESCALATION_WEBHOOK_SECRET="signing-secret",  # noqa: S106 — fake value
    )

    enforce_network_policy(config)


# ── remote access (Phase 5) ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("setting", "url"),
    [
        ("OAUTH_ISSUER", "https://auth.example.com"),
        ("OAUTH_JWKS_URL", "https://auth.example.com/.well-known/jwks.json"),
    ],
)
def test_remote_oauth_endpoints_are_refused_offline(tmp_path: Path, setting: str, url: str) -> None:
    config = _offline(tmp_path, **{setting: url})

    with pytest.raises(NetworkPolicyError, match=f"{setting} -> auth.example.com"):
        enforce_network_policy(config)


def test_local_oauth_endpoints_pass_offline(tmp_path: Path) -> None:
    config = _offline(
        tmp_path,
        OAUTH_ISSUER="http://localhost:8080/realms/support",
        OAUTH_JWKS_URL="http://127.0.0.1:8080/realms/support/certs",
    )

    enforce_network_policy(config)


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "::", "203.0.113.7", "mcp.example.com"])  # noqa: S104
def test_offline_mode_cannot_be_hosted_remotely(tmp_path: Path, bind_host: str) -> None:
    from faq_core.network_guard import enforce_hosting_policy

    with pytest.raises(NetworkPolicyError, match="cannot be hosted remotely"):
        enforce_hosting_policy(_offline(tmp_path), bind_host)


@pytest.mark.parametrize(
    "values",
    [
        {"OAUTH_RESOURCE_URL": "https://mcp.example.com/mcp"},
        {"ALLOWED_HOSTS": "mcp.example.com"},
    ],
)
def test_offline_mode_refuses_public_resource_and_host_names(
    tmp_path: Path, values: dict[str, str]
) -> None:
    from faq_core.network_guard import enforce_hosting_policy

    with pytest.raises(NetworkPolicyError, match="mcp.example.com"):
        enforce_hosting_policy(_offline(tmp_path, **values), "127.0.0.1")


def test_offline_mode_may_serve_loopback_or_an_allowlisted_private_ip(tmp_path: Path) -> None:
    from faq_core.network_guard import enforce_hosting_policy

    enforce_hosting_policy(_offline(tmp_path), "127.0.0.1")
    enforce_hosting_policy(_offline(tmp_path, OFFLINE_ALLOWED_HOSTS="10.0.0.5"), "10.0.0.5")


def test_hosting_policy_is_a_no_op_outside_offline_mode() -> None:
    from faq_core.network_guard import enforce_hosting_policy

    enforce_hosting_policy(AppConfig(_env_file=None, LLM_MODE="client"), "0.0.0.0")  # type: ignore[call-arg, arg-type]  # noqa: S104
