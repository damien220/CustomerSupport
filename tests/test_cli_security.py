"""CLI fail-closed behaviour of the HTTP surfaces (no server is started)."""

from __future__ import annotations

import json

import pytest

import app
from tests.conftest import make_api_token, security_config


def _never_serve(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("uvicorn must not start")


@pytest.mark.parametrize("mode", ["api", "mcp"])
@pytest.mark.parametrize(
    ("overrides", "bind_host", "match"),
    [
        ({"AUTH_MODE": "none"}, "0.0.0.0", "loopback"),  # noqa: S104
        ({"AUTH_MODE": "token", "API_KEYS": ""}, "127.0.0.1", "API_KEYS"),
        ({"AUTH_MODE": "token", "API_KEYS": "k:short"}, "127.0.0.1", "at least 32"),
        ({"AUTH_MODE": "oauth"}, "127.0.0.1", "OAUTH_ISSUER"),
    ],
)
def test_unsafe_auth_settings_exit_2_before_serving(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    overrides: dict[str, str],
    bind_host: str,
    match: str,
) -> None:
    config = security_config(**overrides)
    monkeypatch.setattr("faq_core.config.get_config", lambda: config)
    monkeypatch.setattr("uvicorn.run", _never_serve)

    status = app.serve_http(mode, bind_host, 18000)

    error = json.loads(capsys.readouterr().err)["error"]
    assert status == 2
    assert error["code"] == "security_config_error"
    assert match in error["message"]


def test_offline_mode_on_a_public_bind_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = security_config(
        LLM_MODE="offline", LLM_MODEL="llama3", API_KEYS=f"k:{make_api_token()}"
    )
    monkeypatch.setattr("faq_core.config.get_config", lambda: config)

    status = app.preflight("0.0.0.0")  # noqa: S104

    assert status == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "network_policy_violation"


def test_a_token_mode_api_starts_uvicorn_with_redacted_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    config = security_config(API_KEYS=f"k:{make_api_token()}")
    monkeypatch.setattr("faq_core.config.get_config", lambda: config)
    monkeypatch.setattr("uvicorn.run", lambda _app, **kwargs: calls.append(kwargs))

    assert app.serve_http("mcp", "127.0.0.1", 18001) == 0
    assert calls == [{"host": "127.0.0.1", "port": 18001, "log_config": None}]
