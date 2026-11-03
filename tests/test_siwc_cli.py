"""Проверка готовности не должна путаться с локальным status."""

from types import SimpleNamespace

import pytest

from codex_as_api import siwc_cli
from codex_as_api.auth import ChatGPTOAuthMissingError


def test_legacy_status_reads_only_its_explicit_credentials(monkeypatch, capsys):
    paths = []
    monkeypatch.setattr(siwc_cli.auth, "load_token_data",
                        lambda path: paths.append(path) or SimpleNamespace(expired=False))
    assert siwc_cli.main(["status", "--auth-path", "legacy.json"], backend="codex") == 0
    assert paths == ["legacy.json"]
    assert '"oauth_profile": "codex"' in capsys.readouterr().out


def test_legacy_preflight_uses_classic_provider(monkeypatch, capsys):
    import codex_as_api.provider as provider_module

    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex",
            get_model_catalog=lambda: SimpleNamespace(models=(1,), model=lambda slug: SimpleNamespace(slug=slug)))
    monkeypatch.setattr(provider_module, "ChatGPTOAuthProvider", factory)
    assert siwc_cli.main(["preflight", "--auth-path", "legacy.json"], backend="codex") == 0
    assert calls[0]["backend"] == "codex"
    assert calls[0]["auth_json_path"] == "legacy.json"
    assert '"backend": "codex"' in capsys.readouterr().out


def test_status_is_local_even_when_login_is_required(monkeypatch, capsys):
    monkeypatch.setattr(siwc_cli.siwc_auth, "read_status", lambda path: {"state": "login_required"})
    monkeypatch.setattr(siwc_cli, "_preflight", lambda path: pytest.fail("Status вызвал сеть"))
    assert siwc_cli.main(["status"]) == 0
    assert "login_required" in capsys.readouterr().out


def test_preflight_rejects_missing_credentials_without_exposing_error(monkeypatch, capsys):
    import codex_as_api.provider as provider_module

    def unavailable(**kwargs):
        raise ChatGPTOAuthMissingError("synthetic-private-value")

    monkeypatch.setattr(provider_module, "ChatGPTOAuthProvider", unavailable)
    assert siwc_cli.main(["preflight"]) == 1
    captured = capsys.readouterr()
    assert "synthetic-private-value" not in captured.out + captured.err


def test_preflight_reads_catalog_and_validates_configured_model(monkeypatch, capsys):
    import codex_as_api.provider as provider_module

    calls = []
    catalog = SimpleNamespace(
        models=(object(),),
        model=lambda slug: calls.append(("model", slug)) or SimpleNamespace(slug=slug),
    )

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            base_url="https://api.openai.com/v1",
            get_model_catalog=lambda: calls.append("catalog") or catalog,
        )

    monkeypatch.setattr(provider_module, "ChatGPTOAuthProvider", factory)
    monkeypatch.setenv("CODEX_AS_API_MODEL", "gpt-6.1-sol")
    assert siwc_cli.main(["preflight", "--auth-path", "synthetic.json"]) == 0
    assert calls == [
        {"model": "gpt-6.1-sol", "backend": "siwc", "auth_json_path": "synthetic.json"},
        "catalog", ("model", "gpt-6.1-sol"),
    ]
    assert '"backend": "siwc"' in capsys.readouterr().out
