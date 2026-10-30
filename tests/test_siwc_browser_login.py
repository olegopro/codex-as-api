"""Общий browser loop проверяется с явным SIWC-профилем без внешней сети."""
import asyncio
import base64
import hashlib
import json
import threading
import urllib.request
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from codex_as_api import auth, siwc_auth, siwc_cli
from codex_as_api.browser_login import BrowserLoginManager

from . import test_siwc_auth as helpers
from .test_browser_login import callback


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Незамоканный внешний HTTP-запрос запрещён")

    async def async_forbidden(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(auth, "_urlopen_no_redirect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_forbidden)


@pytest.fixture()
def sign(monkeypatch):
    return helpers.create_keys(monkeypatch)


def test_initial_registration_uses_shared_loopback_pkce_and_verifies_nonce(tmp_path, sign):
    async def scenario():
        path = tmp_path / "siwc" / "credentials.json"
        calls, authorization = [], {}

        async def transport(url, body, headers, timeout):
            calls.append((url, parse_qs(body.decode()), headers, timeout))
            return 200, json.dumps({"access_token": "offline-access", "refresh_token": "offline-refresh",
                "id_token": sign(nonce=authorization["nonce"][0]), "scope": siwc_auth.SCOPES,
                "expires_in": 3600}).encode()

        manager = siwc_auth.make_browser_login_manager(path, transport=transport)
        assert type(manager) is BrowserLoginManager
        try:
            started = await manager.start()
            parsed = urlsplit(started["auth_url"])
            assert parsed.scheme + "://" + parsed.netloc + parsed.path == siwc_auth.AUTHORIZE_URL
            authorization.update(parse_qs(parsed.query))
            assert authorization["client_id"] == [siwc_auth.DYNAMIC_CLIENT_ID]
            assert authorization["agent_name_hint"] == [siwc_auth.AGENT_NAME]
            assert authorization["scope"] == [siwc_auth.SCOPES]
            assert authorization["resource"] == [siwc_auth.RESOURCE]
            assert authorization["nonce"] and authorization["state"] and authorization["ext_agent_host_id"]
            assert "originator" not in authorization and "codex_cli_simplified_flow" not in authorization
            redirect = urlsplit(authorization["redirect_uri"][0])
            assert redirect.hostname == "127.0.0.1" and redirect.port > 0 and redirect.path == "/auth/callback"
            for target, host in [
                ("/other?state=" + authorization["state"][0] + "&code=x&client_id=issued-offline-client", None),
                ("/auth/callback?state=wrong&code=x&client_id=issued-offline-client", None),
                ("/auth/callback?state=" + authorization["state"][0] + "&code=x", "localhost:" + str(redirect.port)),
            ]:
                response = await callback(redirect.port, target, host=host)
                assert b"400 Bad Request" in response
                assert manager.get(started["id"])["status"] == "pending"
            response = await callback(redirect.port, "/auth/callback?" + urlencode({
                "state": authorization["state"][0], "code": "offline-code", "client_id": "issued-offline-client"}))
            assert b"200 OK" in response
            assert manager.get(started["id"])["status"] == "succeeded"
            token = auth.load_token_data(path)
            assert token.oauth_profile == "siwc" and token.account_id == "offline-subject"
            assert len(calls) == 1
            url, exchange, headers, timeout = calls[0]
            assert url == siwc_auth.TOKEN_URL
            assert headers == {"Content-Type": "application/x-www-form-urlencoded"}
            assert 0 < timeout <= 900
            assert exchange["client_id"] == ["issued-offline-client"]
            assert exchange["resource"] == authorization["resource"]
            assert exchange["redirect_uri"] == authorization["redirect_uri"]
            assert "client_secret" not in exchange and "scope" not in exchange
            challenge = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"][0].encode()).digest())
            assert challenge.rstrip(b"=").decode() == authorization["code_challenge"][0]
            result = manager.get(started["id"])
            assert "auth_url" not in result
            assert "offline-refresh" not in str(result) and "offline-code" not in str(result)
            document = json.loads(path.read_text())
            assert document["registration"]["ext_agent_host_id"] == authorization["ext_agent_host_id"][0]
            assert document["grant"]["scopes"] == sorted(siwc_auth.SCOPES.split())
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("returned", ["", "&client_id=dynamic_agent_client", "&client_id=a&client_id=b"])
def test_registration_callback_requires_a_single_issued_client_id(tmp_path, returned):
    async def scenario():
        async def forbidden(*args):
            pytest.fail("Неверный client_id должен отклоняться до token exchange")

        manager = siwc_auth.make_browser_login_manager(tmp_path / "credentials.json", transport=forbidden)
        try:
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            port = urlsplit(query["redirect_uri"][0]).port
            response = await callback(port, "/auth/callback?state=" + query["state"][0] + "&code=x" + returned)
            assert b"offline-refresh" not in response and b"code_verifier" not in response
            assert manager.get(started["id"])["status"] == "failed"
            assert "tokens" not in json.loads((tmp_path / "credentials.json").read_text())
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_issued_registration_survives_exchange_rejection_for_next_login(tmp_path):
    async def scenario():
        path = tmp_path / "credentials.json"

        async def reject(*args):
            return 400, b'{"error":"invalid_grant"}'

        manager = siwc_auth.make_browser_login_manager(path, transport=reject)
        try:
            first = await manager.start()
            query = parse_qs(urlsplit(first["auth_url"]).query)
            host_id = query["ext_agent_host_id"]
            port = urlsplit(query["redirect_uri"][0]).port
            await callback(port, "/auth/callback?" + urlencode({"state": query["state"][0],
                "code": "offline-code", "client_id": "issued-offline-client"}))
            assert manager.get(first["id"])["status"] == "failed"
            second = await manager.start()
            next_query = parse_qs(urlsplit(second["auth_url"]).query)
            assert next_query["client_id"] == ["issued-offline-client"]
            assert next_query["ext_agent_host_id"] == host_id
            assert "agent_name_hint" not in next_query
            assert next_query["nonce"] != query["nonce"] and next_query["state"] != query["state"]
            assert next_query["code_challenge"] != query["code_challenge"]
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("returned", [None, "issued-offline-client", "wrong-client"])
def test_relogin_reuses_registration_and_rejects_client_mismatch(tmp_path, sign, returned):
    async def scenario():
        path = helpers.create_grant(tmp_path / "credentials.json", sign)
        original = path.read_bytes()
        authorization, calls = {}, []

        async def transport(*args):
            calls.append(1)
            return 200, json.dumps({"access_token": "next-access", "refresh_token": "next-refresh",
                "id_token": sign(nonce=authorization["nonce"][0]), "scope": siwc_auth.SCOPES,
                "expires_in": 3600}).encode()

        manager = siwc_auth.make_browser_login_manager(path, transport=transport)
        try:
            started = await manager.start()
            authorization.update(parse_qs(urlsplit(started["auth_url"]).query))
            assert authorization["client_id"] == ["issued-offline-client"]
            assert "agent_name_hint" not in authorization
            query = {"state": authorization["state"][0], "code": "offline-code"}
            if returned is not None:
                query["client_id"] = returned
            await callback(urlsplit(authorization["redirect_uri"][0]).port, "/auth/callback?" + urlencode(query))
            if returned == "wrong-client":
                assert manager.get(started["id"])["status"] == "failed" and not calls
                assert path.read_bytes() == original
            else:
                assert manager.get(started["id"])["status"] == "succeeded" and calls == [1]
                assert auth.load_token_data(path).access_token == "next-access"
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("change", [{"nonce": "wrong"}, {"sub": "wrong"}, {"scope": "openid profile"}])
def test_bad_identity_or_direct_scope_preserves_existing_grant(tmp_path, sign, change):
    async def scenario():
        path = helpers.create_grant(tmp_path / "credentials.json", sign)
        original, authorization = path.read_bytes(), {}

        async def transport(*args):
            claims = {"nonce": authorization["nonce"][0]}
            claims.update({key: value for key, value in change.items() if key != "scope"})
            return 200, json.dumps({"access_token": "next-access", "refresh_token": "next-refresh",
                "id_token": sign(**claims), "scope": change.get("scope", siwc_auth.SCOPES),
                "expires_in": 3600}).encode()

        manager = siwc_auth.make_browser_login_manager(path, transport=transport)
        try:
            started = await manager.start()
            authorization.update(parse_qs(urlsplit(started["auth_url"]).query))
            await callback(urlsplit(authorization["redirect_uri"][0]).port, "/auth/callback?" + urlencode({
                "state": authorization["state"][0], "code": "offline-code"}))
            assert manager.get(started["id"])["status"] == "failed"
            assert path.read_bytes() == original
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_cancellation_uses_existing_commit_guard(tmp_path, sign, monkeypatch):
    async def scenario():
        path = helpers.create_grant(tmp_path / "credentials.json", sign)
        original, authorization = path.read_bytes(), {}
        entered, release = threading.Event(), threading.Event()
        write = auth._write_auth_json

        def blocked_write(destination, data, **kwargs):
            if "tokens" in data:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError
            return write(destination, data, **kwargs)

        monkeypatch.setattr(auth, "_write_auth_json", blocked_write)

        async def transport(*args):
            return 200, json.dumps({"access_token": "next-access", "refresh_token": "next-refresh",
                "id_token": sign(nonce=authorization["nonce"][0]), "scope": siwc_auth.SCOPES,
                "expires_in": 3600}).encode()

        manager = siwc_auth.make_browser_login_manager(path, transport=transport)
        try:
            started = await manager.start()
            authorization.update(parse_qs(urlsplit(started["auth_url"]).query))
            request = asyncio.create_task(callback(urlsplit(authorization["redirect_uri"][0]).port,
                "/auth/callback?" + urlencode({"state": authorization["state"][0], "code": "offline-code"})))
            assert await asyncio.to_thread(entered.wait, 5)
            cancel = asyncio.create_task(manager.cancel(started["id"]))
            await asyncio.sleep(0)
            release.set()
            assert (await asyncio.wait_for(cancel, 5))["status"] == "cancelled"
            await request
            assert path.read_bytes() == original
            assert not list(path.parent.glob(".*.tmp-*"))
        finally:
            release.set()
            await manager.close()

    asyncio.run(scenario())


def test_denial_does_not_exchange_or_create_credentials(tmp_path):
    async def scenario():
        path = tmp_path / "credentials.json"

        async def forbidden(*args):
            pytest.fail("Отказ владельца не должен приводить к token exchange")

        manager = siwc_auth.make_browser_login_manager(path, transport=forbidden)
        try:
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            response = await callback(urlsplit(query["redirect_uri"][0]).port, "/auth/callback?" + urlencode({
                "state": query["state"][0], "error": "access_denied"}))
            assert b"400 Bad Request" in response
            assert manager.get(started["id"])["error_code"] == "login_denied"
            assert "tokens" not in json.loads(path.read_text())
        finally:
            await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("open_browser", [False, True])
def test_cli_login_opens_browser_only_with_explicit_flag(monkeypatch, capsys, open_browser):
    opened, closed = [], []

    class Manager:
        async def start(self):
            return {"id": "offline-login", "auth_url": siwc_auth.AUTHORIZE_URL + "?state=offline-state"}

        def get(self, login_id):
            return {"status": "succeeded"}

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(siwc_auth, "make_browser_login_manager", lambda path: Manager())
    monkeypatch.setattr(siwc_cli.webbrowser, "open", lambda url: opened.append(url))
    argv = ["login"] + (["--open-browser"] if open_browser else [])
    assert siwc_cli.main(argv) == 0
    assert bool(opened) is open_browser and closed == [True]
    output = capsys.readouterr().out
    assert siwc_auth.AUTHORIZE_URL in output
    assert "refresh_token" not in output and "code_verifier" not in output
