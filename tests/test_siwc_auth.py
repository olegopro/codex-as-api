"""Offline-проверки SIWC: подпись, binding, ротация и защищённое хранилище."""
import hashlib
import io
import json
import multiprocessing
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from codex_as_api import auth, siwc_auth, siwc_cli


def create_keys(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    public.update(kid="offline-key", alg="RS256", use="sig")
    client = jwt.PyJWKClient(siwc_auth.JWKS_URL)
    monkeypatch.setattr(client, "fetch_data", lambda: {"keys": [public]})
    monkeypatch.setattr(siwc_auth, "_JWKS_CLIENT", client)

    def sign(**overrides):
        now = int(time.time())
        claims = {"iss": siwc_auth.ISSUER, "aud": "issued-offline-client", "sub": "offline-subject",
            "iat": now - 60, "exp": now + 3600, "nonce": "offline-nonce"}
        claims.update(overrides)
        claims = {key: value for key, value in claims.items() if value is not None}
        return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "offline-key"})

    return sign


def create_grant(path, sign, *, expired=False):
    context = siwc_auth._registration_for_login(path)
    siwc_auth._save_issued_client(path, context, "issued-offline-client")
    context.update(client_id="issued-offline-client", nonce="offline-nonce", received_at=time.time())
    payload = {"access_token": "offline-access", "refresh_token": "offline-refresh", "id_token": sign(),
        "scope": siwc_auth.SCOPES, "expires_in": 3600, "token_type": "Bearer", "_siwc_context": context}
    assert siwc_auth._commit_login(path, payload, lambda: True)
    if expired:
        document = json.loads(path.read_text())
        document["grant"]["access_expires_at"] = time.time() - 60
        auth._write_auth_json(path, document)
    return path


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Незамоканный внешний HTTP-запрос запрещён")

    monkeypatch.setattr(auth, "_urlopen_no_redirect", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


@pytest.fixture()
def sign(monkeypatch):
    return create_keys(monkeypatch)


@pytest.fixture()
def grant(tmp_path, sign):
    return create_grant(tmp_path / "siwc" / "credentials.json", sign)


def test_new_host_id_is_stable_uuid_urn(tmp_path):
    path = tmp_path / "credentials.json"
    first = siwc_auth._registration_for_login(path)
    host = first["ext_agent_host_id"]
    assert uuid.UUID(host).version == 4
    assert uuid.UUID(host).urn == host
    assert siwc_auth._registration_for_login(path)["ext_agent_host_id"] == host


@pytest.mark.parametrize("issued_client", [None, "issued-offline-client"])
def test_only_unregistered_legacy_host_id_is_migrated(tmp_path, issued_client):
    path = tmp_path / "credentials.json"
    old_host = "old_urlsafe_host_identifier"
    auth._write_auth_json(path, {"oauth_profile": "siwc", "version": 1, "registration": {
        "client_id": issued_client, "ext_agent_host_id": old_host,
        "resource": siwc_auth.RESOURCE, "registration_id": "offline-registration",
    }})
    result = siwc_auth._registration_for_login(path)
    if issued_client is None:
        assert result["ext_agent_host_id"] != old_host
        assert uuid.UUID(result["ext_agent_host_id"]).urn == result["ext_agent_host_id"]
        assert result["registration_id"] != "offline-registration"
    else:
        assert result["ext_agent_host_id"] == old_host
        assert result["registration_id"] == "offline-registration"
        assert result["client_id"] == issued_client


def test_siwc_identity_is_verified_without_private_chatgpt_claims(grant):
    token = auth.load_token_data(grant)
    assert token.oauth_profile == "siwc"
    assert token.account_id == token.user_id == "offline-subject"
    assert token.client_id == "issued-offline-client"
    assert token.resource == siwc_auth.RESOURCE
    assert token.plan_type is None and token.fedramp is False
    assert token.access_token == "offline-access" and not token.expired
    assert auth.token_for_request(grant, expected_profile="siwc") == token


@pytest.mark.parametrize("overrides", [
    {"iss": "https://wrong.example"}, {"aud": "other-client"}, {"exp": 1}, {"sub": None},
    {"nonce": "other-nonce"}, {"iat": None}, {"exp": True}, {"aud": ["issued-offline-client", "other"]},
    {"azp": "other-client"},
])
def test_id_token_validation_rejects_invalid_claims(sign, overrides):
    with pytest.raises(auth.ChatGPTOAuthMissingError):
        siwc_auth.verify_id_token(sign(**overrides), "issued-offline-client", nonce="offline-nonce", require_nonce=True)


def test_signature_algorithm_and_subject_are_not_trusted(sign):
    different_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = {"iss": siwc_auth.ISSUER, "aud": "issued-offline-client", "sub": "offline-subject",
        "iat": int(time.time()) - 60, "exp": int(time.time()) + 3600}
    forged = jwt.encode(claims, different_key, algorithm="RS256", headers={"kid": "offline-key"})
    symmetric = jwt.encode(claims, "offline-symmetric-test-key-32-bytes-long", algorithm="HS256")
    for token in (forged, symmetric):
        with pytest.raises(auth.ChatGPTOAuthMissingError):
            siwc_auth.verify_id_token(token, "issued-offline-client")
    with pytest.raises(auth.ChatGPTOAuthMissingError):
        siwc_auth.verify_id_token(sign(), "issued-offline-client", nonce="offline-nonce", subject="another-subject")


@pytest.mark.parametrize("field,value", [
    ("sub", "another-subject"), ("client_id", "another-client"),
    ("resource", "https://wrong.example/v1"), ("scopes", ["openid"]),
    ("id_token_sha256", "not-a-binding"),
])
def test_tampered_binding_is_not_a_verified_identity(grant, field, value):
    data = json.loads(grant.read_text())
    data["grant"][field] = value
    auth._write_auth_json(grant, data)
    with pytest.raises(auth.ChatGPTOAuthMissingError):
        auth.load_token_data(grant)


def test_stored_expired_id_token_does_not_prevent_refresh(grant, sign, monkeypatch):
    data = json.loads(grant.read_text())
    data["tokens"]["id_token"] = sign(exp=int(time.time()) - 30)
    data["grant"]["id_token_sha256"] = hashlib.sha256(data["tokens"]["id_token"].encode()).hexdigest()
    data["grant"]["access_expires_at"] = time.time() - 30
    auth._write_auth_json(grant, data)
    calls = []

    def post(fields):
        calls.append(fields)
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh", "expires_in": 3600}

    monkeypatch.setattr(siwc_auth, "_post_refresh", post)
    refreshed = auth.token_for_request(grant, expected_profile="siwc")
    assert refreshed.access_token == "rotated-access" and refreshed.refresh_token == "rotated-refresh"
    assert refreshed.id_token == data["tokens"]["id_token"]
    assert len(calls) == 1 and calls[0]["client_id"] == "issued-offline-client"
    assert "scope" not in calls[0]


def test_refresh_posts_exact_form_and_rotates_atomically(grant, monkeypatch):
    before = auth.load_token_data(grant)
    calls = []

    class Response(io.BytesIO):
        pass

    def open_request(request, *, timeout):
        calls.append((request.full_url, urllib.parse.parse_qs(request.data.decode()),
            request.get_header("Content-type"), timeout))
        return Response(json.dumps({"access_token": "next-access", "refresh_token": "next-refresh",
            "scope": siwc_auth.SCOPES, "expires_in": 7200}).encode())

    monkeypatch.setattr(auth, "_urlopen_no_redirect", open_request)
    refreshed = auth.refresh_after_unauthorized(before, expected_profile="siwc")
    assert refreshed.access_token == "next-access" and refreshed.refresh_token == "next-refresh"
    assert refreshed.login_generation == before.login_generation
    assert calls == [(siwc_auth.TOKEN_URL, {"grant_type": ["refresh_token"],
        "client_id": ["issued-offline-client"], "refresh_token": ["offline-refresh"],
        "resource": [siwc_auth.RESOURCE]}, "application/x-www-form-urlencoded", auth.REFRESH_TIMEOUT_SECONDS)]
    assert json.loads(grant.read_text())["tokens"]["refresh_token"] == "next-refresh"
    assert not list(grant.parent.glob(".*.tmp-*"))
    assert auth.refresh_after_unauthorized(before).access_token == "next-access"
    assert len(calls) == 1


def test_refresh_id_token_is_optional_and_nonce_is_not_required(grant, sign, monkeypatch):
    monkeypatch.setattr(siwc_auth, "_post_refresh", lambda fields: {
        "access_token": "next-access", "refresh_token": "next-refresh", "expires_in": 3600,
        "id_token": sign(nonce=None)})
    refreshed = auth.refresh_after_unauthorized(auth.load_token_data(grant))
    assert refreshed.user_id == "offline-subject"
    assert "nonce" not in siwc_auth.verify_id_token(refreshed.id_token, "issued-offline-client")


@pytest.mark.parametrize("change", [
    {"scope": "openid profile"}, {"expires_in": 0}, {"token_type": "Basic"},
    {"refresh_token": ""}, {"access_token": "unsafe\nheader"},
    {"id_claims": {"sub": "different-subject"}}, {"id_claims": {"nonce": "different-nonce"}},
    {"id_claims": {"aud": "other-client"}}, {"id_claims": {"exp": 1}},
])
def test_invalid_refresh_never_partially_replaces_tokens(grant, sign, monkeypatch, change):
    original = json.loads(grant.read_text())
    payload = {"access_token": "next-access", "refresh_token": "next-refresh", "expires_in": 3600}
    changes = dict(change)
    if "id_claims" in changes:
        payload["id_token"] = sign(**changes.pop("id_claims"))
    payload.update(changes)
    calls = []

    def post(fields):
        calls.append(fields)
        return payload

    monkeypatch.setattr(siwc_auth, "_post_refresh", post)
    current = auth.load_token_data(grant)
    with pytest.raises(auth.ChatGPTOAuthProtocolError):
        auth.refresh_after_unauthorized(current)
    stored = json.loads(grant.read_text())
    assert stored["tokens"] == original["tokens"] and stored["grant"] == original["grant"]
    assert stored["refresh_error"] == "reauthentication_required"
    with pytest.raises(auth.ChatGPTOAuthRefreshError):
        auth.refresh_after_unauthorized(current)
    assert len(calls) == 1


def test_invalid_grant_is_not_retried(grant, monkeypatch):
    calls = []

    def open_request(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(siwc_auth.TOKEN_URL, 400, "rejected", {}, io.BytesIO(b"private-response"))

    monkeypatch.setattr(auth, "_urlopen_no_redirect", open_request)
    current = auth.load_token_data(grant)
    for _ in range(2):
        with pytest.raises(auth.ChatGPTOAuthRefreshError, match="SIWC") as error:
            auth.refresh_after_unauthorized(current)
        assert "private-response" not in str(error.value)
    assert len(calls) == 1


def test_network_failure_has_no_automatic_retry(grant, monkeypatch):
    calls = []

    def open_request(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("private-network-detail")

    monkeypatch.setattr(auth, "_urlopen_no_redirect", open_request)
    with pytest.raises(auth.ChatGPTOAuthUpstreamError) as error:
        auth.refresh_after_unauthorized(auth.load_token_data(grant))
    assert len(calls) == 1 and "private-network-detail" not in str(error.value)


def test_profile_check_precedes_legacy_refresh(auth_json_factory, monkeypatch):
    path = auth_json_factory(access_payload={"exp": 1})
    with pytest.raises(auth.ChatGPTOAuthMissingError, match="profile"):
        auth.token_for_request(path, expected_profile="siwc")
    legacy = auth.load_token_data(path)
    assert legacy.oauth_profile == "codex" and legacy.client_id is None
    with pytest.raises(auth.ChatGPTOAuthMissingError, match="profile"):
        auth.refresh_after_unauthorized(legacy, expected_profile="siwc")


def test_legacy_credentials_are_not_imported_or_replaced(tmp_path, auth_json_factory):
    legacy = auth_json_factory()
    original = legacy.read_bytes()
    with pytest.raises(auth.ChatGPTOAuthMissingError, match="legacy"):
        siwc_auth._registration_for_login(legacy)
    assert legacy.read_bytes() == original and not legacy.with_name(legacy.name + ".lock").exists()
    assert not (tmp_path / "siwc").exists()


def test_credentials_and_lock_are_owner_only(grant):
    if os.name == "nt":
        pytest.skip("POSIX-права не применяются")
    assert grant.parent.stat().st_mode & 0o777 == 0o700
    assert grant.stat().st_mode & 0o777 == 0o600
    assert grant.with_name(grant.name + ".lock").stat().st_mode & 0o777 == 0o600


def test_status_is_readonly_and_has_no_credentials(grant, monkeypatch, capsys):
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in grant.parent.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("status не должен проверять JWT, создавать хранилище или обновлять grant")

    monkeypatch.setattr(auth, "load_token_data", forbidden)
    monkeypatch.setattr(auth, "token_for_request", forbidden)
    monkeypatch.setattr(siwc_auth, "_registration_for_login", forbidden)
    monkeypatch.setattr(siwc_auth, "verify_id_token", forbidden)
    assert siwc_cli.main(["status", "--auth-path", str(grant)]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["state"] == "credentials_present"
    for secret in ("offline-access", "offline-refresh", "offline-subject", "issued-offline-client", "offline-nonce"):
        assert secret not in output
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in grant.parent.iterdir()} == before


def test_status_missing_store_does_not_create_directories(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing" / "credentials.json"
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "must-not-read-legacy.json"))
    monkeypatch.setenv("CODEX_AS_API_SIWC_AUTH_PATH", str(path))
    assert siwc_cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "oauth_profile": "siwc", "state": "login_required", "registered": False}
    assert not path.parent.exists()


def test_refresh_is_serialized_across_processes(grant, monkeypatch):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("Для проверки межпроцессного flock требуется fork")
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    ready = context.Event()
    result = context.Queue()
    current = auth.load_token_data(grant)

    def post(fields):
        with calls.get_lock():
            calls.value += 1
        ready.set()
        time.sleep(0.15)
        return {"access_token": "next-access", "refresh_token": "next-refresh", "expires_in": 3600}

    def refresh():
        try:
            value = auth.refresh_after_unauthorized(current)
            result.put((value.access_token, value.refresh_token))
        except Exception as error:
            result.put(type(error).__name__)

    monkeypatch.setattr(siwc_auth, "_post_refresh", post)
    first, second = context.Process(target=refresh), context.Process(target=refresh)
    first.start()
    try:
        assert ready.wait(5)
        second.start()
        assert result.get(timeout=5) == ("next-access", "next-refresh")
        assert result.get(timeout=5) == ("next-access", "next-refresh")
        first.join(5)
        second.join(5)
        assert first.exitcode == second.exitcode == 0
        assert calls.value == 1
    finally:
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)
        result.close()
