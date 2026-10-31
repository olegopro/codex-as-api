"""Явный профиль SIWC: проверка OIDC, отдельное хранилище и ротация grant."""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import http.client
import os
import pathlib
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import jwt

from . import auth
from .browser_login import BrowserLoginManager, _BrowserAttempt
from .device_login import LoginError
from .strict_json import strict_json_loads

OAUTH_PROFILE = "siwc"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "codex-as-api-siwc"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
DEFAULT_AUTH_PATH = "~/.codex-as-api-siwc/credentials.json"
CLOCK_SKEW_SECONDS = 30
_JWKS_CLIENT = jwt.PyJWKClient(JWKS_URL, timeout=15, cache_jwk_set=True, lifespan=300)


def resolve_auth_path(raw: str | pathlib.Path | None = None) -> pathlib.Path:
    value = raw if raw is not None else os.getenv("CODEX_AS_API_SIWC_AUTH_PATH", DEFAULT_AUTH_PATH)
    if not str(value).strip():
        raise ValueError("SIWC credential path must be non-empty")
    return pathlib.Path(value).expanduser().absolute()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not auth._is_ascii_visible_token(value):
        raise auth.ChatGPTOAuthMissingError(f"SIWC {field} is invalid")
    return value


def _read_document(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink():
            raise auth.ChatGPTOAuthMissingError("SIWC credential symlinks are not supported")
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError
        data = strict_json_loads(raw)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError):
        raise auth.ChatGPTOAuthMissingError("SIWC credential file is unavailable or invalid") from None
    if not isinstance(data, dict) or data.get("oauth_profile") != OAUTH_PROFILE:
        raise auth.ChatGPTOAuthMissingError(
            "SIWC requires its own credential file; legacy credentials are not imported")
    return data


def _registration(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("oauth_profile") != OAUTH_PROFILE or type(data.get("version")) is not int or data["version"] != 1:
        raise auth.ChatGPTOAuthMissingError("SIWC credential schema is unsupported")
    registration = data.get("registration")
    if not isinstance(registration, dict) or registration.get("resource") != RESOURCE:
        raise auth.ChatGPTOAuthMissingError("SIWC registration resource is invalid")
    for key in ("ext_agent_host_id", "registration_id"):
        _text(registration.get(key), key)
    client_id = registration.get("client_id")
    if client_id is not None:
        _text(client_id, "client_id")
        if client_id == DYNAMIC_CLIENT_ID:
            raise auth.ChatGPTOAuthMissingError("SIWC requires an issued client_id")
    return registration


def _private_directory(path: pathlib.Path) -> None:
    if path.parent.is_symlink():
        raise auth.ChatGPTOAuthMissingError("SIWC credential directory must not be a symlink")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if sys.platform != "win32":
        path.parent.chmod(0o700)


@contextlib.contextmanager
def _credential_lock(path: pathlib.Path) -> Iterator[None]:
    # Один замок охватывает reload → запрос → проверку → atomic replace.
    existing = _read_document(path)
    if existing is not None:
        _registration(existing)
    _private_directory(path)
    local_lock = auth._write_lock(path)
    if not local_lock.acquire(timeout=auth.REFRESH_TIMEOUT_SECONDS):
        raise auth.ChatGPTOAuthRefreshError("SIWC credential lock timed out")
    fd: int | None = None
    locked = False
    try:
        fd = os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if sys.platform != "win32":
            import fcntl

            os.fchmod(fd, 0o600)
        else:
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
        deadline = time.monotonic() + auth.REFRESH_TIMEOUT_SECONDS
        while True:
            try:
                if sys.platform != "win32":
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise auth.ChatGPTOAuthRefreshError("SIWC credential lock timed out") from None
                time.sleep(0.05)
            except OSError as exc:
                if sys.platform != "win32" or exc.errno not in (13, 36):
                    raise
                if time.monotonic() >= deadline:
                    raise auth.ChatGPTOAuthRefreshError("SIWC credential lock timed out") from None
                time.sleep(0.05)
        yield
    finally:
        if fd is not None:
            try:
                if locked:
                    if sys.platform != "win32":
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    else:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
        local_lock.release()


def _registration_for_login(path: pathlib.Path) -> dict[str, Any]:
    with _credential_lock(path):
        data = _read_document(path)
        if data is None:
            data = {"oauth_profile": OAUTH_PROFILE, "version": 1, "registration": {
                "client_id": None, "ext_agent_host_id": uuid.uuid4().urn,
                "resource": RESOURCE, "registration_id": uuid.uuid4().hex,
            }}
            auth._write_auth_json(path, data)
        registration = _registration(data)
        if registration.get("client_id") is None and "tokens" not in data:
            host_id = registration["ext_agent_host_id"]
            try:
                canonical_host = uuid.UUID(host_id).urn == host_id
            except ValueError:
                canonical_host = False
            if not canonical_host:
                registration["ext_agent_host_id"] = uuid.uuid4().urn
                registration["registration_id"] = uuid.uuid4().hex
                auth._write_auth_json(path, data)
        subject = None
        if "tokens" in data:
            subject = token_data_from_document(data, path).user_id
        return {**registration, "sub": subject, "login_generation": data.get("_login_generation")}


def _save_issued_client(path: pathlib.Path, context: dict[str, Any], client_id: str) -> None:
    with _credential_lock(path):
        data = _read_document(path)
        if data is None:
            raise LoginError("login_registration_changed")
        registration = _registration(data)
        if any(registration.get(key) != context.get(key)
                for key in ("registration_id", "ext_agent_host_id", "resource")):
            raise LoginError("login_registration_changed")
        if registration.get("client_id") not in (None, client_id):
            raise LoginError("login_registration_changed")
        if registration.get("client_id") is None:
            registration["client_id"] = client_id
            auth._write_auth_json(path, data)


def verify_id_token(
    token: str, client_id: str, *, nonce: str | None = None, subject: str | None = None,
    require_nonce: bool = False, allow_expired: bool = False,
) -> dict[str, Any]:
    try:
        if jwt.get_unverified_header(token).get("alg") != "RS256":
            raise jwt.InvalidAlgorithmError()
        key = _JWKS_CLIENT.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256"], issuer=ISSUER, audience=client_id,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["iss", "aud", "exp", "iat", "sub"], "verify_exp": not allow_expired})
        verified_sub = _text(claims.get("sub"), "ID token subject")
        _timestamp(claims["exp"], "ID token expiry")
        _timestamp(claims["iat"], "ID token issue time")
        if subject is not None and verified_sub != subject:
            raise ValueError
        audiences = claims["aud"]
        if (isinstance(audiences, list) and len(audiences) > 1 and claims.get("azp") != client_id
                or "azp" in claims and claims["azp"] != client_id):
            raise ValueError
        actual_nonce = claims.get("nonce")
        if ((require_nonce or actual_nonce is not None)
                and (not isinstance(actual_nonce, str) or nonce is None
                     or not secrets.compare_digest(actual_nonce, nonce))):
            raise ValueError
        return dict(claims)
    except (jwt.PyJWTError, ValueError, TypeError, OSError, urllib.error.URLError):
        raise auth.ChatGPTOAuthMissingError("SIWC ID token verification failed") from None


def _granted_scopes(value: object) -> list[str]:
    if not isinstance(value, str):
        raise auth.ChatGPTOAuthMissingError("SIWC granted scope is missing")
    scopes = value.split()
    if not scopes or any(not auth._is_ascii_visible_token(scope) for scope in scopes) or DIRECT_SCOPE not in scopes:
        raise auth.ChatGPTOAuthMissingError("SIWC direct-use permission was not granted")
    return sorted(set(scopes))


def _timestamp(value: object, field: str) -> dt.datetime:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise auth.ChatGPTOAuthMissingError(f"SIWC {field} is invalid")
    try:
        return dt.datetime.fromtimestamp(value, dt.timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise auth.ChatGPTOAuthMissingError(f"SIWC {field} is invalid") from None


def _response_expiry(payload: dict[str, Any], received_at: float) -> float:
    value = payload.get("expires_in")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise auth.ChatGPTOAuthMissingError("SIWC expires_in must be a positive integer")
    expires = received_at + value
    _timestamp(expires, "access expiry")
    return expires


def _response_tokens(payload: dict[str, Any], *, login: bool) -> dict[str, str]:
    tokens = {"access_token": _text(payload.get("access_token"), "access_token")}
    for key in ("refresh_token", "id_token"):
        if login or key in payload:
            tokens[key] = _text(payload.get(key), key)
    if "token_type" in payload and (not isinstance(payload["token_type"], str)
            or payload["token_type"].lower() != "bearer"):
        raise auth.ChatGPTOAuthMissingError("SIWC token_type must be Bearer")
    return tokens


def token_data_from_document(data: dict[str, Any], path: pathlib.Path) -> auth.ChatGPTTokenData:
    registration = _registration(data)
    client_id = _text(registration.get("client_id"), "issued client_id")
    generation = _text(data.get("_login_generation"), "login generation")
    tokens, grant = data.get("tokens"), data.get("grant")
    if not isinstance(tokens, dict) or not isinstance(grant, dict):
        raise auth.ChatGPTOAuthMissingError("SIWC sign-in is required")
    if any(key in data for key in ("access_token", "refresh_token", "id_token", "chatgptAuthTokens")):
        raise auth.ChatGPTOAuthMissingError("SIWC credential schema mixes token namespaces")
    for key in ("access_token", "refresh_token", "id_token"):
        _text(tokens.get(key), key)
    subject = _text(grant.get("sub"), "verified subject")
    if grant.get("issuer") != ISSUER or grant.get("client_id") != client_id or grant.get("resource") != RESOURCE:
        raise auth.ChatGPTOAuthMissingError("SIWC grant binding is invalid")
    scopes = grant.get("scopes")
    if not isinstance(scopes, list) or any(not isinstance(value, str) for value in scopes):
        raise auth.ChatGPTOAuthMissingError("SIWC stored scopes are invalid")
    _granted_scopes(" ".join(scopes))
    _timestamp(grant.get("verified_at"), "verification time")
    if grant.get("id_token_sha256") != hashlib.sha256(tokens["id_token"].encode()).hexdigest():
        raise auth.ChatGPTOAuthMissingError("SIWC stored ID token binding is invalid")
    # Истёкший ID token всё ещё удостоверяет binding, но не срок access/refresh grant.
    verify_id_token(tokens["id_token"], client_id, nonce=grant.get("nonce"), subject=subject, allow_expired=True)
    return auth.ChatGPTTokenData(auth_path=path, access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"], id_token=tokens["id_token"], account_id=subject,
        plan_type=None, user_id=subject, fedramp=False,
        access_expires_at=_timestamp(grant.get("access_expires_at"), "access expiry"),
        login_generation=generation, oauth_profile=OAUTH_PROFILE, client_id=client_id, resource=RESOURCE)


def _commit_login(path: pathlib.Path, payload: dict[str, Any], guard: Callable[[], bool]) -> bool:
    context = payload["_siwc_context"]
    tokens = _response_tokens(payload, login=True)
    client_id = context["client_id"]
    claims = verify_id_token(tokens["id_token"], client_id, nonce=context["nonce"],
        subject=context.get("sub"), require_nonce=True)
    scopes = _granted_scopes(payload.get("scope"))
    expires = _response_expiry(payload, context["received_at"])
    with _credential_lock(path):
        data = _read_document(path)
        if data is None:
            raise LoginError("login_registration_changed")
        registration = _registration(data)
        if (any(registration.get(key) != context.get(key)
                for key in ("client_id", "ext_agent_host_id", "registration_id", "resource"))
                or data.get("_login_generation") != context.get("login_generation")):
            raise LoginError("login_registration_changed")
        data["tokens"] = tokens
        data["grant"] = {"sub": claims["sub"], "issuer": ISSUER, "client_id": client_id,
            "resource": RESOURCE, "scopes": scopes, "access_expires_at": expires,
            "verified_at": time.time(), "nonce": context["nonce"],
            "id_token_sha256": hashlib.sha256(tokens["id_token"].encode()).hexdigest()}
        if "auth_time" in claims:
            _timestamp(claims["auth_time"], "authentication time")
            data["grant"]["auth_time"] = claims["auth_time"]
        data["_login_generation"] = uuid.uuid4().hex
        data.pop("refresh_error", None)
        try:
            auth._write_auth_json(path, data, before_replace=guard)
        except auth._LoginCommitCancelledError:
            return False
    return True


class _SIWCBrowserProfile:
    authorize_url, token_url = AUTHORIZE_URL, TOKEN_URL

    def __init__(self, path: pathlib.Path):
        self.path = path

    async def authorization_parameters(self, attempt: _BrowserAttempt) -> dict[str, str]:
        context = await asyncio.to_thread(_registration_for_login, self.path)
        context["nonce"] = secrets.token_urlsafe(32)
        attempt.profile_data = context
        parameters = {"client_id": context["client_id"] or DYNAMIC_CLIENT_ID, "nonce": context["nonce"],
            "ext_agent_host_id": context["ext_agent_host_id"], "scope": SCOPES, "resource": RESOURCE}
        if context["client_id"] is None:
            parameters["agent_name_hint"] = AGENT_NAME
        return parameters

    async def exchange_parameters(self, attempt: _BrowserAttempt, query: dict[str, list[str]]) -> dict[str, str]:
        context = attempt.profile_data
        returned = query.get("client_id", [])
        if len(returned) > 1 or returned and (not returned[0] or not auth._is_ascii_visible_token(returned[0])):
            raise LoginError("login_client_id_mismatch")
        issued = context.get("client_id")
        if issued is None:
            if not returned or returned[0] == DYNAMIC_CLIENT_ID:
                raise LoginError("login_client_id_missing")
            issued = returned[0]
            await asyncio.to_thread(_save_issued_client, self.path, context, issued)
            context["client_id"] = issued
        elif returned and returned[0] != issued:
            raise LoginError("login_client_id_mismatch")
        return {"client_id": issued, "resource": RESOURCE}

    def token_payload(self, attempt: _BrowserAttempt, payload: dict[str, object]) -> dict[str, Any]:
        return {**payload, "_siwc_context": {**attempt.profile_data, "received_at": time.time()}}


def make_browser_login_manager(auth_path: str | pathlib.Path | None = None, **kwargs: Any) -> BrowserLoginManager:
    path = resolve_auth_path(auth_path)

    async def commit(payload: dict[str, Any], guard: Callable[[], bool]) -> bool:
        return await asyncio.to_thread(_commit_login, path, payload, guard)

    return BrowserLoginManager(commit, ports=(0,), profile=_SIWCBrowserProfile(path), **kwargs)


def _post_refresh(fields: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(TOKEN_URL, data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    deadline = time.monotonic() + auth.REFRESH_TIMEOUT_SECONDS
    try:
        with auth._urlopen_no_redirect(request, timeout=auth.REFRESH_TIMEOUT_SECONDS) as response:
            body = auth._read_response_before_deadline(response, deadline)
        if len(body) > 1024 * 1024:
            raise ValueError
        payload = strict_json_loads(body)
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        if status in (400, 401):
            raise auth.ChatGPTOAuthRefreshError("SIWC refresh was rejected; a new sign-in is required") from None
        raise auth.ChatGPTOAuthUpstreamError(status, f"SIWC token refresh failed: HTTP {status}") from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        raise auth.ChatGPTOAuthUpstreamError(
            502, "SIWC token refresh failed; no automatic retry was performed") from None
    except (ValueError, UnicodeError):
        raise auth.ChatGPTOAuthProtocolError("SIWC token refresh returned invalid JSON") from None


def refresh_credentials(current: auth.ChatGPTTokenData, *, refresh_if_expiring: bool) -> auth.ChatGPTTokenData:
    path = current.auth_path
    with _credential_lock(path):
        data = _read_document(path)
        if data is None:
            raise auth.ChatGPTOAuthMissingError("SIWC credential file not found")
        latest = token_data_from_document(data, path)
        auth._fail_if_account_changed(current, latest)
        if auth._access_token_changed(current, latest) or auth._other_credentials_changed(current, latest):
            return latest
        if refresh_if_expiring and not latest.expires_within():
            return latest
        if data.get("refresh_error") is not None:
            raise auth.ChatGPTOAuthRefreshError("SIWC refresh requires a new sign-in")
        client_id = _text(latest.client_id, "client ID")
        try:
            payload = _post_refresh({"grant_type": "refresh_token", "client_id": client_id,
                "refresh_token": latest.refresh_token, "resource": RESOURCE})
        except (auth.ChatGPTOAuthRefreshError, auth.ChatGPTOAuthProtocolError):
            data["refresh_error"] = "reauthentication_required"
            auth._write_auth_json(path, data)
            raise
        received_at = time.time()
        try:
            tokens = _response_tokens(payload, login=False)
            grant = data["grant"]
            scopes = _granted_scopes(payload["scope"]) if "scope" in payload else grant["scopes"]
            expires = _response_expiry(payload, received_at)
            if "id_token" in tokens:
                claims = verify_id_token(tokens["id_token"], client_id,
                    nonce=grant.get("nonce"), subject=latest.user_id)
                if "auth_time" in claims:
                    _timestamp(claims["auth_time"], "authentication time")
                    if "auth_time" in grant and claims["auth_time"] != grant["auth_time"]:
                        raise auth.ChatGPTOAuthMissingError("SIWC authentication time changed during refresh")
                    grant["auth_time"] = claims["auth_time"]
                grant["id_token_sha256"] = hashlib.sha256(tokens["id_token"].encode()).hexdigest()
                grant["verified_at"] = received_at
            data["tokens"].update(tokens)
            grant.update(scopes=scopes, access_expires_at=expires)
            data["last_refresh"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            validated = token_data_from_document(data, path)
        except auth.ChatGPTOAuthError:
            original = _read_document(path)
            if original is not None:
                original["refresh_error"] = "reauthentication_required"
                auth._write_auth_json(path, original)
            raise auth.ChatGPTOAuthProtocolError("SIWC refresh returned invalid or mismatched credentials") from None
        auth._write_auth_json(path, data)
        return validated


def read_status(auth_path: str | pathlib.Path | None = None) -> dict[str, object]:
    """Только чтение и фиксированный whitelist; проверка JWT и сеть не запускаются."""
    result: dict[str, object] = {"oauth_profile": OAUTH_PROFILE, "state": "login_required", "registered": False}
    try:
        data = _read_document(resolve_auth_path(auth_path))
        if data is None:
            return result
        registration = _registration(data)
        result["registered"] = registration.get("client_id") is not None
        if "tokens" not in data:
            return result
        grant, tokens = data.get("grant"), data.get("tokens")
        if not isinstance(grant, dict) or not isinstance(tokens, dict):
            raise ValueError
        if any(not isinstance(tokens.get(key), str) or not tokens[key]
                for key in ("access_token", "refresh_token", "id_token")):
            raise ValueError
        scopes = grant.get("scopes")
        if not isinstance(scopes, list) or DIRECT_SCOPE not in scopes or grant.get("resource") != RESOURCE:
            raise ValueError
        expires = _timestamp(grant.get("access_expires_at"), "access expiry")
        result["access_expires_at"] = expires.isoformat().replace("+00:00", "Z")
        result["state"] = ("login_required" if data.get("refresh_error") is not None else
            "access_expired" if expires <= dt.datetime.now(dt.timezone.utc) else "credentials_present")
    except (auth.ChatGPTOAuthError, ValueError, OSError):
        result["state"] = "invalid"
    return result
