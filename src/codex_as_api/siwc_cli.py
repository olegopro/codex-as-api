"""Локальные login/status/preflight обоих OAuth-профилей без сервера инференса."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import webbrowser
from collections.abc import Callable, Sequence
from typing import Any

from . import auth, siwc_auth
from .device_login import LoginError


async def _login(auth_path: str | None, *, open_browser: bool, backend: str = "siwc") -> int:
    if backend == "siwc":
        manager = siwc_auth.make_browser_login_manager(auth_path)
    else:
        from .browser_login import BrowserLoginManager

        async def commit(tokens: dict[str, Any], guard: Callable[[], bool]) -> bool:
            return await asyncio.to_thread(auth.commit_login, auth.resolve_auth_path(auth_path), tokens, guard)

        manager = BrowserLoginManager(commit)
    try:
        started = await manager.start()
        url = str(started["auth_url"])
        print("Вход разрешается только подтверждением на странице OpenAI.", flush=True)
        print(url, flush=True)
        if open_browser:
            await asyncio.to_thread(webbrowser.open, url)
        login_id = str(started["id"])
        while True:
            result = manager.get(login_id)
            if result["status"] != "pending":
                break
            await asyncio.sleep(0.25)
        if result["status"] == "succeeded":
            print("Вход завершён. Учётные данные сохранены локально.")
            return 0
        if result.get("error_code") == "login_persistence_uncertain":
            print("Сохранение OAuth не подтверждено; файл мог быть заменён.", file=sys.stderr)
        else:
            print("Вход OAuth не завершён.", file=sys.stderr)
        return 1
    finally:
        await manager.close()


def _preflight(auth_path: str | None, *, backend: str = "siwc") -> int:
    from .provider import ChatGPTOAuthProvider

    try:
        model = os.getenv("CODEX_AS_API_MODEL") or "gpt-6.1-sol"
        provider = ChatGPTOAuthProvider(model=model, backend=backend, auth_json_path=auth_path)
        catalog = provider.get_model_catalog()
        selected = catalog.model(model)
    except (auth.ChatGPTOAuthError, OSError, ValueError):
        print("OAuth preflight не пройден: авторизация или каталог недоступны.", file=sys.stderr)
        return 1
    print(json.dumps({"backend": backend, "upstream_base_url": provider.base_url,
                      "model": selected.slug, "models": len(catalog.models)}, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None, *, backend: str = "siwc") -> int:
    parser = argparse.ArgumentParser(description="Вход и локальный статус выбранного OAuth-профиля")
    parser.add_argument("--auth-path", help="Отдельный файл credentials выбранного профиля")
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="Вход OAuth через браузер и локальный callback")
    status = commands.add_parser("status", help="Только чтение статуса, без сети и обновления токенов")
    preflight = commands.add_parser("preflight", help="Проверка OAuth и GET каталога без генерации")
    for command in (login, status, preflight):
        command.add_argument("--auth-path", default=argparse.SUPPRESS, help="Отдельный файл credentials профиля")
    login.add_argument("--open-browser", action="store_true", help="Явно разрешить открытие страницы входа")
    args = parser.parse_args(argv)
    if args.command == "status":
        if backend == "siwc":
            status_result = siwc_auth.read_status(args.auth_path)
        else:
            try:
                token = auth.load_token_data(args.auth_path)
                status_result = {"oauth_profile": "codex", "state": "credentials_present",
                                 "registered": True, "expired": token.expired}
            except auth.ChatGPTOAuthError:
                status_result = {"oauth_profile": "codex", "state": "login_required", "registered": False}
        print(json.dumps(status_result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "preflight":
        return _preflight(args.auth_path) if backend == "siwc" else _preflight(args.auth_path, backend=backend)
    try:
        return asyncio.run(_login(args.auth_path, open_browser=args.open_browser, backend=backend))
    except KeyboardInterrupt:
        print("Вход OAuth отменён.", file=sys.stderr)
        return 130
    except (auth.ChatGPTOAuthError, LoginError, OSError, ValueError):
        print("Вход OAuth не выполнен. Подробности учётных данных не выводятся.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
