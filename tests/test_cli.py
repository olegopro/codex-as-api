"""Прямой CLI не занимает диагностический журнал работающего сервера."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend", ["codex", "siwc"])
@pytest.mark.parametrize("args", [["status"], ["--help"]])
def test_cli_does_not_contend_with_live_diagnostics(tmp_path, backend, args):
    fcntl = pytest.importorskip("fcntl")
    log = tmp_path / "diagnostics.jsonl"
    with log.with_suffix(".jsonl.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [sys.executable, "-m", "codex_as_api.cli", *args],
            env={"HOME": str(tmp_path), "CODEX_HOME": str(tmp_path / "native"),
                 "CODEX_AS_API_BACKEND": backend, "CODEX_AS_API_DIAGNOSTICS": "1",
                 "CODEX_AS_API_DIAGNOSTICS_PATH": str(log),
                 "CODEX_AS_API_AUTH_PATH": str(tmp_path / "legacy/auth.json"),
                 "CODEX_AS_API_SIWC_AUTH_PATH": str(tmp_path / "siwc/auth.json")},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        )
    assert result.returncode == 0, result.stderr
    assert not log.exists()
    if args == ["status"]:
        assert json.loads(result.stdout)["state"] == "login_required"


def test_server_entry_keeps_diagnostics(monkeypatch):
    from codex_as_api import cli

    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS", "1")
    called = []
    monkeypatch.setitem(sys.modules, "codex_as_api.server", SimpleNamespace(main=lambda args: called.append(args)))
    cli.main([])
    assert called == [[]]
    assert os.environ["CODEX_AS_API_DIAGNOSTICS"] == "1"
