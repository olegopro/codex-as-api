"""CLI отключает серверную диагностику до импорта её эксклюзивного журнала."""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int | None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        os.environ["CODEX_AS_API_DIAGNOSTICS"] = "0"
    from .server import main as server_main

    return server_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
