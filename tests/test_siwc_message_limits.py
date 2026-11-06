"""Real loopback WebSockets; no OAuth or model requests."""
from __future__ import annotations

import threading

import pytest
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, PayloadTooBig
from websockets.frames import Close
from websockets.sync.client import connect
from websockets.sync.server import serve

from codex_as_api import siwc_transport
from codex_as_api.siwc_transport import MAX_MESSAGE_BYTES, SIWCWebSocketPool

from .test_diagnostics import diagnostic_log as diagnostic_log
from .test_diagnostics import records


def test_local_payload_limit_remains_visible_when_peer_uses_another_close_code():
    error = ConnectionClosedError(Close(1000, ""), Close(1009, ""), False)
    error.__cause__ = PayloadTooBig(MAX_MESSAGE_BYTES + 1, MAX_MESSAGE_BYTES)
    code, category, source = siwc_transport._close_details(error, None)
    assert (code, source) == (1000, "received") and category == "message_too_large"


@pytest.mark.parametrize("size,compression", [
    (1024 * 1024 + 1, "deflate"), (8 * 1024 * 1024, "deflate"),
    (MAX_MESSAGE_BYTES + 1, "deflate"), (MAX_MESSAGE_BYTES + 1, None),
])
def test_real_message_limit_and_channel_retirement(monkeypatch, diagnostic_log, size, compression):
    prefix, suffix = '{"type":"response.completed","padding":"', '"}'
    large = prefix + "x" * (size - len(prefix) - len(suffix)) + suffix
    count = 0

    def handler(socket):
        nonlocal count
        count += 1
        socket.recv(timeout=5)
        try:
            socket.send(large if count == 1 else '{"type":"response.completed"}')
            socket.recv(timeout=5)  # Wait for the client to close its owned channel.
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0, compression=compression) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]

        def local_connect(_url, **kwargs):
            return connect(f"ws://127.0.0.1:{port}", **{**kwargs, "proxy": None, "compression": compression})

        monkeypatch.setattr(siwc_transport, "connect", local_connect)
        pool = SIWCWebSocketPool()
        options = {"account_id": "test", "access_token": "synthetic", "timeout": 5}
        body = {"input": [{"role": "user", "content": "fixture"}]}
        try:
            if size <= MAX_MESSAGE_BYTES:
                result = list(pool.events(body, **options))
                assert len(result[0]["padding"]) == size - len(prefix) - len(suffix)
                release = records(diagnostic_log)[-1]
                assert release["max_message_bytes"] == size and release["reason"] == "completed"
            else:
                with pytest.raises(OSError):
                    list(pool.events(body, **options))
                release = records(diagnostic_log)[-1]
                assert release["close_code"] == 1009 and release["close_category"] == "message_too_large"
                assert release["close_source"] in {"sent", "received"}
                assert pool.active_requests == 0
                # A rejected channel cannot be selected for the next request.
                assert list(pool.events(body, **options)) == [{"type": "response.completed"}]
                assert count == 2
        finally:
            pool.close()
        server.shutdown()
        thread.join(5)
        assert not thread.is_alive()
