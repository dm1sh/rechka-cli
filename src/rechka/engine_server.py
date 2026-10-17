"""Unix-socket control server shared by the IM engine processes.

The daemon (and `rechka transcribe -o ibus|fcitx5`) are clients; the
framework engine process owns this server. One JSON object per line:

  {"cmd": "commit", "text": "..."} -> the engine's ack dict
  {"cmd": "ping"}                   -> {"ok": true, "framework": ..., ...}
  {"cmd": "shutdown"}               -> {"ok": true} and the server stops

Socket: $XDG_RUNTIME_DIR/rechka-engine-<framework>.sock
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any, Callable

from . import paths


def engine_socket_path(framework: str) -> Path:
    return paths.runtime_dir() / f"rechka-engine-{framework}.sock"


class EngineServer:
    def __init__(
        self,
        framework: str,
        on_commit: Callable[[str], dict[str, Any]],
        on_ping: Callable[[], dict[str, Any]] | None = None,
    ):
        self.framework = framework
        self.socket_path = engine_socket_path(framework)
        self._on_commit = on_commit
        self._on_ping = on_ping
        self._stopping = threading.Event()

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()  # stale socket from a previous engine
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(8)
        server.settimeout(0.5)
        try:
            while not self._stopping.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._serve_connection, args=(conn,), daemon=True
                ).start()
        finally:
            server.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

    def request_stop(self) -> None:
        self._stopping.set()

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            buffer = b""
            while b"\n" not in buffer:
                data = conn.recv(65536)
                if not data:
                    return
                buffer += data
            try:
                request = json.loads(buffer.splitlines()[0].decode("utf-8"))
                response = self._handle(request)
            except Exception as exc:  # one bad request must not kill the engine
                response = {"ok": False, "error": f"bad request: {exc}"}
            try:
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except OSError:
                pass

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("cmd", "")
        if command == "ping":
            extra = self._on_ping() if self._on_ping else {}
            return {"ok": True, "framework": self.framework, **extra}
        if command == "commit":
            return self._on_commit(str(request.get("text", "")))
        if command == "shutdown":
            self.request_stop()
            return {"ok": True, "detail": "engine stopping"}
        return {"ok": False, "error": f"unknown command: {command!r}"}


class CommitBridge:
    """Focus/queue state machine behind every engine (GI-independent).

    deliver(text) -> bool performs the framework-specific insertion
    (IBus commit_text on the focused context, or a typing tool) and
    reports success. When the engine cannot insert right now — IBus:
    Rechka is not the active input source in any focused window — text
    is queued and flushed, in order, when insertion becomes possible.
    """

    def __init__(self, deliver: Callable[[str], bool], queue_detail: str):
        self._deliver = deliver
        self._queue_detail = queue_detail
        self._lock = threading.Lock()
        self._focused = False
        self._pending: list[str] = []

    @property
    def focused(self) -> bool:
        with self._lock:
            return self._focused

    @property
    def queued(self) -> int:
        with self._lock:
            return len(self._pending)

    def commit(self, text: str) -> dict[str, Any]:
        if not text:
            return {"ok": True, "committed": True, "detail": "empty text ignored"}
        with self._lock:
            # Always append first so queued text can never be overtaken.
            self._pending.append(text)
            if self._focused:
                self._flush_locked()
            committed = not self._pending  # ours is last: gone == delivered
            return {
                "ok": True,
                "committed": committed,
                "queued": len(self._pending),
                "detail": "committed" if committed else self._queue_detail,
            }

    def _flush_locked(self) -> int:
        flushed = 0
        while self._pending:
            if not self._deliver(self._pending[0]):
                break
            self._pending.pop(0)
            flushed += 1
        return flushed

    def set_focused(self, focused: bool) -> int:
        """Update insertion availability; flush the queue. Returns flush count."""
        with self._lock:
            self._focused = focused
            return self._flush_locked() if focused else 0

    def ping_state(self) -> dict[str, Any]:
        return {"focused": self.focused, "queued": self.queued}
