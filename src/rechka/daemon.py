"""Daemon: owns model/mic/engine state and answers CLI control commands.

Protocol: one JSON object per line over a Unix socket, one JSON response
per line. Commands: ping, status, start, stop, toggle, shutdown.

Dictation flow: `start` loads the recognizer (once), opens the
microphone and streams audio through the VAD segmenter on a worker
thread; every completed segment is transcribed, dictionary-processed and
committed through the IM engine (queued text is reported in status).
`stop` flushes the segmenter so the trailing phrase is not lost.

The daemon is a plain background process (optionally a systemd *user*
service). It never blocks session startup: it binds its socket, logs
problems, and keeps serving control commands even if pieces (mic capture,
IM engines) are unavailable.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .config import Config
from .dictionary import Dictionary
from .im import (
    FRAMEWORK_IBUS,
    FRAMEWORK_NONE,
    EngineClient,
    EngineUnavailableError,
    IBusEngineSwitcher,
    detect_framework,
    start_engine,
    stop_engine,
)
from .mic import MicrophoneCapture, MicrophoneError
from .segmentation import StreamingSegmenter
from .transcriber import Transcriber, TranscriberError

VOICE_SUFFIX = "[текст распознан с голоса]"


class DaemonUnavailableError(RuntimeError):
    pass


def send_command(socket_path: Path, command: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Client side: send one command to a running daemon."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
    except OSError as exc:
        raise DaemonUnavailableError(
            f"Rechka daemon is not reachable at {socket_path}. "
            "Start it with `rechka serve` (or enable the systemd user unit)."
        ) from exc
    try:
        sock.sendall((json.dumps(command) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
    finally:
        sock.close()
    raw = b"".join(chunks).splitlines()
    if not raw:
        raise DaemonUnavailableError("Empty response from Rechka daemon")
    return json.loads(raw[0].decode("utf-8"))


class Daemon:
    def __init__(
        self,
        cfg: Config,
        *,
        mic: MicrophoneCapture | None = None,
        transcriber_factory: Callable[[], Transcriber] | None = None,
        commit_fn: Callable[[str], dict[str, Any]] | None = None,
        ibus_switcher: IBusEngineSwitcher | None = None,
    ):
        self.cfg = cfg
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.mic = mic if mic is not None else MicrophoneCapture(cfg, self._on_samples)
        self._transcriber_factory = (
            transcriber_factory if transcriber_factory is not None
            else (lambda: Transcriber(cfg))
        )
        self._commit_fn = commit_fn if commit_fn is not None else self._engine_commit
        self.state = "stopped"  # stopped | listening
        self._shutdown = threading.Event()
        self._transcriber: Transcriber | None = None
        self._dictionary: Dictionary | None = None
        self._segmenter: StreamingSegmenter | None = None
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        # Observability for `rechka status`
        self.last_text = ""
        self.last_error: str | None = None
        self.last_commit: dict[str, Any] = {}
        self.segments_dictated = 0
        detection = detect_framework(cfg)
        self.im_detection = detection
        self.im_status = (
            start_engine(detection.framework, cfg)
            if cfg.daemon.start_im_engine
            else None
        )
        # Automatic input switching for IBus dictation: `start` moves
        # the session to the Rechka engine and `stop` moves it back to
        # whatever was active before (after the final phrase has been
        # flushed). The switcher is only driven here; it never raises.
        self._ibus_switcher = (
            ibus_switcher
            if ibus_switcher is not None
            else (IBusEngineSwitcher() if detection.framework == FRAMEWORK_IBUS else None)
        )
        self._ibus_restore_pending = False
        self.im_switch_detail = ""

    # -- command handling -------------------------------------------------
    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        action = command.get("cmd", "")
        if action == "ping":
            return {"ok": True, "version": __version__}
        if action == "status":
            return {"ok": True, **self.status()}
        if action == "start":
            return self._set_listening(True)
        if action == "stop":
            return self._set_listening(False)
        if action == "toggle":
            return self._set_listening(self.state != "listening")
        if action == "shutdown":
            self._shutdown.set()
            return {"ok": True, "detail": "shutting down"}
        return {"ok": False, "error": f"unknown command: {action!r}"}

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "microphone": "listening" if self.mic.listening else "stopped",
            "mic_backend": self.mic.backend_name,
            "model_dir": str(self.cfg.model.dir),
            "model_loaded": self._transcriber is not None,
            "last_text": self.last_text,
            "last_error": self.last_error,
            "segments_dictated": self.segments_dictated,
            "im_framework": self.im_detection.framework,
            "im_evidence": self.im_detection.evidence,
            "im_engine_started": bool(self.im_status and self.im_status.started),
            "im_detail": self.im_status.detail if self.im_status else "disabled in config",
            "im_switch_active": bool(self._ibus_switcher and self._ibus_switcher.active),
            "im_previous_engine": (
                self._ibus_switcher.previous if self._ibus_switcher is not None else None
            ),
            "im_switch_detail": self.im_switch_detail,
            "im_last_commit": self.last_commit,
            "socket": str(self.cfg.daemon.socket_path),
            "pid": os.getpid(),
        }

    def _ensure_im_engine(self) -> None:
        """(Re)detect the IM framework and (re)start its engine.

        Detection normally happens once at daemon startup — but under
        systemd the daemon can start before GNOME/IBus, conclude "no
        framework" and stay input-less until restarted. So on every
        dictation start, re-check while no framework is known (config
        permitting), and retry the engine start if it previously failed.
        """
        if (
            self.cfg.im.framework == "auto"
            and self.im_detection.framework == FRAMEWORK_NONE
        ):
            detection = detect_framework(self.cfg)
            if detection.framework != FRAMEWORK_NONE:
                self.im_detection = detection
                self.im_status = None  # force the engine start below
                if (
                    detection.framework == FRAMEWORK_IBUS
                    and self._ibus_switcher is None
                ):
                    self._ibus_switcher = IBusEngineSwitcher()
        if (
            self.cfg.daemon.start_im_engine
            and self.im_detection.framework != FRAMEWORK_NONE
            and (self.im_status is None or not self.im_status.started)
        ):
            self.im_status = start_engine(self.im_detection.framework, self.cfg)

    def _set_listening(self, want_listening: bool) -> dict[str, Any]:
        with self._lock:
            if want_listening:
                if self.state == "listening":
                    return {"ok": True, "state": self.state}
                try:
                    self._ensure_im_engine()
                    self._ensure_pipeline()
                    self.mic.start()
                except (MicrophoneError, TranscriberError) as exc:
                    self.last_error = str(exc)
                    return {"ok": False, "state": self.state, "error": str(exc)}
                self.state = "listening"
                self.last_error = None
                self._ibus_restore_pending = False
                self._activate_ibus()
            else:
                if self.state != "listening":
                    return {"ok": True, "state": self.state}
                self.mic.stop()
                self._queue.put(("flush", None))
                self.state = "stopped"
                if self._ibus_switcher is not None and self._ibus_switcher.active:
                    # Restore only after the worker has flushed and
                    # committed the trailing phrase (see _worker_loop).
                    self._ibus_restore_pending = True
                    self.im_switch_detail = (
                        "Rechka input active; will switch back after the final phrase"
                    )
            return {"ok": True, "state": self.state}

    # -- IBus automatic input switching ------------------------------------
    def _activate_ibus(self) -> None:
        if self._ibus_switcher is None:
            return
        try:
            result = self._ibus_switcher.activate()
            self.im_switch_detail = result.detail
        except Exception as exc:  # switching must never break dictation
            self.im_switch_detail = f"input switch failed: {exc}"

    def _restore_ibus(self) -> None:
        self._ibus_restore_pending = False
        if self._ibus_switcher is None:
            return
        try:
            result = self._ibus_switcher.restore()
            self.im_switch_detail = result.detail
        except Exception as exc:  # switching must never break shutdown
            self.im_switch_detail = f"input restore failed: {exc}"

    # -- dictation pipeline (worker thread) -------------------------------
    def _ensure_pipeline(self) -> None:
        """Load recognizer/dictionary/segmenter once, before opening the mic."""
        # Segmenter (VAD) first, recognizer second — the Android recorder
        # builds/resets its VAD before fetching the transcriber too. The
        # historical reason (garbage first decode when the recognizer was
        # created first) is fixed in segmentation's drain order, but keep
        # the same construction order as the file pipeline regardless.
        self._segmenter = StreamingSegmenter(self.cfg)
        if self._transcriber is None:
            self._transcriber = self._transcriber_factory()
        self._dictionary = (
            Dictionary.load(self.cfg.text.dictionary_path)
            if self.cfg.text.dictionary_enabled
            else None
        )
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True, name="rechka-dictation"
            )
            self._worker.start()

    def _on_samples(self, chunk: np.ndarray) -> None:
        if self.state == "listening":
            self._queue.put(("samples", chunk))

    def _worker_loop(self) -> None:
        session_committed = False  # any text inserted in this session yet
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "samples":
                    for segment in self._segmenter.feed(payload):
                        session_committed = self._dictate(segment, session_committed)
                elif kind == "flush":
                    try:
                        for segment in self._segmenter.flush():
                            session_committed = self._dictate(segment, session_committed)
                        if session_committed and self.cfg.text.voice_suffix:
                            self._commit(f" {VOICE_SUFFIX}")
                        session_committed = False
                    finally:
                        if self._ibus_restore_pending:
                            self._restore_ibus()
            except Exception as exc:  # a bad segment must not kill dictation
                self.last_error = str(exc)
                print(f"rechka: dictation: {exc}", file=sys.stderr, flush=True)

    def _dictate(self, segment, session_committed: bool) -> bool:
        text = self._transcriber.transcribe(segment.samples)
        if self._dictionary is not None:
            text = self._dictionary.apply(text)
        if not text:
            return session_committed
        self.last_text = text
        self.segments_dictated += 1
        # Segments are inserted one by one; join them like the file pipeline.
        self._commit(f" {text}" if session_committed else text)
        return True

    def _commit(self, text: str) -> None:
        self.last_commit = self._commit_fn(text)

    def _engine_commit(self, text: str) -> dict[str, Any]:
        framework = self.im_detection.framework
        if framework == "none" or self.im_status is None or not self.im_status.started:
            return {
                "ok": False,
                "committed": False,
                "detail": "no IM engine available; text not inserted "
                "(see im_detail in status)",
            }
        try:
            return EngineClient(framework).commit(text)
        except EngineUnavailableError as exc:
            return {"ok": False, "committed": False, "detail": str(exc)}

    # -- serving ----------------------------------------------------------
    def serve_forever(self) -> None:
        socket_path = self.cfg.daemon.socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()  # stale socket from a previous run

        if self.im_status is not None:
            print(f"rechka: IM: {self.im_status.detail}", file=sys.stderr)
        print(
            f"rechka: daemon listening on {socket_path} "
            f"(framework={self.im_detection.framework} via {self.im_detection.evidence})",
            file=sys.stderr,
        )

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(16)
        server.settimeout(0.5)

        def _signal(_signum, _frame):
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _signal)
        signal.signal(signal.SIGINT, _signal)

        try:
            while not self._shutdown.is_set():
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
            self.mic.stop()
            if self._ibus_switcher is not None and self._ibus_switcher.active:
                self._restore_ibus()
            stop_engine(self.im_status)
            server.close()
            if socket_path.exists():
                socket_path.unlink()
            print("rechka: daemon stopped.", file=sys.stderr)

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            buffer = b""
            while b"\n" not in buffer:
                data = conn.recv(65536)
                if not data:
                    return
                buffer += data
            try:
                command = json.loads(buffer.splitlines()[0].decode("utf-8"))
                response = self.handle(command)
            except Exception as exc:  # keep the daemon alive on bad input
                response = {"ok": False, "error": f"bad request: {exc}"}
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
