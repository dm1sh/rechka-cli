"""Output sinks: stdout, file, and input-method framework commit.

Every sink receives finalized segments as they are recognized, then one
finish() with the assembled text. The IM sink forwards segments to the
framework engine process (im.py / engine_server.py): a real IBus engine
when the session runs IBus, the typing engine for Fcitx5.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


class SinkError(RuntimeError):
    pass


class Sink:
    name = "base"

    def segment(self, text: str) -> None:
        raise NotImplementedError

    def finish(self, full_text: str) -> None:
        pass

    def close(self) -> None:
        pass


class StdoutSink(Sink):
    name = "stdout"

    def segment(self, text: str) -> None:
        print(text, flush=True)


class FileSink(Sink):
    name = "file"

    def __init__(self, path: Path, append: bool):
        self._path = path
        self._append = append
        self._parts: list[str] = []

    def segment(self, text: str) -> None:
        self._parts.append(text)

    def finish(self, full_text: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        with self._path.open(mode, encoding="utf-8") as handle:
            handle.write(full_text + "\n")


class ImSink(Sink):
    """Commit text into the focused application via an IM engine process.

    Connects to the already-running engine for the framework (started by
    the daemon), or starts one when given the config. Construction fails
    fast with SinkError when no engine can serve commits, so
    `transcribe -o ibus` errors before any decoding/model loading.

    Segments that the engine cannot deliver right now (e.g. the Rechka
    IBus source is not active in the focused window) raise SinkError:
    a one-shot CLI run must not silently drop text into a queue. The
    daemon's dictation path uses EngineClient directly and surfaces
    queueing in `rechka status` instead.
    """

    def __init__(self, framework: str, cfg: "Config | None" = None):
        from .im import EngineClient, EngineUnavailableError, start_engine, stop_engine

        self.name = framework
        self._framework = framework
        self._client = EngineClient(framework)
        self._started = None  # EngineStatus when *we* spawned the engine
        self._stop_engine = stop_engine
        try:
            self._client.ping()
            return
        except EngineUnavailableError:
            pass
        if cfg is None:
            raise SinkError(
                f"{framework} engine is not running. Start the daemon "
                "(`rechka serve`, it launches the engine) or check "
                f"`rechka check` for engine availability."
            )
        status = start_engine(framework, cfg)
        if not status.started:
            raise SinkError(f"Cannot start {framework} engine: {status.detail}")
        self._started = status
        try:
            self._client.ping()
        except EngineUnavailableError as exc:
            raise SinkError(f"Cannot reach {framework} engine: {exc}") from exc

    def segment(self, text: str) -> None:
        from .im import EngineUnavailableError

        try:
            ack = self._client.commit(text)
        except EngineUnavailableError as exc:
            raise SinkError(str(exc)) from exc
        if not ack.get("committed"):
            raise SinkError(
                f"{self._framework} engine could not insert the text: "
                f"{ack.get('detail', 'unknown reason')}"
            )

    def close(self) -> None:
        if self._started is not None:
            self._stop_engine(self._started)
            self._started = None


def sinks_for_output(
    output: str | None, source: Path, cfg: "Config | None" = None
) -> list[Sink]:
    """Resolve the -o/--output argument into sinks.

    None          -> stdout (the default)
    "-"           -> stdout
    ibus / fcitx5 -> (case-insensitive) the input-method engine sink
    existing dir  -> file "<source stem>.txt" inside that directory
    anything else -> that exact file path (overwritten)
    """
    if output is None:
        return [StdoutSink()]
    if output == "-":
        return [StdoutSink()]
    lowered = output.lower()
    if lowered in ("ibus", "fcitx5"):
        return [ImSink(lowered, cfg)]
    target = Path(output).expanduser()
    if target.is_dir():
        return [FileSink(target / f"{source.stem}.txt", append=False)]
    return [FileSink(target, append=False)]


def emit_segment(sinks: list[Sink], text: str) -> None:
    for sink in sinks:
        sink.segment(text)


def emit_finish(sinks: list[Sink], full_text: str) -> None:
    for sink in sinks:
        sink.finish(full_text)
