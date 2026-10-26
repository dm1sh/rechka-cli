"""Microphone capture for dictation (`rechka start` / `toggle`).

Backends, in `auto` order (config audio.backend pins one):
- sounddevice — PortAudio via the optional `sounddevice` package
  (`uv sync --extra mic`); best device selection, lowest latency.
- pipewire    — `pw-cat --record` (PipeWire native systems).
- pulse       — `parec` (PulseAudio / pipewire-pulse systems).
- alsa        — `arecord` (bare ALSA systems).

All backends deliver float32 mono 16 kHz chunks of WINDOW_SAMPLES to the
callback, on a backend-owned thread. Capture failure raises
MicrophoneError with an actionable message; the daemon reports it and
keeps running.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from typing import Callable

import numpy as np

from .config import Config

WINDOW_SAMPLES = 512  # Silero window at 16 kHz, matches segmentation.py

OnSamples = Callable[[np.ndarray], None]

# backend name -> executable for the subprocess backends
_SUBPROCESS_BACKENDS = {
    "pipewire": "pw-cat",
    "pulse": "parec",
    "alsa": "arecord",
}
BACKEND_ORDER = ("sounddevice", "pipewire", "pulse", "alsa")


class MicrophoneError(RuntimeError):
    pass


def _sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return False
    return True


def backend_available(name: str) -> bool:
    if name == "sounddevice":
        return _sounddevice_available()
    executable = _SUBPROCESS_BACKENDS.get(name)
    return executable is not None and shutil.which(executable) is not None


def available_backends() -> list[str]:
    return [name for name in BACKEND_ORDER if backend_available(name)]


def select_backend(cfg: Config) -> str:
    """Resolve config audio.backend to a usable backend name."""
    wanted = cfg.audio.backend
    if wanted != "auto":
        if not backend_available(wanted):
            raise MicrophoneError(_unavailable_detail(wanted))
        return wanted
    for name in BACKEND_ORDER:
        if backend_available(name):
            return name
    raise MicrophoneError(
        "No microphone capture backend available. Install one of: "
        "the `sounddevice` extra (`uv sync --extra mic`, PortAudio), "
        "PipeWire's `pw-cat`, PulseAudio's `parec`, or ALSA's `arecord`."
    )


_pw_cat_raw_supported: bool | None = None


def pw_cat_supports_raw() -> bool:
    """Whether `pw-cat --raw` exists (PipeWire >= 1.0-ish; 0.3.x lacks it).

    Since PipeWire 1.4, `pw-cat --record ... -` (stdout) without --raw
    writes a Sun/AU container (header + big-endian samples) instead of
    raw PCM, which a raw-s16le reader hears as byte-swapped noise.
    Probed once via --help so old PipeWire keeps working unchanged.
    """
    global _pw_cat_raw_supported
    if _pw_cat_raw_supported is None:
        try:
            result = subprocess.run(
                ["pw-cat", "--help"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            _pw_cat_raw_supported = "--raw" in (result.stdout + result.stderr)
        except Exception:
            _pw_cat_raw_supported = False
    return _pw_cat_raw_supported


def _unavailable_detail(name: str) -> str:
    if name == "sounddevice":
        return (
            "audio.backend=sounddevice but sounddevice/PortAudio is not "
            "importable. Install the extra (`uv sync --extra mic`) and the "
            "system PortAudio library (e.g. Debian/Ubuntu: libportaudio2)."
        )
    executable = _SUBPROCESS_BACKENDS[name]
    return (
        f"audio.backend={name} but `{executable}` was not found on PATH. "
        f"Install it or set audio.backend = \"auto\"."
    )


class _SoundDeviceBackend:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._stream = None

    def start(self, on_samples: OnSamples) -> None:
        import sounddevice as sd

        device = self._cfg.audio.device
        device_arg: int | str | None = None
        if device and device != "default":
            device_arg = int(device) if device.isdigit() else device

        def _callback(indata, frames, time_info, status) -> None:
            if status:
                print(f"rechka: mic: {status}", flush=True)
            on_samples(np.array(indata[:, 0], dtype=np.float32, copy=True))

        self._stream = sd.InputStream(
            samplerate=self._cfg.audio.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=WINDOW_SAMPLES,
            device=device_arg,
            callback=_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class _SubprocessBackend:
    """Raw s16le PCM from a recorder process's stdout."""

    def __init__(self, cfg: Config, name: str):
        self._cfg = cfg
        self._name = name
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        # Set by MicrophoneCapture; called if the recorder dies on its own.
        self.on_error: Callable[[str], None] | None = None

    def _command(self) -> list[str]:
        rate = str(self._cfg.audio.sample_rate)
        device = self._cfg.audio.device
        if self._name == "pipewire":
            cmd = ["pw-cat", "--record", "--rate", rate, "--channels", "1",
                   "--format", "s16", "-"]
            if pw_cat_supports_raw():
                cmd.insert(2, "--raw")  # raw PCM to stdout, no AU container
            if device and device != "default":
                cmd += ["--target", device]
            return cmd
        if self._name == "pulse":
            cmd = ["parec", f"--rate={rate}", "--channels=1", "--format=s16le"]
            if device and device != "default":
                cmd.append(f"--device={device}")
            return cmd
        cmd = ["arecord", "-q", "-r", rate, "-f", "S16_LE", "-c", "1", "-t", "raw"]
        if device and device != "default":
            cmd += ["-D", device]
        return cmd

    def start(self, on_samples: OnSamples) -> None:
        self._stopping = False
        # stderr is inherited on purpose: recorder errors (device gone,
        # format rejected after a PipeWire upgrade, ...) must reach the
        # daemon's stderr/journal instead of vanishing into DEVNULL.
        self._process = subprocess.Popen(self._command(), stdout=subprocess.PIPE)
        self._thread = threading.Thread(
            target=self._pump, args=(on_samples,), daemon=True,
            name=f"rechka-mic-{self._name}",
        )
        self._thread.start()

    def _pump(self, on_samples: OnSamples) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        pending = np.empty(0, dtype=np.float32)
        while True:
            data = stream.read(4096)  # short reads are fine: pending re-aligns
            if not data:
                break  # recorder exited / was stopped
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            pending = np.concatenate([pending, samples])
            while pending.size >= WINDOW_SAMPLES:
                on_samples(np.array(pending[:WINDOW_SAMPLES], copy=True))
                pending = pending[WINDOW_SAMPLES:]
        if not self._stopping and self.on_error is not None:
            returncode = self._process.wait()
            if returncode != 0:
                self.on_error(
                    f"microphone recorder ({self._name}) exited unexpectedly "
                    f"with code {returncode}; its error output is above"
                )

    def stop(self) -> None:
        self._stopping = True
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None


def _make_backend(name: str, cfg: Config):
    if name == "sounddevice":
        return _SoundDeviceBackend(cfg)
    return _SubprocessBackend(cfg, name)


class MicrophoneCapture:
    """Owns one capture backend; thread-safe start/stop."""

    def __init__(self, cfg: Config, on_samples: OnSamples):
        self._cfg = cfg
        self._on_samples = on_samples
        self.listening = False
        self.backend_name: str | None = None
        self._backend = None
        # Diagnostics: why capture died, and whether audio is arriving
        # at all (chunks/peak distinguish "recorder dead" from "silence").
        self.last_error: str | None = None
        self.chunks_received = 0
        self.peak_level = 0.0

    def _deliver(self, chunk: np.ndarray) -> None:
        self.chunks_received += 1
        if chunk.size:
            peak = float(np.max(np.abs(chunk)))
            if peak > self.peak_level:
                self.peak_level = peak
        self._on_samples(chunk)

    def _backend_failed(self, message: str) -> None:
        self.last_error = message
        self.listening = False
        print(f"rechka: mic: {message}", file=sys.stderr, flush=True)

    def start(self) -> None:
        if self.listening:
            return
        name = select_backend(self._cfg)  # raises MicrophoneError
        backend = _make_backend(name, self._cfg)
        self.last_error = None
        self.chunks_received = 0
        self.peak_level = 0.0
        if hasattr(backend, "on_error"):
            backend.on_error = self._backend_failed
        try:
            backend.start(self._deliver)
        except MicrophoneError:
            raise
        except Exception as exc:  # PortAudio errors, Popen failures, ...
            raise MicrophoneError(
                f"Microphone capture via {name} failed: {exc}"
            ) from exc
        self._backend = backend
        self.backend_name = name
        self.listening = True

    def stop(self) -> None:
        backend, self._backend = self._backend, None
        self.listening = False
        if backend is not None:
            try:
                backend.stop()
            except Exception:
                pass  # stopping must never take the daemon down
