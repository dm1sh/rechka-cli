import os
import stat
import sys
import time

import numpy as np
import pytest

from rechka import mic
from rechka.config import Config
from rechka.mic import MicrophoneCapture, MicrophoneError, select_backend


def _cfg(backend="auto"):
    cfg = Config()
    cfg.audio.backend = backend
    return cfg


def test_pinned_backend_unavailable(monkeypatch):
    monkeypatch.setattr(mic, "backend_available", lambda name: False)
    with pytest.raises(MicrophoneError, match="sounddevice"):
        select_backend(_cfg("sounddevice"))


def test_auto_without_backends(monkeypatch):
    monkeypatch.setattr(mic, "backend_available", lambda name: False)
    with pytest.raises(MicrophoneError, match="No microphone capture backend"):
        select_backend(_cfg("auto"))


def test_auto_picks_first_available(monkeypatch):
    monkeypatch.setattr(
        mic, "backend_available", lambda name: name in ("pulse", "alsa")
    )
    assert select_backend(_cfg("auto")) == "pulse"


class _FakeBackend:
    def __init__(self, fail=False):
        self.started = False
        self._fail = fail

    def start(self, on_samples):
        if self._fail:
            raise RuntimeError("no such device")
        self.started = True

    def stop(self):
        self.started = False


def test_capture_lifecycle(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(mic, "_make_backend", lambda name, cfg: backend)
    monkeypatch.setattr(mic, "backend_available", lambda name: True)
    capture = MicrophoneCapture(_cfg("alsa"), lambda chunk: None)
    capture.start()
    assert capture.listening and capture.backend_name == "alsa"
    capture.start()  # idempotent
    capture.stop()
    assert not capture.listening and not backend.started
    capture.stop()  # idempotent


def test_capture_wraps_backend_errors(monkeypatch):
    monkeypatch.setattr(
        mic, "_make_backend", lambda name, cfg: _FakeBackend(fail=True)
    )
    monkeypatch.setattr(mic, "backend_available", lambda name: True)
    capture = MicrophoneCapture(_cfg("alsa"), lambda chunk: None)
    with pytest.raises(MicrophoneError, match="no such device"):
        capture.start()
    assert not capture.listening


def _write_fake_arecord(directory, samples: np.ndarray) -> None:
    script = directory / "arecord"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"data = {samples.astype('<i2').tobytes()!r}\n"
        "sys.stdout.buffer.write(data)\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def test_subprocess_backend_streams_windows(tmp_path, monkeypatch):
    expected = (np.arange(3 * 512, dtype=np.int16) % 2000) - 1000
    _write_fake_arecord(tmp_path, expected)
    monkeypatch.setenv("PATH", str(tmp_path))  # only our fake arecord exists
    chunks: list[np.ndarray] = []
    capture = MicrophoneCapture(_cfg("alsa"), chunks.append)
    capture.start()
    deadline = time.monotonic() + 10
    while sum(c.size for c in chunks) < expected.size and time.monotonic() < deadline:
        time.sleep(0.02)
    capture.stop()
    assert capture.backend_name == "alsa"
    assert all(c.size == 512 for c in chunks)
    received = np.concatenate(chunks)
    assert received.size == expected.size
    assert np.allclose(received, expected.astype(np.float32) / 32768.0, atol=1e-4)
    assert capture.chunks_received == 3
    assert capture.peak_level > 0.02  # seen audio, not silence
    assert capture.last_error is None  # clean exit (code 0) is not a failure
    assert os.name  # silence linters; PATH restore is monkeypatch's job


def test_subprocess_backend_failure_is_reported(tmp_path, monkeypatch):
    """A recorder dying on its own must surface, not vanish silently."""
    script = tmp_path / "arecord"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('fake arecord: device exploded', file=sys.stderr)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))
    capture = MicrophoneCapture(_cfg("alsa"), lambda chunk: None)
    capture.start()
    deadline = time.monotonic() + 10
    while capture.last_error is None and time.monotonic() < deadline:
        time.sleep(0.02)
    capture.stop()
    assert capture.last_error is not None and "code 3" in capture.last_error
    assert not capture.listening
    assert capture.chunks_received == 0


def test_pipewire_command_uses_raw_when_supported(monkeypatch):
    backend = mic._SubprocessBackend(_cfg("pipewire"), "pipewire")
    monkeypatch.setattr(mic, "pw_cat_supports_raw", lambda: True)
    assert "--raw" in backend._command()
    monkeypatch.setattr(mic, "pw_cat_supports_raw", lambda: False)
    assert "--raw" not in backend._command()


def test_pw_cat_raw_probe_reads_help(monkeypatch):
    class Result:
        stdout = "usage: pw-cat [options]  -a, --raw  RAW mode"
        stderr = ""

    monkeypatch.setattr(mic, "_pw_cat_raw_supported", None)
    monkeypatch.setattr(mic.subprocess, "run", lambda *a, **k: Result())
    assert mic.pw_cat_supports_raw() is True

    class OldResult:
        stdout = "usage: pw-cat [options]  --record"
        stderr = ""

    monkeypatch.setattr(mic, "_pw_cat_raw_supported", None)
    monkeypatch.setattr(mic.subprocess, "run", lambda *a, **k: OldResult())
    assert mic.pw_cat_supports_raw() is False
