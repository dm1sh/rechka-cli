import time

import numpy as np
import pytest

from rechka.config import Config
from rechka.daemon import Daemon
from rechka.im import IBusEngineSwitcher
from rechka.transcriber import TranscriberError


class FakeMic:
    def __init__(self):
        self.listening = False
        self.backend_name = None
        self.on_samples = None

    def start(self):
        self.listening = True
        self.backend_name = "fake"

    def stop(self):
        self.listening = False

    def emit(self, samples):
        assert self.on_samples is not None
        self.on_samples(samples)


class FakeTranscriber:
    def transcribe(self, samples):
        return "привет" if samples.size >= 1600 else ""


def _daemon(tmp_path, commits, transcriber_factory=None):
    cfg = Config()
    cfg.model.dir = tmp_path
    cfg.im.framework = "none"
    cfg.daemon.start_im_engine = False
    mic = FakeMic()
    daemon = Daemon(
        cfg,
        mic=mic,
        transcriber_factory=transcriber_factory or FakeTranscriber,
        commit_fn=lambda text: commits.append(text) or {"ok": True, "committed": True},
    )
    mic.on_samples = daemon._on_samples
    return daemon, mic


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_dictation_round_trip(tmp_path):
    commits: list[str] = []
    daemon, mic = _daemon(tmp_path, commits)

    assert daemon.handle({"cmd": "toggle"}) == {"ok": True, "state": "listening"}
    mic.emit(np.zeros(16000, dtype=np.float32))  # 1 s, flushed as one segment
    assert daemon.handle({"cmd": "toggle"}) == {"ok": True, "state": "stopped"}

    assert _wait_for(lambda: commits == ["привет"])
    status = daemon.handle({"cmd": "status"})
    assert status["model_loaded"] is True
    assert status["last_text"] == "привет"
    assert status["segments_dictated"] == 1
    assert status["microphone"] == "stopped"


def test_start_reports_model_failure(tmp_path):
    commits: list[str] = []

    def broken_factory():
        raise TranscriberError("model missing")

    daemon, mic = _daemon(tmp_path, commits, transcriber_factory=broken_factory)
    response = daemon.handle({"cmd": "start"})
    assert response["ok"] is False
    assert response["state"] == "stopped"
    assert "model missing" in response["error"]
    assert not mic.listening  # mic must not open when the model cannot load


def test_stop_when_stopped_is_ok(tmp_path):
    daemon, _mic = _daemon(tmp_path, [])
    assert daemon.handle({"cmd": "stop"}) == {"ok": True, "state": "stopped"}
    assert daemon.handle({"cmd": "bogus"})["ok"] is False


def test_dictation_switches_ibus_input_and_restores_it(tmp_path):
    commits: list[str] = []
    state = {"current": "xkb:ru::rus"}

    def get_current():
        return state["current"]

    def set_current(name):
        state["current"] = name
        return True

    cfg = Config()
    cfg.model.dir = tmp_path
    cfg.im.framework = "ibus"
    cfg.daemon.start_im_engine = False
    mic = FakeMic()
    daemon = Daemon(
        cfg,
        mic=mic,
        transcriber_factory=FakeTranscriber,
        commit_fn=lambda text: commits.append(text) or {"ok": True, "committed": True},
        ibus_switcher=IBusEngineSwitcher(get_current=get_current, set_current=set_current),
    )
    mic.on_samples = daemon._on_samples

    assert daemon.handle({"cmd": "start"}) == {"ok": True, "state": "listening"}
    assert state["current"] == "rechka"
    status = daemon.handle({"cmd": "status"})
    assert status["im_switch_active"] is True
    assert status["im_previous_engine"] == "xkb:ru::rus"

    mic.emit(np.zeros(16000, dtype=np.float32))
    assert daemon.handle({"cmd": "stop"}) == {"ok": True, "state": "stopped"}

    # The restore happens after the worker flush commits the final phrase.
    assert _wait_for(lambda: commits == ["привет"] and state["current"] == "xkb:ru::rus")
    status = daemon.handle({"cmd": "status"})
    assert status["im_switch_active"] is False
    assert "switched input back" in status["im_switch_detail"]


def test_start_redetects_framework_that_appeared_after_daemon_start(tmp_path, monkeypatch):
    """Regression: under systemd the daemon may start before IBus and
    conclude "no framework"; a later start must re-detect, not stay mute."""
    import rechka.daemon as daemon_module
    from rechka.im import Detection, EngineStatus

    detections = iter([
        Detection("none", "daemon started before the graphical session"),
        Detection("ibus", "ibus-daemon appeared later"),
    ])
    monkeypatch.setattr(
        daemon_module, "detect_framework", lambda cfg: next(detections)
    )
    monkeypatch.setattr(
        daemon_module,
        "start_engine",
        lambda framework, cfg: EngineStatus(
            framework, framework != "none", "fake engine"
        ),
    )

    class FakeSwitcher:
        active = False
        previous = None

        def activate(self):
            return type("Result", (), {"detail": "fake switch"})()

        def restore(self):
            return type("Result", (), {"detail": "fake restore"})()

    monkeypatch.setattr(daemon_module, "IBusEngineSwitcher", FakeSwitcher)

    commits: list[str] = []
    cfg = Config()
    cfg.model.dir = tmp_path
    cfg.daemon.start_im_engine = True  # framework stays "auto"
    mic = FakeMic()
    daemon = Daemon(
        cfg,
        mic=mic,
        transcriber_factory=FakeTranscriber,
        commit_fn=lambda text: commits.append(text) or {"ok": True, "committed": True},
    )
    mic.on_samples = daemon._on_samples

    assert daemon.im_detection.framework == "none"
    assert daemon.handle({"cmd": "start"}) == {"ok": True, "state": "listening"}
    assert daemon.im_detection.framework == "ibus"
    assert daemon.im_status.started is True
    assert isinstance(daemon._ibus_switcher, FakeSwitcher)
    daemon.handle({"cmd": "stop"})
