import threading
import time
from pathlib import Path

import pytest

from rechka import im
from rechka.config import Config
from rechka.engine_server import CommitBridge, EngineServer
from rechka.fcitx5_engine import choose_typing_tool
from rechka.im import EngineClient, EngineUnavailableError, start_engine


# -- CommitBridge -----------------------------------------------------------

def test_bridge_queues_until_focused():
    delivered: list[str] = []
    bridge = CommitBridge(lambda text: delivered.append(text) or True, "not focused")
    ack = bridge.commit("раз")
    assert ack["committed"] is False and ack["queued"] == 1
    assert delivered == []
    flushed = bridge.set_focused(True)
    assert flushed == 1 and delivered == ["раз"]
    assert bridge.queued == 0


def test_bridge_preserves_order_after_failed_delivery():
    delivered: list[str] = []
    state = {"fail": True}

    def deliver(text):
        if state["fail"]:
            return False
        delivered.append(text)
        return True

    bridge = CommitBridge(deliver, "cannot deliver")
    bridge.set_focused(True)
    assert bridge.commit("первое")["committed"] is False
    state["fail"] = False
    ack = bridge.commit("второе")
    assert ack["committed"] is True
    assert delivered == ["первое", "второе"]  # earlier text never overtaken


def test_bridge_focus_out_requeues():
    delivered: list[str] = []
    bridge = CommitBridge(lambda text: delivered.append(text) or True, "not focused")
    bridge.set_focused(True)
    bridge.commit("один")
    bridge.set_focused(False)
    ack = bridge.commit("Norbert")
    assert ack["committed"] is False and bridge.queued == 1
    assert delivered == ["один"]


# -- typing tool selection ----------------------------------------------------

def test_typing_tool_wayland_prefers_ydotool():
    env = {"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0"}
    which = lambda name: f"/usr/bin/{name}" if name in ("ydotool", "xdotool") else None
    assert choose_typing_tool(env, which) == "ydotool"


def test_typing_tool_x11_uses_xdotool():
    env = {"DISPLAY": ":0"}
    which = lambda name: "/usr/bin/xdotool" if name == "xdotool" else None
    assert choose_typing_tool(env, which) == "xdotool"


def test_typing_tool_none_available():
    assert choose_typing_tool({}, lambda name: None) is None
    assert choose_typing_tool({"WAYLAND_DISPLAY": "w"}, lambda name: None) is None


# -- engine socket round-trip ---------------------------------------------------

def test_engine_server_client_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    delivered: list[str] = []
    bridge = CommitBridge(lambda text: delivered.append(text) or True, "not focused")
    bridge.set_focused(True)
    server = EngineServer("fcitx5", bridge.commit, on_ping=bridge.ping_state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = EngineClient("fcitx5")
    deadline = time.monotonic() + 5
    while not client.socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        ping = client.ping()
        assert ping["ok"] and ping["framework"] == "fcitx5"
        ack = client.commit("привет")
        assert ack["committed"] is True and delivered == ["привет"]
        assert client.shutdown()["ok"]
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_engine_client_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(EngineUnavailableError):
        EngineClient("ibus").ping()


def test_start_engine_none_is_skipped():
    cfg = Config()
    status = start_engine("none", cfg)
    assert status.started is False
    assert "skipped" in status.detail


def test_ibus_prerequisites_report_unavailable_reason():
    # In environments without PyGObject the detail must say what to install;
    # where bindings exist the check simply passes.
    ok, detail = im.engine_prerequisites("ibus")
    if not ok:
        assert (
            "python3-gobject" in detail and "gir1.2-ibus-1.0" in detail
        ) or "DBUS_SESSION_BUS_ADDRESS" in detail


# -- IBus component installation ---------------------------------------------

def _isolated_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("IBUS_COMPONENT_PATH", raising=False)


def test_component_xml_names_engine_and_exec():
    xml = im.component_xml("/opt/rechka/ibus-engine --flag 'a&b'")
    assert "<name>org.freedesktop.IBus.Rechka</name>" in xml
    assert "<name>rechka</name>" in xml
    assert "a&amp;b" in xml  # the exec command is XML-escaped


def test_install_and_uninstall_user_component(tmp_path, monkeypatch):
    _isolated_xdg(tmp_path, monkeypatch)
    result = im.install_ibus_component(Config())
    assert result.component_path.exists()
    xml = result.component_path.read_text(encoding="utf-8")
    assert "<name>rechka</name>" in xml
    assert result.launcher_path is not None and result.launcher_path.exists()
    launcher = result.launcher_path.read_text(encoding="utf-8")
    assert "exec_by_ibus=True" in launcher
    assert result.dropin_path is not None and result.dropin_path.exists()
    dropin = result.dropin_path.read_text(encoding="utf-8")
    assert "/usr/share/ibus/component" in dropin
    assert str(result.component_path.parent) in dropin
    assert "--cache none" in dropin  # Fedora system cache must not shadow it
    removed = im.uninstall_ibus_component()
    assert result.component_path in removed
    assert not result.component_path.exists()
    assert not result.launcher_path.exists()


def test_ibus_engine_command_prefers_current_python(monkeypatch):
    import sys

    monkeypatch.setattr(im, "_ibus_python", lambda: sys.executable)
    argv, detail = im._ibus_engine_command(Config())
    assert argv[:4] == [sys.executable, "-m", "rechka", "im-engine"]
    assert "PyGObject" in detail


def test_ibus_engine_command_falls_back_to_distro_python(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(im, "_ibus_python", lambda: "/usr/bin/python3")
    argv, detail = im._ibus_engine_command(Config())
    assert argv[0] == "/usr/bin/python3"
    launcher = Path(argv[1])
    assert launcher.exists()
    assert "exec_by_ibus=False" in launcher.read_text(encoding="utf-8")
    assert "distro Python fallback" in detail


def test_stop_engine_leaves_external_engine_alone(monkeypatch):
    class ExplodingClient:
        def __init__(self, framework):
            raise AssertionError("external engine must not be contacted")

    monkeypatch.setattr(im, "EngineClient", ExplodingClient)
    status = im.EngineStatus("ibus", True, "engine already running", process=None)
    im.stop_engine(status)  # must not raise


# -- IBus automatic input switching -------------------------------------------

class _FakeInputState:
    def __init__(self, current="xkb:ru::rus"):
        self.current = current
        self.calls: list[str] = []

    def get(self):
        return self.current

    def set(self, name):
        self.calls.append(name)
        self.current = name
        return True


def _switcher(state):
    return im.IBusEngineSwitcher(get_current=state.get, set_current=state.set)


def test_switcher_switches_to_rechka_and_back():
    state = _FakeInputState()
    switcher = _switcher(state)
    result = switcher.activate()
    assert result.ok and state.current == "rechka"
    assert switcher.active and switcher.previous == "xkb:ru::rus"
    result = switcher.restore()
    assert result.ok and state.current == "xkb:ru::rus"
    assert not switcher.active and switcher.previous is None


def test_switcher_leaves_users_manual_choice_alone():
    state = _FakeInputState()
    switcher = _switcher(state)
    switcher.activate()
    state.current = "xkb:us::eng"  # user switched away during dictation
    result = switcher.restore()
    assert result.ok and state.current == "xkb:us::eng"
    assert not switcher.active


def test_switcher_does_not_adopt_rechka_the_user_already_had():
    state = _FakeInputState(current="rechka")
    switcher = _switcher(state)
    result = switcher.activate()
    assert result.ok and not switcher.active
    assert state.calls == []  # nothing switched, so nothing to restore


def test_switcher_reports_failed_switch_without_raising():
    state = _FakeInputState()
    switcher = im.IBusEngineSwitcher(get_current=state.get, set_current=lambda name: False)
    result = switcher.activate()
    assert result.ok is False
    assert not switcher.active


def test_set_current_ibus_engine_verifies_despite_nonzero_cli_exit(monkeypatch):
    class Result:
        returncode = 1  # e.g. follow-up setxkbmap failed after switching

    monkeypatch.setattr(im.shutil, "which", lambda name: "/usr/bin/ibus")
    monkeypatch.setattr(im.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.setattr(im, "get_current_ibus_engine", lambda: "rechka")
    monkeypatch.setattr(im.time, "monotonic", lambda: 0.0)
    assert im.set_current_ibus_engine("rechka") is True


# -- GNOME input-source list (im hide / im unhide) ---------------------------

class _FakeGsettings:
    """In-memory gsettings stand-in keyed by the schema key name."""

    def __init__(self, values):
        self.values = dict(values)
        self.sets: list[tuple[str, str]] = []

    def __call__(self, argv):
        import subprocess

        key = argv[3]
        if argv[1] == "get":
            return subprocess.CompletedProcess(argv, 0, self.values.get(key, "@as []"), "")
        self.sets.append((key, argv[4]))
        self.values[key] = argv[4]
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_parse_and_format_gnome_sources():
    raw = "[('xkb', 'us'), ('xkb', 'ru'), ('ibus', 'rechka')]"
    parsed = im.parse_gnome_sources(raw)
    assert parsed == [("xkb", "us"), ("xkb", "ru"), ("ibus", "rechka")]
    assert im.format_gnome_sources(parsed) == raw
    assert im.parse_gnome_sources("@as []") is None
    assert im.parse_gnome_sources("not-a-list") is None


def test_hide_removes_rechka_from_sources_and_mru():
    fake = _FakeGsettings({
        "sources": "[('xkb', 'us'), ('ibus', 'rechka')]",
        "mru-sources": "[('ibus', 'rechka'), ('xkb', 'us')]",
    })
    result = im.set_gnome_engine_listed("rechka", listed=False, runner=fake)
    assert result.ok
    assert fake.values["sources"] == "[('xkb', 'us')]"
    assert fake.values["mru-sources"] == "[('xkb', 'us')]"
    # Hiding again is a no-op success.
    result = im.set_gnome_engine_listed("rechka", listed=False, runner=fake)
    assert result.ok and "nothing to change" in result.detail
    assert fake.sets == [
        ("sources", "[('xkb', 'us')]"),
        ("mru-sources", "[('xkb', 'us')]"),
    ]


def test_unhide_adds_rechka_back():
    fake = _FakeGsettings({"sources": "[('xkb', 'us')]", "mru-sources": "[('xkb', 'us')]"})
    result = im.set_gnome_engine_listed("rechka", listed=True, runner=fake)
    assert result.ok
    assert fake.values["sources"] == "[('xkb', 'us'), ('ibus', 'rechka')]"
    result = im.set_gnome_engine_listed("rechka", listed=True, runner=fake)
    assert result.ok and "already" in result.detail


def test_hide_reports_gsettings_failure():
    import subprocess

    def failing(argv):
        return subprocess.CompletedProcess(argv, 1, "", "No such schema")

    result = im.set_gnome_engine_listed("rechka", listed=False, runner=failing)
    assert not result.ok and "No such schema" in result.detail

