"""IBus engine process (`rechka im-engine ibus`, spawned by the daemon).

A real, transparent IBus engine: it registers the "rechka" component
with the session's IBus daemon while it runs, passes every key through
untouched (normal typing is unaffected), and inserts dictated text with
commit_text into the focused application whenever "Rechka" is the active
input source there. Text dictated while another source is active is
queued by the CommitBridge and inserted on the next focus-in.

Requires PyGObject with the IBus typelib (Fedora: python3-gobject and
ibus-libs; Debian/Ubuntu: python3-gi and gir1.2-ibus-1.0) visible to
this Python — checked up front so the daemon can report precisely that
instead of a dead engine.  When this module is run by the distro
/usr/bin/python3 (as the daemon does as a fallback), the distro packages
are enough; the uv environment itself does not need PyGObject.
"""

from __future__ import annotations

import os
import sys
import threading

from .config import Config
from .engine_server import CommitBridge, EngineServer

_ENGINE_NAME = "rechka"
_BUS_NAME = "org.freedesktop.IBus.Rechka"
_QUEUE_DETAIL = (
    "Rechka is not the active input source in the focused window; "
    "text queued — it will be inserted when Rechka gains focus "
    "(add 'Rechka' in GNOME Settings → Keyboard / ibus-setup and select it)"
)


def run(cfg: Config, *, exec_by_ibus: bool = False) -> int:
    try:
        import gi

        gi.require_version("IBus", "1.0")
        from gi.repository import GLib, IBus
    except Exception as exc:
        print(
            f"rechka im-engine: IBus bindings unavailable: {exc}. "
            "Install the IBus typelib (Fedora: ibus-libs; Debian/Ubuntu: "
            "gir1.2-ibus-1.0) and PyGObject for this Python (Fedora: "
            "python3-gobject; or run rechka with `uv run --extra ibus` / "
            "`--with PyGObject`), or set im.framework = \"fcitx5\"/\"none\".",
            file=sys.stderr,
        )
        return 1
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        print(
            "rechka im-engine: DBUS_SESSION_BUS_ADDRESS is not set; "
            "the IBus engine must run inside a desktop session.",
            file=sys.stderr,
        )
        return 1

    current: dict[str, object] = {"engine": None}  # focused Engine instance

    def _deliver(text: str) -> bool:
        engine = current["engine"]
        if engine is None:
            return False
        # commit_text must run on the GLib main thread.
        GLib.idle_add(engine.commit_text, IBus.Text.new_from_string(text))
        return True

    bridge = CommitBridge(deliver=_deliver, queue_detail=_QUEUE_DETAIL)

    class Engine(IBus.Engine):
        __gtype_name__ = "RechkaEngine"

        def __init__(self, bus_, object_path: str) -> None:
            props: dict[str, object] = {
                "connection": bus_.get_connection(),
                "object_path": object_path,
            }
            # IBus >= 1.5.27 can report a focus id; opt in when the
            # installed typelib exposes the property (as typing-booster
            # and other maintained Python engines do).
            if hasattr(IBus.Engine.props, "has_focus_id"):
                props["has_focus_id"] = True
            super().__init__(**props)

        def do_focus_in(self) -> None:
            current["engine"] = self
            bridge.set_focused(True)

        def do_focus_in_id(self, object_path: str, client: str) -> None:
            self.do_focus_in()

        def do_focus_out(self) -> None:
            if current["engine"] is self:
                current["engine"] = None
            bridge.set_focused(False)

        def do_focus_out_id(self, object_path: str) -> None:
            self.do_focus_out()

        def do_disable(self) -> None:
            self.do_focus_out()

        def do_process_key_event(self, keyval, keycode, state) -> bool:
            return False  # transparent: never consume keys

    class Factory(IBus.Factory):
        __gtype_name__ = "RechkaFactory"

        def __init__(self, bus_) -> None:
            self._bus = bus_
            self._engines: list[object] = []
            self._counter = 0
            super().__init__(
                connection=bus_.get_connection(),
                object_path=getattr(IBus, "PATH_FACTORY", "/org/freedesktop/IBus/Factory"),
            )

        def do_create_engine(self, name: str):
            if name != _ENGINE_NAME:
                raise ValueError(f"unknown engine: {name!r}")
            self._counter += 1
            path = f"/org/freedesktop/IBus/Engine/Rechka/{self._counter}"
            engine = Engine(self._bus, path)
            self._engines.append(engine)  # keep the Python wrapper alive
            return engine

    bus = IBus.Bus()
    if not bus.is_connected():  # GI name (the old python-ibus had connected())
        print(
            "rechka im-engine: cannot connect to the IBus daemon. "
            "Is ibus-daemon running in this session?",
            file=sys.stderr,
        )
        return 1

    loop = GLib.MainLoop()
    bus.connect("disconnected", lambda *_args: loop.quit())
    try:
        factory = Factory(bus)
    except Exception as exc:
        print(
            f"rechka im-engine: cannot create the IBus factory object: {exc}; "
            "the installed IBus bindings are incompatible.",
            file=sys.stderr,
        )
        return 1
    component = IBus.Component.new(
        _BUS_NAME,
        "Rechka voice dictation",
        "0.1.0",
        "MIT",
        "Rechka",
        "",
        "",
        "rechka",
    )
    component.add_engine(
        IBus.EngineDesc.new(
            _ENGINE_NAME,
            "Rechka",
            "Rechka offline voice dictation (Russian)",
            "ru",
            "MIT",
            "Rechka",
            "audio-input-microphone-symbolic",
            "default",
        )
    )
    if exec_by_ibus:
        # Launched by ibus-daemon from the installed component XML: the
        # daemon already owns the component description, so (as other
        # Python engines do) only claim the bus name.  Dynamic
        # registration here would add a duplicate component and the
        # engine would still be missing from the XML-backed list that
        # GNOME Settings reads.
        if bus.request_name(_BUS_NAME, 0) <= 0:
            print(
                "rechka im-engine: could not claim the IBus component name",
                file=sys.stderr,
            )
            return 1
        mode = "launched by IBus"
    else:
        # Started by `rechka serve` (or by hand).  Runtime registration
        # makes the engine activatable in this session, but GNOME
        # Settings discovers engines from component XML; see
        # `rechka im install` for the persistent installation.
        if hasattr(bus, "register_component"):
            try:
                bus.register_component(component)
            except Exception as exc:
                print(f"rechka im-engine: register_component failed: {exc}", file=sys.stderr)
        bus.request_name(_BUS_NAME, 0)
        mode = "registered"

    server = EngineServer("ibus", bridge.commit, on_ping=bridge.ping_state)
    server_thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="rechka-ibus-socket"
    )
    server_thread.start()
    print(
        f"rechka im-engine: IBus engine 'rechka' {mode}. Select 'Rechka' "
        "as the input source in the app you dictate into; other sources "
        "queue text until then.",
        file=sys.stderr,
    )
    try:
        loop.run()
    finally:
        server.request_stop()
    return 0
