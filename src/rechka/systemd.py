"""systemd --user unit for autostart at login.

Design constraints from the requirements: start in the background at login,
never block session initialization, and a failure must not break anything
else. A oneshot-free Type=simple user unit with Restart=on-failure gives
exactly that — systemd starts it asynchronously and the session proceeds
regardless of its state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import paths

UNIT_NAME = "rechka.service"


def unit_text(exec_start: str) -> str:
    return f"""[Unit]
Description=Rechka offline speech recognition daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def default_exec_start() -> str:
    found = shutil.which("rechka")
    if found:
        return found
    return str(Path.home() / ".local" / "bin" / "rechka")


def unit_path() -> Path:
    return paths.config_home() / "systemd" / "user" / UNIT_NAME


def install_unit(exec_start: str | None = None) -> Path:
    target = unit_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit_text(exec_start or default_exec_start()), encoding="utf-8")
    return target
