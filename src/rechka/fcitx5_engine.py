"""Fcitx5-side engine process (`rechka im-engine fcitx5`).

A native Fcitx5 addon would be a compiled C++ module installed into
Fcitx5's addon directories — out of scope for a pip/uv-installed Python
package. The reasonable equivalent, and what dictation tools on Fcitx5
systems use in practice, is virtual-keyboard typing: the engine types
recognized text into the focused window with ydotool (Wayland, works
regardless of compositor), wtype (wlroots), or xdotool (X11). It is
started only when the session actually runs Fcitx5 (see im.py) and
behaves like the IBus engine from the daemon's point of view: same
socket, same commit acks, same queue-when-undeliverable semantics.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable, Mapping

from .config import Config
from .engine_server import CommitBridge, EngineServer

_QUEUE_DETAIL = (
    "no typing tool available (install ydotool for Wayland or xdotool "
    "for X11); text queued and retried on the next commit"
)


def _tool_argv(tool: str, text: str) -> list[str]:
    if tool == "ydotool":
        return ["ydotool", "type", "--key-delay=0", "--", text]
    if tool == "wtype":
        return ["wtype", "--", text]
    return ["xdotool", "type", "--clearmodifiers", "--", text]


def choose_typing_tool(
    env: Mapping[str, str], which: Callable[[str], str | None] = shutil.which
) -> str | None:
    """First usable typing tool for this session type, or None."""
    candidates: list[str] = []
    if env.get("WAYLAND_DISPLAY"):
        candidates += ["ydotool", "wtype"]
    if env.get("DISPLAY"):
        candidates += ["xdotool"]
    if env.get("WAYLAND_DISPLAY"):
        candidates += ["xdotool"]  # XWayland fallback
    else:
        candidates += ["ydotool"]  # works on X11 too, via uinput
    for tool in dict.fromkeys(candidates):  # dedupe, keep order
        if which(tool):
            return tool
    return None


def _type_text(text: str) -> bool:
    tool = choose_typing_tool(os.environ)
    if tool is None:
        return False
    try:
        result = subprocess.run(
            _tool_argv(tool, text),
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def run(cfg: Config) -> int:
    bridge = CommitBridge(deliver=_type_text, queue_detail=_QUEUE_DETAIL)
    bridge.set_focused(True)  # typing targets whatever window is focused
    server = EngineServer("fcitx5", bridge.commit, on_ping=bridge.ping_state)
    tool = choose_typing_tool(os.environ)
    if tool is None:
        print(
            "rechka im-engine: Fcitx5 typing engine running, but no typing "
            "tool found (install ydotool / wtype / xdotool) — commits will "
            "queue until one exists.",
            file=sys.stderr,
        )
    else:
        print(
            f"rechka im-engine: Fcitx5 typing engine running (tool: {tool}).",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
