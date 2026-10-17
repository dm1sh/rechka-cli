"""XDG-based locations.

Kept tiny and dependency-free on purpose. Linux is the primary target, but
all path resolution is isolated here so other platforms can map these
functions to their own conventions later without touching the rest.
"""

from __future__ import annotations

import os
from pathlib import Path


def config_home() -> Path:
    raw = os.environ.get("XDG_CONFIG_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".config"


def data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".local" / "share"


def runtime_dir() -> Path:
    raw = os.environ.get("XDG_RUNTIME_DIR")
    return Path(raw).expanduser() if raw else data_home() / "rechka"


def default_config_path() -> Path:
    override = os.environ.get("RECHKA_CONFIG")
    if override:
        return Path(override).expanduser()
    return config_home() / "rechka" / "rechka.toml"


def default_model_dir() -> Path:
    override = os.environ.get("RECHKA_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return data_home() / "rechka" / "models" / "gigaam-v3"


def default_socket_path() -> Path:
    return runtime_dir() / "rechka.sock"


def default_transcript_path() -> Path:
    return data_home() / "rechka" / "transcripts.txt"
