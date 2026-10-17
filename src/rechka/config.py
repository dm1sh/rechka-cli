"""Configuration loading (TOML) with defaults.

Every value the Android app kept in SharedPreferences lives here instead.
The model directory is external data — it is never part of the package.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from . import paths


def _warn(message: str) -> None:
    print(f"rechka: warning: {message}", file=sys.stderr)


class ConfigError(Exception):
    """Raised for unreadable or invalid configuration."""


@dataclass
class ModelConfig:
    # Directory holding the four GigaAM files (see model.py). Required in
    # practice; the default keeps `rechka check` useful on a fresh install.
    dir: Path = field(default_factory=paths.default_model_dir)
    threads: int = 4


@dataclass
class AudioConfig:
    # Microphone capture for `rechka start` / `toggle` (see mic.py).
    device: str = "default"
    sample_rate: int = 16000
    # Capture backend: auto | sounddevice | pipewire | pulse | alsa.
    # auto = first available of sounddevice, pw-cat, parec, arecord.
    backend: str = "auto"


@dataclass
class VadConfig:
    enabled: bool = True
    # Empty means: look for silero_vad.onnx next to the GigaAM model files.
    model: Path | None = None
    threshold: float = 0.5
    # Mirrors the Android pause-length setting (short=0.5s is the default).
    min_silence_seconds: float = 0.5
    min_speech_seconds: float = 0.25
    max_speech_seconds: float = 30.0
    # File mode: segment with VAD, or use fixed chunks of chunk_seconds.
    file_enabled: bool = True
    chunk_seconds: float = 30.0


@dataclass
class TextConfig:
    dictionary_enabled: bool = False
    dictionary_path: Path = field(
        default_factory=lambda: paths.config_home() / "rechka" / "dictionary.txt"
    )
    # Append "[текст распознан с голоса]" after mic dictation (Android parity).
    voice_suffix: bool = False


@dataclass
class ImConfig:
    # "auto" starts whichever of ibus/fcitx5 the running session uses.
    framework: str = "auto"  # auto | ibus | fcitx5 | none


@dataclass
class DownloadConfig:
    # GitHub repository + release tag hosting the four GigaAM files.
    repository: str = "dm1sh/rechka"
    tag: str = "model-gigaam-v3"
    # Also fetch silero_vad.onnx (small) next to the model files.
    include_vad: bool = True


@dataclass
class DaemonConfig:
    socket_path: Path = field(default_factory=paths.default_socket_path)
    start_im_engine: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    text: TextConfig = field(default_factory=TextConfig)
    im: ImConfig = field(default_factory=ImConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    source: Path | None = None

    def vad_model_path(self) -> Path | None:
        """Explicit VAD model, else silero_vad.onnx beside the GigaAM files."""
        if self.vad.model is not None:
            return self.vad.model
        candidate = self.model.dir / "silero_vad.onnx"
        return candidate


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "audio": AudioConfig,
    "vad": VadConfig,
    "text": TextConfig,
    "im": ImConfig,
    "daemon": DaemonConfig,
    "download": DownloadConfig,
}

_PATH_FIELDS = {"dir", "model", "dictionary_path", "socket_path"}
_VALID_FRAMEWORKS = {"auto", "ibus", "fcitx5", "none"}


def _apply(section: Any, values: dict[str, Any], where: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            _warn(f"ignoring unknown key '{where}.{key}' in config file")
            continue
        if key in _PATH_FIELDS:
            value = Path(str(value)).expanduser() if value is not None else None
        setattr(section, key, value)


def load_config(path: Path | None = None) -> Config:
    """Load config from *path* (or the default location); defaults fill gaps."""
    cfg = Config()
    config_path = path or paths.default_config_path()
    cfg.source = config_path
    if not config_path.exists():
        return cfg
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read config {config_path}: {exc}") from exc

    for name, values in raw.items():
        section_type = _SECTIONS.get(name)
        if section_type is None:
            _warn(f"ignoring unknown section '[{name}]' in config file")
            continue
        if not isinstance(values, dict):
            raise ConfigError(f"Section '[{name}]' must be a table")
        _apply(getattr(cfg, name), values, name)

    if cfg.im.framework not in _VALID_FRAMEWORKS:
        raise ConfigError(f"im.framework must be one of {sorted(_VALID_FRAMEWORKS)}")
    if cfg.audio.sample_rate != 16000:
        # GigaAM/sherpa pipeline in the Android app is fixed at 16 kHz.
        raise ConfigError("audio.sample_rate must be 16000 (GigaAM requirement)")
    return cfg


def effective_summary(cfg: Config) -> dict[str, Any]:
    """JSON-friendly dump of the effective configuration (for `config show`)."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    result = {name: convert(getattr(cfg, name)) for name in _SECTIONS}
    result["source"] = str(cfg.source) if cfg.source else None
    return result
