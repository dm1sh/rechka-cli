"""Decoding audio/video files to 16 kHz mono float32 samples.

WAV is decoded with the standard library; anything else (mp3, m4a, mp4…)
goes through the system ffmpeg if present — the Linux counterpart of the
Android MediaCodec + swresample path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

TARGET_RATE = 16000


class AudioDecodeError(RuntimeError):
    pass


def decode_file(path: Path) -> np.ndarray:
    """Return the whole file as float32 mono samples at 16 kHz."""
    if not path.is_file():
        raise AudioDecodeError(f"No such file: {path}")
    if path.suffix.lower() == ".wav":
        samples, rate = _decode_wav(path)
        if rate == TARGET_RATE:
            return samples
        via_ffmpeg = _decode_ffmpeg(path)
        if via_ffmpeg is not None:
            return via_ffmpeg
        print(
            f"rechka: {path.name} is {rate} Hz and ffmpeg is unavailable; "
            "using linear resampling (quality may suffer).",
            file=sys.stderr,
        )
        return _resample_linear(samples, rate, TARGET_RATE)
    decoded = _decode_ffmpeg(path)
    if decoded is None:
        raise AudioDecodeError(
            f"Cannot decode {path.name}: ffmpeg not found. Install ffmpeg, "
            "or provide a 16 kHz mono WAV file."
        )
    return decoded


def _decode_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDecodeError(f"Cannot read WAV {path.name}: {exc}") from exc

    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError(f"Unsupported WAV sample width: {width} bytes")
    if channels > 1:
        usable = data.size - (data.size % channels)
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data, rate


def _decode_ffmpeg(path: Path) -> np.ndarray | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(TARGET_RATE), "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = message[-1] if message else "unknown ffmpeg error"
        raise AudioDecodeError(f"ffmpeg failed on {path.name}: {tail}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.size == 0:
        return samples
    duration = samples.size / src_rate
    target_len = max(1, int(duration * dst_rate))
    positions = np.linspace(0, samples.size - 1, target_len)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
