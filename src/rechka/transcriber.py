"""GigaAM v3 recognizer on sherpa-onnx (Linux build of the Android engine)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config
from .model import require_model

if TYPE_CHECKING:  # keep CLI startup cheap; sherpa loads lazily
    import numpy as np


class TranscriberError(RuntimeError):
    pass


class Transcriber:
    """One loaded recognizer; segment-at-a-time decoding, like the app."""

    def __init__(self, cfg: Config):
        require_model(cfg.model.dir)
        try:
            import sherpa_onnx  # noqa: F401  (presence check)
        except ImportError as exc:
            raise TranscriberError(
                "sherpa-onnx is not installed. Install the project with "
                "`uv sync` / `uv tool install rechka-cli` first."
            ) from exc
        self._cfg = cfg
        self._recognizer = self._build(cfg, sherpa_onnx)

    @staticmethod
    def _build(cfg: Config, sherpa_onnx):
        model_dir: Path = cfg.model.dir
        try:
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(model_dir / "gigaam_v3_e2e_rnnt_encoder_int8.onnx"),
                decoder=str(model_dir / "gigaam_v3_e2e_rnnt_decoder.onnx"),
                joiner=str(model_dir / "gigaam_v3_e2e_rnnt_joint.onnx"),
                tokens=str(model_dir / "gigaam_v3_e2e_rnnt_tokens.txt"),
                num_threads=cfg.model.threads,
                sample_rate=cfg.audio.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
            )
        except Exception as exc:  # native loader errors surface as RuntimeError
            raise TranscriberError(f"Failed to load GigaAM recognizer: {exc}") from exc

    def transcribe(self, samples: "np.ndarray") -> str:
        """Recognize one speech segment (float32, 16 kHz, mono)."""
        if samples.size == 0:
            return ""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._cfg.audio.sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()
