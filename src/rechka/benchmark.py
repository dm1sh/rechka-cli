"""`rechka benchmark` — startup and offline transcription speed.

Reports:
- startup time: model loading (sherpa-onnx import + recognizer build);
- ahead-of-time ratio for the full pipeline with Silero VAD segmentation
  and without it (fixed chunks): audio_seconds / elapsed_seconds, i.e. how
  many times faster than realtime the file was transcribed (RTF = 1/ratio);
- time to first output per mode: segmentation + recognition up to the
  first segment that yields text — when `rechka transcribe` would print
  its first line. Startup and file decode are excluded (reported above).

The recognizer is loaded once and shared by both modes. Per-mode pipeline
time is segmentation + recognition of the decoded audio; model loading is
reported separately as startup. Segmentation runs before the recognizer is
created (same order as `rechka transcribe`). (An earlier "<unk>"-spam bug
was once attributed to that ordering; the actual cause was reading VAD
segment samples after pop() freed them — see segmentation.py.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from .audiofile import decode_file
from .config import Config
from .dictionary import Dictionary
from .segmentation import segment_waveform
from .transcriber import Transcriber


def ahead_of_time_ratio(audio_seconds: float, elapsed_seconds: float) -> float:
    """Speed relative to realtime (>1 means faster than realtime)."""
    if elapsed_seconds <= 0:
        return float("inf")
    return audio_seconds / elapsed_seconds


@dataclass
class ModeResult:
    name: str
    segments: int
    segmentation_seconds: float
    recognition_seconds: float
    # Segmentation + recognition until the first non-empty segment text
    # (best across runs); None when no run produced any text.
    first_output_seconds: float | None
    ratio: float
    text: str

    @property
    def elapsed_seconds(self) -> float:
        return self.segmentation_seconds + self.recognition_seconds


@dataclass
class BenchmarkReport:
    file: Path
    audio_seconds: float
    startup_seconds: float
    decode_seconds: float
    runs: int
    modes: list[ModeResult]

    def render(self, show_text: bool = False) -> str:
        lines = [
            f"file:      {self.file}",
            f"audio:     {self.audio_seconds:.2f} s",
            f"startup:   {self.startup_seconds:.3f} s (model load, shared by modes)",
            f"decode:    {self.decode_seconds:.3f} s (once, not in mode timings)",
        ]
        for mode in self.modes:
            first = (
                f"{mode.first_output_seconds:.3f} s"
                if mode.first_output_seconds is not None
                else "n/a"
            )
            lines.append(
                f"{mode.name:<10} {mode.elapsed_seconds:8.3f} s  "
                f"{mode.ratio:7.2f}x realtime  (RTF {1.0 / mode.ratio:.3f}, "
                f"first output {first}, "
                f"{mode.segments} segments, seg {mode.segmentation_seconds:.3f}s + "
                f"recog {mode.recognition_seconds:.3f}s, best of {self.runs})"
            )
            if show_text:
                lines.append(f"  text: {mode.text}")
        return "\n".join(lines)


def _recognize_best(
    segments,
    transcriber: Transcriber,
    dictionary: Dictionary | None,
    runs: int,
) -> tuple[str, float, float | None]:
    """Best-of-N recognition over precomputed segments.

    Returns (text, total recognition seconds, recognition seconds until
    the first non-empty segment text or None if no run produced text).
    Totals and first-output times are each the best across runs.
    """
    best_text, best_elapsed = "", float("inf")
    best_first: float | None = None
    for _ in range(max(1, runs)):
        started = time.perf_counter()
        parts: list[str] = []
        first_elapsed: float | None = None
        for segment in segments:
            text = transcriber.transcribe(segment.samples)
            if dictionary is not None:
                text = dictionary.apply(text)
            if text:
                parts.append(text)
                if first_elapsed is None:
                    first_elapsed = time.perf_counter() - started
        elapsed = time.perf_counter() - started
        if elapsed < best_elapsed:
            best_text, best_elapsed = " ".join(parts), elapsed
        if first_elapsed is not None and (
            best_first is None or first_elapsed < best_first
        ):
            best_first = first_elapsed
    return best_text, best_elapsed, best_first


def run_benchmark(path: Path, cfg: Config, runs: int = 3) -> BenchmarkReport:
    started = time.perf_counter()
    samples = decode_file(path)
    decode_seconds = time.perf_counter() - started
    if samples.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    audio_seconds = samples.size / cfg.audio.sample_rate

    dictionary = (
        Dictionary.load(cfg.text.dictionary_path) if cfg.text.dictionary_enabled else None
    )

    # Segment both modes *before* loading the recognizer (production order
    # in pipeline.transcribe_file — see module docstring).
    vad_cfg = replace(cfg, vad=replace(cfg.vad, enabled=True, file_enabled=True))
    fixed_cfg = replace(cfg, vad=replace(cfg.vad, file_enabled=False))
    prepared: list[tuple[str, list, float]] = []
    for name, mode_cfg in (("vad", vad_cfg), ("no-vad", fixed_cfg)):
        started = time.perf_counter()
        segments = segment_waveform(samples, mode_cfg)
        prepared.append((name, segments, time.perf_counter() - started))

    started = time.perf_counter()
    transcriber = Transcriber(cfg)  # includes sherpa-onnx import + model load
    startup = time.perf_counter() - started

    modes: list[ModeResult] = []
    for name, segments, segmentation_seconds in prepared:
        text, recognition_seconds, first_recognition = _recognize_best(
            segments, transcriber, dictionary, runs
        )
        result = ModeResult(
            name=name,
            segments=len(segments),
            segmentation_seconds=segmentation_seconds,
            recognition_seconds=recognition_seconds,
            first_output_seconds=(
                segmentation_seconds + first_recognition
                if first_recognition is not None
                else None
            ),
            ratio=0.0,
            text=text,
        )
        result.ratio = ahead_of_time_ratio(audio_seconds, result.elapsed_seconds)
        modes.append(result)

    return BenchmarkReport(
        file=path,
        audio_seconds=audio_seconds,
        startup_seconds=startup,
        decode_seconds=decode_seconds,
        runs=max(1, runs),
        modes=modes,
    )
