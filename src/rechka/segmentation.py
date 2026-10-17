"""Splitting decoded audio into recognition segments.

Two strategies, mirroring the Android app:
- VAD (Silero via sherpa-onnx) when enabled and the VAD model is available;
- fixed chunks of vad.chunk_seconds otherwise / as fallback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    import numpy as np

WINDOW_SAMPLES = 512  # Silero window at 16 kHz, same as the Android recorder
# The VAD notices speech a little late, so its segment start can clip the
# first phonemes. Prepend a few windows of the audio that immediately
# precedes the detected segment (5 windows = 160 ms) — but never audio
# that was already sent to the recognizer with the previous segment.
PRE_ROLL_WINDOWS = 5
PRE_ROLL_SAMPLES = WINDOW_SAMPLES * PRE_ROLL_WINDOWS


@dataclass
class Segment:
    samples: "np.ndarray"  # float32 mono 16 kHz
    end_sample: int  # offset just past this segment in the source waveform


def segment_waveform(samples: "np.ndarray", cfg: Config) -> list[Segment]:
    if cfg.vad.enabled and cfg.vad.file_enabled:
        vad_path = cfg.vad_model_path()
        if vad_path is not None and Path(vad_path).is_file():
            return _vad_segments(samples, cfg, Path(vad_path))
        print(
            "rechka: VAD model not found "
            f"({vad_path or cfg.model.dir / 'silero_vad.onnx'}); "
            "falling back to fixed chunks. Set vad.model or vad.file_enabled=false "
            "to silence this.",
            file=sys.stderr,
        )
    return _fixed_segments(samples, cfg)


def _fixed_segments(samples: "np.ndarray", cfg: Config) -> list[Segment]:
    chunk = max(1, int(cfg.vad.chunk_seconds * cfg.audio.sample_rate))
    return [
        Segment(samples=samples[start : start + chunk], end_sample=min(start + chunk, samples.size))
        for start in range(0, samples.size, chunk)
    ]


def _vad_segments(samples: "np.ndarray", cfg: Config, vad_path: Path) -> list[Segment]:
    import numpy as np

    try:
        import sherpa_onnx
    except ImportError:
        return _fixed_segments(samples, cfg)

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = str(vad_path)
    vad_config.silero_vad.threshold = cfg.vad.threshold
    vad_config.silero_vad.min_silence_duration = cfg.vad.min_silence_seconds
    vad_config.silero_vad.min_speech_duration = cfg.vad.min_speech_seconds
    vad_config.silero_vad.window_size = WINDOW_SAMPLES
    vad_config.silero_vad.max_speech_duration = cfg.vad.max_speech_seconds
    vad_config.sample_rate = cfg.audio.sample_rate

    detector_cls = getattr(sherpa_onnx, "VoiceActivityDetector", None)
    if detector_cls is None:
        return _fixed_segments(samples, cfg)
    vad = detector_cls(vad_config, buffer_size_in_seconds=120)

    segments: list[Segment] = []
    raw_lengths: list[int] = []
    processed_end = 0  # source offset just past the last emitted segment
    consumed = 0  # samples fed so far; VAD segment starts are relative to its buffer

    def drain() -> None:
        nonlocal processed_end
        while not vad.empty():
            piece = vad.front
            # piece.samples is a view onto the native segment owned by the
            # detector queue: pop() frees it, and reading the samples after
            # pop() returns freed-memory garbage (intermittently huge float
            # values that make the recognizer emit "<unk><unk>...").
            # Copy BEFORE pop().
            piece_samples = np.array(piece.samples, dtype=np.float32, copy=True)
            start_attr = getattr(piece, "start", None)
            vad.pop()
            if piece_samples.size == 0:
                continue
            # Track absolute end by cumulative segment lengths; the VAD
            # buffer starts at the first fed sample for our one-shot use.
            # Silence between segments means cumulative lengths undercount;
            # piece.start (samples @16k) is exposed by newer bindings — it
            # was read above, before pop() freed the segment.
            if isinstance(start_attr, (int, float)):
                absolute_start = int(start_attr)
            else:
                absolute_start = processed_end
            vad_length = len(piece_samples)
            segment_end = absolute_start + vad_length
            # Pre-roll: the recognizer also gets the few windows just
            # before the detected start, clamped to the end of the
            # previous segment so nothing is recognized twice.
            expanded_start = max(
                absolute_start - PRE_ROLL_SAMPLES, processed_end, 0
            )
            if expanded_start < absolute_start:
                prefix = np.asarray(
                    samples[expanded_start:absolute_start], dtype=np.float32
                )
                if prefix.size:
                    piece_samples = np.concatenate([prefix, piece_samples])
            segments.append(
                Segment(samples=piece_samples, end_sample=segment_end)
            )
            raw_lengths.append(vad_length)
            processed_end = max(processed_end, segment_end)

    for offset in range(0, samples.size - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        vad.accept_waveform(samples[offset : offset + WINDOW_SAMPLES])
        consumed = offset + WINDOW_SAMPLES
        drain()
    # Trailing partial window + VAD tail (last phrase must not be lost).
    if samples.size > consumed:
        import numpy as np

        tail = samples[consumed:]
        padded = np.pad(tail, (0, WINDOW_SAMPLES - tail.size))
        vad.accept_waveform(padded)
        drain()
    vad.flush()
    drain()
    # Keep the historical minimum-length selection on the *detected*
    # segment length, so pre-roll cannot promote a click into a segment.
    minimum = cfg.audio.sample_rate // 4
    return [
        segment
        for segment, raw_length in zip(segments, raw_lengths)
        if raw_length >= minimum
    ]


class StreamingSegmenter:
    """Incremental segmentation for live microphone capture.

    Same Silero VAD settings as file mode when the VAD model is available;
    otherwise buffers audio and cuts at vad.max_speech_seconds (and on
    flush). feed()/flush() return completed Segments (float32, 16 kHz,
    end_sample relative to the start of the stream).
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._fed = 0  # samples accepted so far (stream position)
        self._vad = None
        self._window_rest = None  # partial VAD window carried between feeds
        self._chunks: list = []  # fallback buffer (no VAD)
        self._buffered = 0
        # Raw-audio history for VAD pre-roll (see PRE_ROLL_SAMPLES): the
        # last emitted segment's end, plus recent fed samples indexed by
        # their absolute stream position (_history_start).
        self._last_processed_end = 0
        self._history: list = []
        self._history_start = 0
        self._history_samples = 0
        if cfg.vad.enabled:
            vad_path = cfg.vad_model_path()
            if vad_path is not None and Path(vad_path).is_file():
                try:
                    self._vad = self._build_vad(Path(vad_path))
                except Exception as exc:  # sherpa missing/broken -> fallback
                    print(
                        f"rechka: VAD unavailable for mic capture ({exc}); "
                        "using fixed-length segments.",
                        file=sys.stderr,
                    )
                    self._vad = None

    @property
    def using_vad(self) -> bool:
        return self._vad is not None

    def _build_vad(self, vad_path: Path):
        import sherpa_onnx

        cfg = self._cfg
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(vad_path)
        vad_config.silero_vad.threshold = cfg.vad.threshold
        vad_config.silero_vad.min_silence_duration = cfg.vad.min_silence_seconds
        vad_config.silero_vad.min_speech_duration = cfg.vad.min_speech_seconds
        vad_config.silero_vad.window_size = WINDOW_SAMPLES
        vad_config.silero_vad.max_speech_duration = cfg.vad.max_speech_seconds
        vad_config.sample_rate = cfg.audio.sample_rate
        detector_cls = getattr(sherpa_onnx, "VoiceActivityDetector", None)
        if detector_cls is None:
            raise RuntimeError("sherpa_onnx has no VoiceActivityDetector")
        return detector_cls(vad_config, buffer_size_in_seconds=120)

    def feed(self, samples) -> list[Segment]:
        """Accept one capture chunk; return segments completed by it."""
        import numpy as np

        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.size == 0:
            return []
        self._fed += int(chunk.size)
        if self._vad is not None:
            self._remember_history(chunk)
        if self._vad is None:
            self._chunks.append(chunk)
            self._buffered += int(chunk.size)
            limit = int(self._cfg.vad.max_speech_seconds * self._cfg.audio.sample_rate)
            if self._buffered < limit:
                return []
            return [self._cut_buffer()]
        if self._window_rest is not None:
            chunk = np.concatenate([self._window_rest, chunk])
            self._window_rest = None
        whole = chunk.size - (chunk.size % WINDOW_SAMPLES)
        for offset in range(0, whole, WINDOW_SAMPLES):
            self._vad.accept_waveform(chunk[offset : offset + WINDOW_SAMPLES])
        if whole < chunk.size:
            self._window_rest = chunk[whole:]
        return self._drain()

    def flush(self) -> list[Segment]:
        """End of capture: return whatever speech is still pending."""
        import numpy as np

        if self._vad is None:
            # Short remainders are dropped in file mode via the VAD filter;
            # for an explicit stop, keep anything word-length and up.
            if self._buffered < self._cfg.audio.sample_rate // 10:
                self._chunks, self._buffered = [], 0
                return []
            return [self._cut_buffer()]
        if self._window_rest is not None:
            padded = np.pad(
                self._window_rest, (0, WINDOW_SAMPLES - self._window_rest.size)
            )
            self._vad.accept_waveform(padded.astype(np.float32))
            self._window_rest = None
        self._vad.flush()
        return self._drain()

    def _cut_buffer(self) -> Segment:
        import numpy as np

        samples = (
            np.concatenate(self._chunks)
            if len(self._chunks) > 1
            else self._chunks[0]
        )
        self._chunks, self._buffered = [], 0
        return Segment(samples=np.array(samples, dtype=np.float32, copy=True),
                       end_sample=self._fed)

    def _history_capacity_samples(self) -> int:
        cfg = self._cfg
        seconds = (
            cfg.vad.max_speech_seconds + cfg.vad.min_silence_seconds + 5.0
        )
        return int(seconds * cfg.audio.sample_rate) + PRE_ROLL_SAMPLES

    def _remember_history(self, chunk) -> None:
        import numpy as np

        self._history.append(np.array(chunk, dtype=np.float32, copy=True))
        self._history_samples += int(chunk.size)
        # Samples before the last emitted segment can never be used as
        # pre-roll again (that would re-recognize them); drop them, and
        # cap the history so a long silence cannot grow it without bound.
        self._drop_history_before(self._last_processed_end)
        capacity = self._history_capacity_samples()
        if self._history_samples > capacity:
            self._drop_history_before(
                self._history_start + self._history_samples - capacity
            )

    def _drop_history_before(self, index: int) -> None:
        import numpy as np

        drop = min(int(index) - self._history_start, self._history_samples)
        while drop > 0 and self._history:
            first = self._history[0]
            if first.size <= drop:
                drop -= int(first.size)
                self._history_samples -= int(first.size)
                self._history_start += int(first.size)
                self._history.pop(0)
            else:
                self._history[0] = np.array(first[drop:], dtype=np.float32, copy=True)
                self._history_samples -= drop
                self._history_start += drop
                drop = 0

    def _history_slice(self, start: int, end: int):
        import numpy as np

        start = max(int(start), self._history_start)
        end = min(int(end), self._history_start + self._history_samples)
        if end <= start:
            return np.empty(0, dtype=np.float32)
        parts = []
        position = self._history_start
        for chunk in self._history:
            chunk_end = position + chunk.size
            if chunk_end > start and position < end:
                parts.append(
                    chunk[max(0, start - position) : min(chunk.size, end - position)]
                )
            position = chunk_end
        if not parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _drain(self) -> list[Segment]:
        import numpy as np

        segments: list[Segment] = []
        while not self._vad.empty():
            piece = self._vad.front
            # Same use-after-pop hazard as file mode: piece.samples views
            # the native segment that pop() frees — copy (and read start)
            # BEFORE pop(), or the segment intermittently contains garbage
            # samples that decode as "<unk><unk>...".
            piece_samples = np.array(piece.samples, dtype=np.float32, copy=True)
            start_attr = getattr(piece, "start", None)
            self._vad.pop()
            if piece_samples.size == 0:
                continue
            absolute_start = (
                int(start_attr)
                if isinstance(start_attr, (int, float))
                else self._fed - piece_samples.size
            )
            segment_end = absolute_start + piece_samples.size
            # Pre-roll: prepend the few windows just before the detected
            # start, clamped to the previous segment's end so audio is
            # never sent to the recognizer twice.
            expanded_start = max(
                absolute_start - PRE_ROLL_SAMPLES,
                self._last_processed_end,
                self._history_start,
            )
            if expanded_start < absolute_start:
                prefix = self._history_slice(expanded_start, absolute_start)
                if prefix.size:
                    piece_samples = np.concatenate([prefix, piece_samples])
            segments.append(
                Segment(samples=piece_samples, end_sample=segment_end)
            )
            self._last_processed_end = max(self._last_processed_end, segment_end)
        if segments:
            self._drop_history_before(self._last_processed_end)
        return segments
