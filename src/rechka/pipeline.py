"""File transcription pipeline: decode -> segment -> recognize -> sinks."""

from __future__ import annotations

import sys
from pathlib import Path

from .audiofile import decode_file
from .config import Config
from .dictionary import Dictionary
from .outputs import emit_finish, emit_segment, sinks_for_output
from .segmentation import segment_waveform
from .transcriber import Transcriber


def transcribe_file(
    path: Path,
    cfg: Config,
    output: str | None = None,
    progress: bool = False,
) -> str:
    """Transcribe one audio/video file; returns the assembled transcript."""
    # Resolve sinks first: an unimplemented IM output must fail before we
    # spend time decoding or loading the model.
    sinks = sinks_for_output(output, source=path, cfg=cfg)
    show_progress = progress
    samples = decode_file(path)
    if samples.size == 0:
        raise ValueError(f"No audio decoded from {path}")

    segments = segment_waveform(samples, cfg)
    dictionary = (
        Dictionary.load(cfg.text.dictionary_path) if cfg.text.dictionary_enabled else None
    )
    transcriber = Transcriber(cfg)

    parts: list[str] = []
    total = max(1, samples.size)
    try:
        for index, segment in enumerate(segments, start=1):
            text = transcriber.transcribe(segment.samples)
            if dictionary is not None:
                text = dictionary.apply(text)
            if text:
                parts.append(text)
                emit_segment(sinks, text)
            if show_progress:
                percent = int(segment.end_sample * 100 / total)
                print(f"\rrechka: {percent}% (segment {index}/{len(segments)})",
                      end="", file=sys.stderr, flush=True)
        if show_progress:
            print(file=sys.stderr)
        full_text = " ".join(parts)
        emit_finish(sinks, full_text)
    finally:
        for sink in sinks:
            sink.close()
    return full_text
