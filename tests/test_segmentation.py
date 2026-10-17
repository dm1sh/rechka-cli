import numpy as np

from rechka.config import Config
from rechka.segmentation import StreamingSegmenter


def _segmenter(tmp_path, **vad_overrides):
    cfg = Config()
    cfg.model.dir = tmp_path  # no silero_vad.onnx here -> fallback path
    for key, value in vad_overrides.items():
        setattr(cfg.vad, key, value)
    return StreamingSegmenter(cfg), cfg


def test_fallback_buffers_until_flush(tmp_path):
    segmenter, cfg = _segmenter(tmp_path)
    assert not segmenter.using_vad
    chunk = np.zeros(512, dtype=np.float32)
    for _ in range(10):  # ~0.32 s: below the max-length cut
        assert segmenter.feed(chunk) == []
    segments = segmenter.flush()
    assert len(segments) == 1
    assert segments[0].samples.size == 10 * 512
    assert segments[0].end_sample == 10 * 512


def test_fallback_drops_tiny_remainder(tmp_path):
    segmenter, cfg = _segmenter(tmp_path)
    segmenter.feed(np.zeros(100, dtype=np.float32))  # 6 ms click
    assert segmenter.flush() == []


def test_fallback_cuts_at_max_speech(tmp_path):
    segmenter, cfg = _segmenter(tmp_path, max_speech_seconds=0.5)
    total = 0
    sizes = []
    chunk = np.zeros(512, dtype=np.float32)
    for _ in range(40):  # 20480 samples = 1.28 s
        for segment in segmenter.feed(chunk):
            sizes.append(segment.samples.size)
            total += segment.samples.size
    for segment in segmenter.flush():
        sizes.append(segment.samples.size)
        total += segment.samples.size
    assert total == 40 * 512
    assert len(sizes) >= 2
    assert all(size <= 8000 + 512 for size in sizes)


class _FakePiece:
    """Mimics a sherpa SpeechSegment: samples view the native buffer that
    pop() frees, so reads after pop() see garbage (the <unk>-spam bug)."""

    def __init__(self, samples, start=0):
        self._samples = samples
        self.start = start
        self.freed = False

    @property
    def samples(self):
        if self.freed:
            return np.full(self._samples.shape, 1e30, dtype=np.float32)
        return self._samples


class _FakeVad:
    def __init__(self, pieces=()):
        self._queue = list(pieces)

    def empty(self):
        return not self._queue

    @property
    def front(self):
        return self._queue[0]

    def pop(self):
        self._queue.pop(0).freed = True

    def accept_waveform(self, samples):
        pass

    def flush(self):
        pass


def test_streaming_drain_copies_samples_before_pop(tmp_path):
    segmenter, cfg = _segmenter(tmp_path)
    expected = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    segmenter._vad = _FakeVad([_FakePiece(expected, start=16)])
    segments = segmenter.flush()
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0].samples, expected)
    assert segments[0].end_sample == 16 + expected.size


def test_file_drain_copies_samples_before_pop(tmp_path, monkeypatch):
    import sys
    import types

    (tmp_path / "silero_vad.onnx").write_bytes(b"fake")
    expected = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)

    class _Silero:
        pass

    class _VadModelConfig:
        def __init__(self):
            self.silero_vad = _Silero()
            self.sample_rate = 16000

    class _Detector(_FakeVad):
        def __init__(self, config, buffer_size_in_seconds=120):
            super().__init__([])

        def flush(self):
            self._queue.append(_FakePiece(expected, start=0))

    fake_sherpa = types.SimpleNamespace(
        VadModelConfig=_VadModelConfig,
        VoiceActivityDetector=_Detector,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

    from rechka.segmentation import segment_waveform

    cfg = Config()
    cfg.model.dir = tmp_path
    segments = segment_waveform(np.zeros(16000, dtype=np.float32), cfg)
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0].samples, expected)


class _FlushVad(_FakeVad):
    """Fake VAD that surfaces the given pieces only on flush()."""

    def __init__(self, pieces):
        super().__init__([])
        self._pending = list(pieces)

    def flush(self):
        self._queue.extend(self._pending)
        self._pending = []


def _feed_waveform(segmenter, waveform):
    for offset in range(0, waveform.size, 512):
        segmenter.feed(waveform[offset : offset + 512])


def test_streaming_preroll_prepends_windows_before_segment(tmp_path):
    from rechka.segmentation import PRE_ROLL_SAMPLES

    segmenter, cfg = _segmenter(tmp_path)
    waveform = np.arange(20000, dtype=np.float32)
    start, length = 8192, 8000
    piece = _FakePiece(waveform[start : start + length].copy(), start=start)
    segmenter._vad = _FlushVad([piece])
    _feed_waveform(segmenter, waveform)
    segments = segmenter.flush()
    assert len(segments) == 1
    np.testing.assert_array_equal(
        segments[0].samples, waveform[start - PRE_ROLL_SAMPLES : start + length]
    )
    assert segments[0].end_sample == start + length


def test_streaming_preroll_never_repeats_previous_segment(tmp_path):
    segmenter, cfg = _segmenter(tmp_path)
    waveform = np.arange(30000, dtype=np.float32)
    first = _FakePiece(waveform[4096:12096].copy(), start=4096)
    # Second segment starts only 500 samples after the first one ended:
    # its pre-roll must be clamped to the first segment's end.
    second = _FakePiece(waveform[12596:20596].copy(), start=12596)
    segmenter._vad = _FlushVad([first, second])
    _feed_waveform(segmenter, waveform)
    segments = segmenter.flush()
    assert len(segments) == 2
    np.testing.assert_array_equal(segments[1].samples, waveform[12096:20596])
    # No sample index is sent to the recognizer twice.
    first_start = 4096
    assert segments[0].end_sample == 12096
    assert segments[1].samples.size == 20596 - 12096


def test_file_preroll_prepends_windows_before_segment(tmp_path, monkeypatch):
    import sys
    import types

    from rechka.segmentation import PRE_ROLL_SAMPLES, segment_waveform

    (tmp_path / "silero_vad.onnx").write_bytes(b"fake")
    waveform = np.arange(24000, dtype=np.float32)
    start, length = 8192, 8000
    piece = _FakePiece(waveform[start : start + length].copy(), start=start)

    class _Silero:
        pass

    class _VadModelConfig:
        def __init__(self):
            self.silero_vad = _Silero()
            self.sample_rate = 16000

    class _Detector(_FlushVad):
        def __init__(self, config, buffer_size_in_seconds=120):
            super().__init__([piece])

    fake_sherpa = types.SimpleNamespace(
        VadModelConfig=_VadModelConfig,
        VoiceActivityDetector=_Detector,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

    cfg = Config()
    cfg.model.dir = tmp_path
    segments = segment_waveform(waveform, cfg)
    assert len(segments) == 1
    np.testing.assert_array_equal(
        segments[0].samples, waveform[start - PRE_ROLL_SAMPLES : start + length]
    )
    assert segments[0].end_sample == start + length
