import wave

import numpy as np

from rechka.audiofile import decode_file
from rechka.config import Config
from rechka.segmentation import segment_waveform


def make_wav(path, seconds=1.0, rate=16000):
    frames = int(seconds * rate)
    tone = (np.sin(2 * np.pi * 220 * np.arange(frames) / rate) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(tone.tobytes())


def test_decode_wav_16k(tmp_path):
    path = tmp_path / "tone.wav"
    make_wav(path, seconds=1.0)
    samples = decode_file(path)
    assert samples.dtype == np.float32
    assert abs(samples.size - 16000) < 100


def test_fixed_segmentation(tmp_path):
    cfg = Config()
    cfg.vad.enabled = False
    samples = np.zeros(16000 * 65, dtype=np.float32)
    segments = segment_waveform(samples, cfg)
    assert [s.samples.size for s in segments] == [480000, 480000, 80000]
    assert segments[-1].end_sample == samples.size
