import math
from types import SimpleNamespace

from rechka.benchmark import _recognize_best, ahead_of_time_ratio


class _FakeTranscriber:
    """Returns queued texts, one per segment."""

    def __init__(self, texts):
        self._texts = list(texts)

    def transcribe(self, samples):
        return self._texts.pop(0)


def _segments(count):
    return [SimpleNamespace(samples=None) for _ in range(count)]


def test_ratio_faster_than_realtime():
    # 60 s of audio transcribed in 6 s -> 10x realtime.
    assert ahead_of_time_ratio(60.0, 6.0) == 10.0


def test_ratio_slower_than_realtime():
    assert ahead_of_time_ratio(10.0, 20.0) == 0.5


def test_ratio_zero_elapsed_is_infinite():
    assert math.isinf(ahead_of_time_ratio(10.0, 0.0))


def test_first_output_after_first_nonempty_segment():
    # First segment yields no text; output starts at the second one.
    text, elapsed, first = _recognize_best(
        _segments(3), _FakeTranscriber(["", "привет", "мир"]), None, runs=1
    )
    assert text == "привет мир"
    assert first is not None
    assert 0.0 <= first <= elapsed


def test_first_output_none_when_silent():
    text, _elapsed, first = _recognize_best(
        _segments(2), _FakeTranscriber(["", ""]), None, runs=1
    )
    assert text == ""
    assert first is None
