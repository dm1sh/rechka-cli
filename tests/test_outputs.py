from pathlib import Path

import pytest

from rechka.outputs import FileSink, SinkError, StdoutSink, sinks_for_output

SOURCE = Path("/music/My Lecture.mp3")


def resolve(output):
    return sinks_for_output(output, source=SOURCE)


def test_default_is_stdout():
    sinks = resolve(None)
    assert len(sinks) == 1 and isinstance(sinks[0], StdoutSink)


def test_dash_is_stdout():
    assert isinstance(resolve("-")[0], StdoutSink)


def test_im_output_without_engine_or_config_fails_fast():
    # Engine selection is case-insensitive. With no engine running and no
    # config to start one from, resolution must fail fast with SinkError.
    for value in ("ibus", "IBus", "IBUS", "fcitx5", "Fcitx5"):
        with pytest.raises(SinkError):
            resolve(value)


def test_im_sink_starts_engine_and_reports_undeliverable(tmp_path, monkeypatch):
    from rechka.config import Config
    from rechka.im import EngineClient, EngineUnavailableError
    from rechka.outputs import ImSink

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))  # no typing tools anywhere
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    sink = ImSink("fcitx5", Config())
    try:
        # Engine runs, but with no typing tool it cannot insert text, and a
        # one-shot sink must say so instead of queueing silently.
        with pytest.raises(SinkError, match="could not insert"):
            sink.segment("привет")
    finally:
        sink.close()
    with pytest.raises(EngineUnavailableError):
        EngineClient("fcitx5").ping()


def test_existing_folder_gets_source_stem_txt(tmp_path):
    sink = resolve(str(tmp_path))[0]
    assert isinstance(sink, FileSink)
    assert sink._path == tmp_path / "My Lecture.txt"


def test_plain_path_is_output_file(tmp_path):
    target = tmp_path / "result.txt"
    sink = resolve(str(target))[0]
    assert isinstance(sink, FileSink)
    assert sink._path == target


def test_file_sink_writes(tmp_path):
    sink = resolve(str(tmp_path))[0]
    sink.segment("Привет.")
    sink.finish("Привет.")
    assert (tmp_path / "My Lecture.txt").read_text(encoding="utf-8") == "Привет.\n"
