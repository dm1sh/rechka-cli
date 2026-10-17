from pathlib import Path

import pytest

from rechka.config import ConfigError, load_config


def test_defaults_without_file(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.model.threads == 4
    assert cfg.audio.sample_rate == 16000
    assert cfg.im.framework == "auto"


def test_overrides(tmp_path):
    path = tmp_path / "rechka.toml"
    path.write_text(
        '[model]\ndir = "/tmp/model"\nthreads = 2\n[im]\nframework = "ibus"\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.model.dir == Path("/tmp/model")
    assert cfg.model.threads == 2
    assert cfg.im.framework == "ibus"


def test_unknown_key_warned_and_ignored(tmp_path, capsys):
    path = tmp_path / "rechka.toml"
    path.write_text("[model]\nbogus = 1\nthreads = 2\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.threads == 2
    assert "ignoring unknown key 'model.bogus'" in capsys.readouterr().err


def test_unknown_section_warned_and_ignored(tmp_path, capsys):
    path = tmp_path / "rechka.toml"
    path.write_text(
        '[output]\nsinks = ["stdout"]\n[model]\nthreads = 3\n', encoding="utf-8"
    )
    cfg = load_config(path)
    assert cfg.model.threads == 3
    assert "ignoring unknown section '[output]'" in capsys.readouterr().err


def test_bad_sample_rate_rejected(tmp_path):
    path = tmp_path / "rechka.toml"
    path.write_text("[audio]\nsample_rate = 8000\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
