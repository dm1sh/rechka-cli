from rechka import systemd


def test_default_exec_start_runs_serve(monkeypatch):
    monkeypatch.setattr(systemd.shutil, "which", lambda name: "/usr/bin/rechka")
    assert systemd.default_exec_start() == "/usr/bin/rechka serve"


def test_default_exec_start_falls_back_to_user_bin(monkeypatch):
    monkeypatch.setattr(systemd.shutil, "which", lambda name: None)
    monkeypatch.setattr(systemd.Path, "home", classmethod(lambda cls: systemd.Path("/home/test")))
    assert systemd.default_exec_start() == "/home/test/.local/bin/rechka serve"


def test_unit_text_uses_given_exec_start_verbatim():
    text = systemd.unit_text("/opt/rechka serve --config /tmp/r.toml")
    assert "ExecStart=/opt/rechka serve --config /tmp/r.toml" in text
    assert "[Install]" in text and "WantedBy=graphical-session.target" in text
    assert "After=graphical-session.target" in text


def test_unit_text_appends_serve_when_missing():
    text = systemd.unit_text("/opt/rechka")
    assert "ExecStart=/opt/rechka serve" in text
