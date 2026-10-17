from pathlib import Path

from rechka.config import Config
from rechka.download import build_items, download_item


def test_build_items_uses_config(tmp_path):
    cfg = Config()
    cfg.model.dir = tmp_path / "model"
    items = build_items(cfg)
    names = [item.spec.name for item in items]
    assert names[0] == "gigaam_v3_e2e_rnnt_encoder_int8.onnx"
    assert names[-1] == "silero_vad.onnx"
    for item in items[:-1]:
        assert item.url.startswith(
            "https://github.com/dm1sh/rechka/releases/download/model-gigaam-v3/"
        )
        assert item.dest.parent == tmp_path / "model"
    # VAD model has its own upstream (sherpa-onnx), stored next to GigaAM.
    assert "k2-fsa/sherpa-onnx" in items[-1].url
    assert items[-1].dest == tmp_path / "model" / "silero_vad.onnx"


def test_build_items_without_vad(tmp_path):
    cfg = Config()
    cfg.model.dir = tmp_path / "model"
    items = build_items(cfg, include_vad=False)
    assert len(items) == 4


def test_download_item_file_url_and_skip(tmp_path):
    from rechka.model import ModelFile, sha256_of

    source = tmp_path / "source.bin"
    source.write_bytes(b"hello rechka" * 100)
    spec = ModelFile(source.name, source.stat().st_size, sha256_of(source))
    dest = tmp_path / "dest" / source.name

    from rechka.download import DownloadItem

    item = DownloadItem(spec=spec, url=source.as_uri(), dest=dest)
    result = download_item(item)
    assert result.status == "downloaded"
    assert dest.read_bytes() == source.read_bytes()

    again = download_item(item)
    assert again.status == "skipped"
