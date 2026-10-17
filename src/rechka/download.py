"""`rechka download` — fetch the model files from the GitHub release.

Storage follows the config: GigaAM files go to [model].dir, the Silero
VAD model to [vad].model (default: next to the GigaAM files). Files that
are already present and valid are skipped; partial downloads land in
"<name>.part" and are only moved into place after size + SHA-256 pass.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .model import FILES, VAD_FILE, VAD_URL, ModelFile, sha256_of

_CHUNK = 1024 * 1024
_USER_AGENT = "rechka-cli (https://github.com/dm1sh/rechka)"


@dataclass
class DownloadItem:
    spec: ModelFile
    url: str
    dest: Path


@dataclass
class ItemResult:
    name: str
    status: str  # downloaded | skipped | failed
    detail: str = ""


def release_base(repository: str, tag: str) -> str:
    return f"https://github.com/{repository}/releases/download/{tag}"


def build_items(
    cfg: Config,
    repository: str | None = None,
    tag: str | None = None,
    include_vad: bool | None = None,
) -> list[DownloadItem]:
    repo = repository or cfg.download.repository
    rel_tag = tag or cfg.download.tag
    base = release_base(repo, rel_tag)

    items = [
        DownloadItem(spec=spec, url=f"{base}/{spec.name}", dest=cfg.model.dir / spec.name)
        for spec in FILES
    ]
    want_vad = cfg.download.include_vad if include_vad is None else include_vad
    if want_vad:
        vad_dest = cfg.vad_model_path() or (cfg.model.dir / VAD_FILE.name)
        items.append(DownloadItem(spec=VAD_FILE, url=VAD_URL, dest=Path(vad_dest)))
    return items


def _is_valid(path: Path, spec: ModelFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec.size_bytes
        and sha256_of(path) == spec.sha256
    )


def download_item(item: DownloadItem, force: bool = False) -> ItemResult:
    spec, dest = item.spec, item.dest
    if not force and _is_valid(dest, spec):
        return ItemResult(spec.name, "skipped", "already present and verified")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(item.url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else spec.size_bytes
            done = 0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r{spec.name}: {done * 100 // total}% "
                        f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
            print(file=sys.stderr)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        return ItemResult(spec.name, "failed", f"download failed: {exc}")

    got_size = partial.stat().st_size
    if got_size != spec.size_bytes:
        partial.unlink(missing_ok=True)
        return ItemResult(
            spec.name,
            "failed",
            f"size mismatch after download: {got_size} (expected {spec.size_bytes})",
        )
    if sha256_of(partial) != spec.sha256:
        partial.unlink(missing_ok=True)
        return ItemResult(spec.name, "failed", "SHA-256 mismatch after download")
    os.replace(partial, dest)
    return ItemResult(spec.name, "downloaded", f"{spec.size_bytes / 1e6:.0f} MB verified")


def download_all(
    cfg: Config,
    repository: str | None = None,
    tag: str | None = None,
    include_vad: bool | None = None,
    force: bool = False,
) -> list[ItemResult]:
    items = build_items(cfg, repository=repository, tag=tag, include_vad=include_vad)
    total_mb = sum(i.spec.size_bytes for i in items) / 1e6
    print(
        f"rechka: downloading {len(items)} files ({total_mb:.0f} MB) "
        f"from {repository or cfg.download.repository}@{tag or cfg.download.tag} "
        f"into {cfg.model.dir}",
        file=sys.stderr,
    )
    return [download_item(item, force=force) for item in items]
