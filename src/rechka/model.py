"""GigaAM v3 model files: names, sizes and SHA-256 (same set as Android)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    size_bytes: int
    sha256: str


FILES: tuple[ModelFile, ...] = (
    ModelFile(
        "gigaam_v3_e2e_rnnt_encoder_int8.onnx",
        318_995_997,
        "2cac62d0c270bd128f898f2be1a2d34780d524a6e9483888ebac7b00f97410f1",
    ),
    ModelFile(
        "gigaam_v3_e2e_rnnt_decoder.onnx",
        4_600_058,
        "781971998e6a355d6a714f6932a30eab295e7ba0d14fd7e0f78c83b87e811860",
    ),
    ModelFile(
        "gigaam_v3_e2e_rnnt_joint.onnx",
        2_712_896,
        "602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c",
    ),
    ModelFile(
        "gigaam_v3_e2e_rnnt_tokens.txt",
        13_353,
        "7ddf22514c42c531358182c81446a8159771e9921019f09ae743ea622d40221d",
    ),
)

TOTAL_BYTES = sum(f.size_bytes for f in FILES)

# Silero VAD model for segmentation. Not part of GigaAM; distributed with
# sherpa-onnx (asr-models release) and stored next to the GigaAM files.
VAD_FILE = ModelFile(
    "silero_vad.onnx",
    643_854,
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
)
VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"asr-models/{VAD_FILE.name}"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Problem:
    name: str
    detail: str


def verify_model(model_dir: Path, full_hash: bool = False) -> list[Problem]:
    """Return a list of problems with the model in *model_dir* ([] = healthy)."""
    problems: list[Problem] = []
    if not model_dir.is_dir():
        return [Problem(str(model_dir), "model directory does not exist")]
    for expected in FILES:
        candidate = model_dir / expected.name
        if not candidate.is_file():
            problems.append(Problem(expected.name, "missing"))
            continue
        size = candidate.stat().st_size
        if size != expected.size_bytes:
            problems.append(
                Problem(expected.name, f"wrong size: {size} (expected {expected.size_bytes})")
            )
            continue
        if full_hash and sha256_of(candidate) != expected.sha256:
            problems.append(Problem(expected.name, "SHA-256 mismatch"))
    return problems


class ModelNotReadyError(RuntimeError):
    pass


def require_model(model_dir: Path) -> None:
    """Fast (size-only) gate used before loading the recognizer."""
    problems = verify_model(model_dir, full_hash=False)
    if problems:
        summary = "; ".join(f"{p.name}: {p.detail}" for p in problems)
        raise ModelNotReadyError(
            f"GigaAM model in {model_dir} is not usable — {summary}. "
            "Point model.dir in the config at the downloaded GigaAM v3 files."
        )
