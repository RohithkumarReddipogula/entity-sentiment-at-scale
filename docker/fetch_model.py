"""Download the served checkpoint archive, verify it, unpack it. Used by the Dockerfile.

Usage: python fetch_model.py <url> <archive_sha256> <model_pt_sha256> <dest_dir>

Fails (non-zero exit) if either SHA-256 differs. Standard library only, so it runs in the slim
build stage.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, archive_sha256: str, model_pt_sha256: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        got = sha256_of(archive)
        if got != archive_sha256:
            raise SystemExit(f"archive sha256 {got} != expected {archive_sha256}")

        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")  # no absolute paths, links or devices
        folders = [p for p in unpacked.iterdir() if p.is_dir()]
        if len(folders) != 1:
            raise SystemExit(f"expected one checkpoint folder in the archive, got {folders}")
        got = sha256_of(folders[0] / "model.pt")
        if got != model_pt_sha256:
            raise SystemExit(f"model.pt sha256 {got} != expected {model_pt_sha256}")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folders[0], dest)
    print(f"model verified and unpacked to {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    fetch(sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]))
