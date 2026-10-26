"""M9 deployment: the release download is verified, the Dockerfile matches deploy/model.json,
and the server listens on PORT. CI-safe: uses small local archives, no network."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = json.loads((REPO / "deploy" / "model.json").read_text())
DOCKERFILE = (REPO / "Dockerfile").read_text()


def fetch_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_model", REPO / "docker" / "fetch_model.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_archive(tmp: Path, weights: bytes = b"weights") -> tuple[Path, str, str]:
    folder = tmp / "src" / "c_final_s1_x"
    (folder / "tokenizer").mkdir(parents=True)
    (folder / "model.pt").write_bytes(weights)
    (folder / "config.json").write_text("{}")
    archive = tmp / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return (
        archive,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        hashlib.sha256(weights).hexdigest(),
    )


def test_fetch_verifies_and_unpacks(tmp_path: Path) -> None:
    archive, archive_sha, weights_sha = make_archive(tmp_path)
    dest = tmp_path / "model"
    fetch_module().fetch(archive.as_uri(), archive_sha, weights_sha, dest)
    assert (dest / "model.pt").read_bytes() == b"weights"
    assert (dest / "config.json").exists() and (dest / "tokenizer").is_dir()


def test_fetch_rejects_a_wrong_archive_hash(tmp_path: Path) -> None:
    archive, _, weights_sha = make_archive(tmp_path)
    with pytest.raises(SystemExit, match="archive sha256"):
        fetch_module().fetch(archive.as_uri(), "0" * 64, weights_sha, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_fetch_rejects_wrong_weights(tmp_path: Path) -> None:
    archive, archive_sha, _ = make_archive(tmp_path)
    with pytest.raises(SystemExit, match=r"model\.pt sha256"):
        fetch_module().fetch(archive.as_uri(), archive_sha, "0" * 64, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_fetch_refuses_paths_outside_the_target(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(tarfile.TarError):
        fetch_module().fetch(archive.as_uri(), sha, "0" * 64, tmp_path / "model")
    assert not (tmp_path / "escaped.txt").exists()


def build_arg(name: str) -> set[str]:
    return set(re.findall(rf"^ARG {name}=(\S+)$", DOCKERFILE, flags=re.M))


def test_dockerfile_matches_deploy_model_json() -> None:
    assert build_arg("MODEL_URL") == {DEPLOY["url"]}
    assert build_arg("MODEL_ARCHIVE_SHA256") == {DEPLOY["archive_sha256"]}
    assert build_arg("MODEL_PT_SHA256") == {DEPLOY["model_pt_sha256"]}  # both stages agree
    assert DEPLOY["url"].endswith(f"/releases/download/{DEPLOY['release_tag']}/{DEPLOY['archive']}")


def test_deployed_weights_are_the_served_run() -> None:
    record = json.loads((REPO / "results" / f"{DEPLOY['run_id']}.json").read_text())
    assert record["checkpoint"]["model_sha256"] == DEPLOY["model_pt_sha256"]
    test = json.loads(sorted((REPO / "results" / "test").glob("m6_test_*.json"))[-1].read_text())
    served = test["served"]
    assert record["config"]["name"] == "c_final" and record["seed"]["train"] == served["seed"]
    assert served["architecture"] == "C"


def test_server_listens_on_port_and_keeps_the_startup_check() -> None:
    assert "--port ${PORT:-8000}" in DOCKERFILE
    assert 'os.environ.get(\\"PORT\\", \\"8000\\")' in DOCKERFILE  # HEALTHCHECK follows PORT
    assert "MODEL_SHA256=${MODEL_PT_SHA256}" in DOCKERFILE


def test_build_context_whitelist() -> None:
    lines = (REPO / ".dockerignore").read_text().split()
    assert lines[lines.index("*") + 1 :][:3] == [
        "!docker/requirements-serve.txt",
        "!docker/fetch_model.py",
        "!src/",
    ]
