"""M7 API tests. CI-safe: the tiny model-C encoder is saved as a real checkpoint in a temp folder,
so the full load path runs without downloads. One local-only test checks the real C seed 1."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from test_training import DEV, TRAIN, tiny_cfg, tiny_pretrained

from entsent.api import MAX_ENTITIES, MAX_TEXT_CHARS, create_app
from entsent.data import LABELS, RAW_PATH, load_split, sha256_file
from entsent.experiment import run_training

REPO = Path(__file__).resolve().parents[1]
TEXT = "nike beats forecasts while adidas cuts its outlook"


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api")
    cfg = tiny_cfg(architecture="single_pass", save_checkpoint=True)
    path = run_training(
        cfg,
        TRAIN,
        DEV,
        device=torch.device("cpu"),
        purpose="test",
        config_path="tiny",
        results_dir=root / "results",
        checkpoint_dir=root / "checkpoints",
        pretrained=tiny_pretrained,
        n_bootstrap=50,
        log=lambda _: None,
    )
    checkpoint: str = json.loads(path.read_text())["checkpoint"]["path"]
    return root / checkpoint


@pytest.fixture(scope="module")
def client(tiny_checkpoint: Path) -> Iterator[TestClient]:
    app = create_app(tiny_checkpoint, sha256_file(tiny_checkpoint / "model.pt"))
    with TestClient(app) as c:
        yield c


def body(**overrides: Any) -> dict[str, Any]:
    return {
        "text": TEXT,
        "entities": [{"name": "nike", "spans": [[0, 4]]}, {"name": "adidas", "spans": [[27, 33]]}],
    } | overrides


# --- health ---------------------------------------------------------------------------------


def test_health_reports_the_loaded_model(client: TestClient, tiny_checkpoint: Path) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    cfg = json.loads((tiny_checkpoint / "config.json").read_text())
    assert r.json() == {
        "status": "ok",
        "model": f"{cfg['name']}_s{cfg['seed']}",  # from the checkpoint, not its folder name
        "sha256": sha256_file(tiny_checkpoint / "model.pt"),
    }


def test_health_is_503_when_model_dir_is_missing(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "nowhere")) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=body()).status_code == 503


def test_health_is_503_on_sha256_mismatch(tiny_checkpoint: Path) -> None:
    with TestClient(create_app(tiny_checkpoint, "0" * 64)) as c:
        r = c.get("/health")
        assert r.status_code == 503 and "does not match" in r.json()["detail"]


# --- predictions ------------------------------------------------------------------------------


def test_predict_returns_one_answer_per_entity(client: TestClient) -> None:
    r = client.post("/predict", json=body())
    assert r.status_code == 200
    out = r.json()
    assert out["model"]["architecture"] == "single_pass"
    assert [e["name"] for e in out["entities"]] == ["nike", "adidas"]
    for e in out["entities"]:
        probs = e["probabilities"]
        assert set(probs) == set(LABELS)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
        assert e["label"] == max(probs, key=probs.__getitem__)
        assert e["confidence"] == pytest.approx(probs[e["label"]])


def test_predict_matches_the_model_used_in_evaluation(
    client: TestClient, tiny_checkpoint: Path
) -> None:
    from entsent.experiment import load_checkpoint
    from entsent.single_pass import single_pass_instances
    from entsent.training import make_loader_for, predict

    model, tokenizer, cfg = load_checkpoint(tiny_checkpoint)
    loader = make_loader_for(cfg, single_pass_instances(DEV), tokenizer, shuffle=False)
    expected = [LABELS[p] for p in predict(model, loader, torch.device("cpu"))]
    got = []
    for ex in DEV:
        ents = [{"spans": [list(s) for s in e.spans]} for e in ex.entities]
        out = client.post("/predict", json={"text": ex.text, "entities": ents}).json()
        got += [e["label"] for e in out["entities"]]
    assert got == expected


def test_multiple_mentions_and_optional_names(client: TestClient) -> None:
    text = "acme shares rise while acme corp reports"
    r = client.post("/predict", json={"text": text, "entities": [{"spans": [[0, 4], [23, 27]]}]})
    assert r.status_code == 200
    assert r.json()["entities"][0]["name"] is None


def test_same_request_same_answer(client: TestClient) -> None:
    assert (
        client.post("/predict", json=body()).json() == client.post("/predict", json=body()).json()
    )


# --- input validation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"text": ""}, "at least 1 character"),
        ({"text": "   "}, "text is empty"),
        ({"text": "x" * (MAX_TEXT_CHARS + 1)}, "at most"),
        ({"entities": []}, "at least 1 item"),
        ({"entities": [{"spans": [[0, 4]]}] * (MAX_ENTITIES + 1)}, "at most"),
        ({"entities": [{"spans": [[0, 99]]}]}, "outside the text"),
        ({"entities": [{"spans": [[4, 4]]}]}, "outside the text"),
        ({"entities": [{"spans": [[4, 5]]}]}, "only whitespace"),
        ({"entities": [{"spans": [[0, 4]]}, {"spans": [[2, 8]]}]}, "overlap"),
        ({"entities": [{"spans": [[0, 10]]}, {"spans": [[0, 4]]}, {"spans": [[5, 9]]}]}, "overlap"),
        ({"entities": [{"spans": [[0]]}]}, "at least 2 items"),
        ({"entities": [{"spans": [[0, 4]], "label": "positive"}]}, "Extra inputs"),
    ],
)
def test_bad_requests_are_rejected_with_a_reason(
    client: TestClient, change: dict[str, Any], message: str
) -> None:
    r = client.post("/predict", json=body(**change))
    assert r.status_code == 422
    assert message in json.dumps(r.json())


def test_too_many_tokens_is_422_not_500(client: TestClient) -> None:
    text = "nike beats " * 80
    r = client.post("/predict", json={"text": text, "entities": [{"spans": [[0, 4]]}]})
    assert r.status_code == 422 and "max_length" in r.json()["detail"]


# --- the real served model (local only) ------------------------------------------------------


C1 = REPO / "checkpoints" / "c_final_s1_20261025T160107Z"


@pytest.mark.skipif(
    not (C1 / "model.pt").exists() or not RAW_PATH.exists(),
    reason="needs the local c_final seed-1 checkpoint and raw data",
)
def test_real_api_reproduces_c_seed1_dev_predictions() -> None:
    record = json.loads((REPO / "results" / f"{C1.name}.json").read_text())
    lines = (REPO / "results" / "predictions" / f"{C1.name}.jsonl").read_text().splitlines()
    recorded: dict[int, list[str]] = {}
    for line in lines:
        row = json.loads(line)
        recorded.setdefault(row["id"], []).append(row["pred"])
    with TestClient(create_app(C1, record["checkpoint"]["model_sha256"])) as c:
        for ex in load_split("dev")[:200]:
            ents = [{"spans": [list(s) for s in e.spans]} for e in ex.entities]
            out = c.post("/predict", json={"text": ex.text, "entities": ents}).json()
            assert [e["label"] for e in out["entities"]] == recorded[ex.id], ex.id
