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


# --- M9: names instead of spans, and the web page ------------------------------------------


def test_predict_text_matches_predict_with_spans(client: TestClient) -> None:
    by_names = client.post("/predict_text", json={"text": TEXT, "names": ["nike", "adidas"]})
    by_spans = client.post("/predict", json=body())
    assert by_names.status_code == 200
    assert by_names.json()["entities"] == by_spans.json()["entities"]


def test_predict_text_finds_every_mention_with_training_rules(client: TestClient) -> None:
    text = "acme shares rise while acme corp reports"
    r = client.post("/predict_text", json={"text": text, "names": ["acme"]})
    assert r.status_code == 200
    assert r.json()["entities"][0]["spans"] == [[0, 4], [23, 27]]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["puma"], "not found in the headline: puma"),
        (["nike", "Puma", "reebok"], "not found in the headline: Puma, reebok"),
        (["  ", " "], "no entity names given"),
        (["nike", ""], "at least 1 character"),
        ([], "at least 1 item"),
    ],
)
def test_predict_text_rejects_bad_names(client: TestClient, names: list[str], message: str) -> None:
    r = client.post("/predict_text", json={"text": TEXT, "names": names})
    assert r.status_code == 422 and message in json.dumps(r.json())


def test_predict_text_ignores_duplicate_names(client: TestClient) -> None:
    r = client.post("/predict_text", json={"text": TEXT, "names": ["nike", " nike ", "adidas"]})
    assert [e["name"] for e in r.json()["entities"]] == ["nike", "adidas"]


def test_page_is_served_with_a_strict_policy(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    html = r.text
    assert '<form id="form">' in html and "/predict_text" in html


def test_page_loads_nothing_from_outside_and_never_injects_html(client: TestClient) -> None:
    html = client.get("/").text
    assert "innerHTML" not in html and "outerHTML" not in html and "insertAdjacentHTML" not in html
    assert "<script src" not in html and "<link" not in html and "@import" not in html
    assert "—" not in html  # no em dashes, same style rule as the docs


# --- the serving image installs base + serve only --------------------------------------------


def test_api_works_without_training_only_packages(tiny_checkpoint: Path) -> None:
    """The Docker image has no pandas, scikit-learn, scipy, sentencepiece, protobuf or dotenv.

    transformers imports scikit-learn and sentencepiece when they are installed, so instead of
    checking imports, this blocks those packages (as in the image) and runs the API.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "BANNED = {'pandas', 'sklearn', 'scipy', 'sentencepiece', 'google', 'dotenv'}\n"
        "for name in BANNED:  # like a missing package: find_spec -> None, import -> error\n"
        "    sys.modules[name] = None\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "from entsent.api import create_app\n"
        f"with TestClient(create_app(Path({str(tiny_checkpoint)!r}))) as c:\n"
        "    assert c.get('/health').status_code == 200\n"
        "    body = {'text': 'nike beats', 'entities': [{'spans': [[0, 4]]}]}\n"
        "    r = c.post('/predict', json=body)\n"
        "    assert r.status_code == 200, r.text\n"
        "print(sorted(m for m in BANNED if sys.modules.get(m) is not None))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


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
