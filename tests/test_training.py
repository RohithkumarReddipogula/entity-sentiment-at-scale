"""Training setup tests. CI-safe: a tiny RoBERTa and tokenizer are built in code, so nothing is
downloaded and everything runs on CPU in a few seconds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch import nn
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from entsent.config import CONFIG_DIR, TrainConfig, load_config
from entsent.data import RAW_PATH, Entity, Example, load_split
from entsent.experiment import run_training
from entsent.training import (
    MARKER_CLOSE,
    MARKER_OPEN,
    build_instances,
    mark_entity,
    per_entity_instances,
    select_device,
    train,
)

WORDS = [
    "nike",
    "adidas",
    "beats",
    "forecasts",
    "while",
    "cuts",
    "its",
    "outlook",
    "shares",
    "rise",
    "fall",
    "profit",
    "loss",
    "acme",
    "corp",
    "reports",
    "record",
    "quarter",
    "slump",
    "lawsuit",
    "deal",
    "wins",
]


def tiny_pretrained(cfg: TrainConfig) -> tuple[Any, nn.Module]:
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab |= {w: i + 4 for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    encoder = RobertaModel(
        RobertaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            pad_token_id=0,
        )
    )
    return tokenizer, encoder


def ex(i: int, text: str, labels: dict[str, str]) -> Example:
    ents = []
    for name, label in labels.items():
        start = text.index(name)
        ents.append(Entity(name, label, ((start, start + len(name)),)))  # type: ignore[arg-type]
    return Example(i, text, tuple(ents))


TRAIN = [
    ex(
        1,
        "nike beats forecasts while adidas cuts its outlook",
        {"nike": "positive", "adidas": "negative"},
    ),
    ex(2, "acme shares rise on record quarter", {"acme": "positive"}),
    ex(3, "corp shares fall after lawsuit", {"corp": "negative"}),
    ex(4, "nike reports quarter", {"nike": "neutral"}),
    ex(5, "adidas wins deal", {"adidas": "positive"}),
    ex(6, "acme profit slump", {"acme": "negative"}),
] * 4
DEV = TRAIN[:6]


def tiny_cfg(**overrides: Any) -> TrainConfig:
    base = load_config(CONFIG_DIR / "b_distilroberta.yaml").model_dump()
    base["optim"] |= {"epochs": 2, "batch_size": 4, "lr": 1e-3, "warmup_ratio": 0.0}
    base |= {"device": "cpu"} | overrides
    return TrainConfig.model_validate(base)


# --- configs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.name == path.stem


def test_encoder_check_configs_differ_only_in_encoder() -> None:
    a = load_config(CONFIG_DIR / "b_distilroberta.yaml").model_dump()
    b = load_config(CONFIG_DIR / "b_deberta.yaml").model_dump()
    for d in (a, b):
        d.pop("name"), d.pop("encoder")
    assert a == b


@pytest.mark.parametrize(
    "change",
    [
        {"optimizer": "adam"},  # unknown key
        {"optim": {"lr": 0.5}},  # out of range (and missing fields)
        {"encoder": {"name": "x", "revision": "main"}},  # revision not pinned
        {"architecture": "two_pass"},
        {"device": "tpu"},
    ],
)
def test_bad_configs_are_rejected(change: dict[str, Any], tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIG_DIR / "b_distilroberta.yaml").read_text()) | change
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(path)


# --- devices ----------------------------------------------------------------------------------


def _fake_hardware(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"), [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_auto_device_prefers_cuda_then_mps(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    _fake_hardware(monkeypatch, cuda, mps)
    assert select_device("auto").type == expected


def test_unavailable_device_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_hardware(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError):
        select_device("mps")
    assert select_device("cpu").type == "cpu"


# --- model B inputs -----------------------------------------------------------------------


def test_mark_entity() -> None:
    text = "Nike beats forecasts while Adidas cuts its outlook."
    assert mark_entity(text, (27, 33)) == (
        "Nike beats forecasts while [E] Adidas [/E] cuts its outlook."
    )


def test_one_instance_per_entity_marking_first_mention() -> None:
    example = Example(
        9,
        "Infosys up; analysts back Infosys",
        (Entity("Infosys", "positive", ((0, 7), (26, 33))),),
    )
    [inst] = per_entity_instances([example])
    assert inst.text == "[E] Infosys [/E] up; analysts back Infosys"
    assert (inst.headline_id, inst.entity_index, inst.label) == (9, 0, 0)


def test_single_pass_not_built_yet() -> None:
    with pytest.raises(NotImplementedError):
        build_instances(tiny_cfg(architecture="single_pass"), TRAIN)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run `make data` first")
def test_markers_wrap_exact_mention_on_real_data() -> None:
    for example in load_split("train")[:2000]:
        for inst, ent in zip(per_entity_instances([example]), example.entities, strict=True):
            inner = inst.text.split(MARKER_OPEN + " ", 1)[1].split(" " + MARKER_CLOSE, 1)[0]
            assert inner == ent.name


# --- training -------------------------------------------------------------------------------


def run_tiny(cfg: TrainConfig) -> Any:
    return train(
        cfg, TRAIN, DEV, device=torch.device("cpu"), pretrained=tiny_pretrained, log=lambda _: None
    )


def test_markers_are_single_tokens() -> None:
    from entsent.training import prepare

    tokenizer, encoder = tiny_pretrained(tiny_cfg())
    prepare(tokenizer, encoder, 0.1)
    ids = tokenizer("[E] nike [/E] beats")["input_ids"]
    assert tokenizer.convert_ids_to_tokens(ids)[1:4] == ["[E]", "nike", "[/E]"]


HF_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "hf" / "hub"


@pytest.mark.skipif(not HF_CACHE.exists(), reason="encoders not downloaded (local cache only)")
@pytest.mark.parametrize("config", ["b_distilroberta.yaml", "b_deberta.yaml"])
def test_real_tokenizers_keep_markers_whole_and_cls_first(config: str) -> None:
    from transformers import AutoTokenizer

    cfg = load_config(CONFIG_DIR / config)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder.name,
        revision=cfg.encoder.revision,
        cache_dir=HF_CACHE,
        local_files_only=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [MARKER_OPEN, MARKER_CLOSE]})
    text = mark_entity("Nike beats forecasts while Adidas cuts its outlook.", (27, 33))
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
    assert tokens[0] == tokenizer.cls_token
    assert tokens.count(MARKER_OPEN) == 1 and tokens.count(MARKER_CLOSE) == 1
    inner = tokens[tokens.index(MARKER_OPEN) + 1 : tokens.index(MARKER_CLOSE)]
    assert tokenizer.convert_tokens_to_string(inner).strip() == "Adidas"


def test_half_precision_encoder_is_rejected() -> None:
    # deberta-v3-small's checkpoint is float16; mixed with the float32 head it aborted MPS.
    from entsent.training import prepare

    tokenizer, encoder = tiny_pretrained(tiny_cfg())
    with pytest.raises(TypeError, match="float32"):
        prepare(tokenizer, encoder.half(), 0.1)


@pytest.mark.skipif(not HF_CACHE.exists(), reason="encoders not downloaded (local cache only)")
def test_float16_checkpoint_loads_as_float32() -> None:
    # Offline, project cache only: see tests/conftest.py.
    from entsent.training import load_pretrained, prepare

    tokenizer, encoder = load_pretrained(load_config(CONFIG_DIR / "b_deberta.yaml"))
    model = prepare(tokenizer, encoder, 0.1)
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_same_seed_same_result_on_cpu() -> None:
    a, b = run_tiny(tiny_cfg(seed=7)), run_tiny(tiny_cfg(seed=7))
    assert [e.train_loss for e in a.epochs] == [e.train_loss for e in b.epochs]
    assert a.dev_pred == b.dev_pred


def test_different_seed_different_result() -> None:
    a, b = run_tiny(tiny_cfg(seed=7)), run_tiny(tiny_cfg(seed=8))
    assert [e.train_loss for e in a.epochs] != [e.train_loss for e in b.epochs]


def test_training_lowers_loss() -> None:
    out = run_tiny(tiny_cfg(optim={**tiny_cfg().optim.model_dump(), "epochs": 8}))
    assert out.epochs[-1].train_loss < out.epochs[0].train_loss
    assert out.best_epoch == max(range(1, 9), key=lambda e: out.epochs[e - 1].dev_macro_f1)


def test_run_writes_record_predictions_and_checkpoint(tmp_path: Path) -> None:
    cfg = tiny_cfg(save_checkpoint=True)
    path = run_training(
        cfg,
        TRAIN,
        DEV,
        device=torch.device("cpu"),
        purpose="test",
        config_path="tiny",
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        pretrained=tiny_pretrained,
        n_bootstrap=100,
        log=lambda _: None,
    )
    record = json.loads(path.read_text())
    assert record["run_id"].startswith(f"b_distilroberta_s{cfg.seed}_")  # no doubled "b_b_"
    assert record["model_id"] == "B" and record["split"] == "dev"
    assert record["seed"]["train"] == cfg.seed and record["config"] == cfg.model_dump()
    assert "commit" in record["git"] and record["environment"]["device"] == "cpu"
    assert len(record["training"]["epochs"]) == 2
    assert len(record["metrics"]["all"]["macro_f1_ci95"]) == 2
    preds = (tmp_path / "results" / "predictions" / f"{record['run_id']}.jsonl").read_text()
    assert len(preds.splitlines()) == record["n_entities"] == 7
    assert (tmp_path / record["checkpoint"] / "model.pt").exists()
