"""Model C: character span -> token alignment and mention pooling. This is where bugs hide.

CI-safe tests use a tiny word-level tokenizer and encoder built in code. Tests marked
`real_tokenizer` use the pinned distilroberta tokenizer from the local cache only, and the
property tests also need the raw data; both are skipped when those are absent (as in CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from test_training import DEV, TRAIN, run_tiny, tiny_cfg, tiny_pretrained

from entsent.config import CONFIG_DIR, load_config
from entsent.data import LABELS, RAW_PATH, Entity, Example, load_split
from entsent.single_pass import (
    MentionPoolingClassifier,
    collate_single_pass,
    entity_weights,
    mention_token_indices,
    single_pass_instances,
)

REPO = Path(__file__).resolve().parents[1]
HF_CACHE = REPO / ".cache" / "hf" / "hub"
needs_cache = pytest.mark.skipif(not HF_CACHE.exists(), reason="encoder not in local cache")
needs_data = pytest.mark.skipif(not RAW_PATH.exists(), reason="run `make data` first")


def example(text: str, entities: dict[str, str], *, all_mentions: bool = False) -> Example:
    """Build an Example; spans are found by str.find (first mention, or every mention)."""
    ents = []
    for name, label in entities.items():
        starts = [text.index(name)]
        if all_mentions:
            starts = [i for i in range(len(text)) if text.startswith(name, i)]
        spans = tuple((s, s + len(name)) for s in starts)
        ents.append(Entity(name, label, spans))  # type: ignore[arg-type]
    return Example(1, text, tuple(ents))


# --- the overlap rule, on hand-written offsets -------------------------------------------


def test_overlap_rule_selects_exactly_overlapping_tokens() -> None:
    # tokens: <s> "Nike"(0,4) "beats"(5,10) "Adidas"(11,17) </s> <pad>
    offsets = [(0, 0), (0, 4), (5, 10), (11, 17), (0, 0), (0, 0)]
    assert mention_token_indices(offsets, (0, 4)) == [1]
    assert mention_token_indices(offsets, (11, 17)) == [3]
    assert mention_token_indices(offsets, (5, 17)) == [2, 3]


def test_partial_token_overlap_is_included() -> None:
    offsets = [(0, 0), (0, 5), (6, 13), (0, 0)]  # "Sebis"(0,5) "reading"(6,13)
    assert mention_token_indices(offsets, (0, 4)) == [1]  # "Sebi" inside "Sebis"


def test_special_and_padding_tokens_never_selected() -> None:
    offsets = [(0, 0), (0, 4), (0, 0), (0, 0)]
    assert mention_token_indices(offsets, (0, 4)) == [1]
    with pytest.raises(ValueError, match="covers no token"):
        mention_token_indices(offsets, (10, 12))


def test_weights_mean_per_mention_then_across_mentions() -> None:
    # entity with a 1-token mention and a 3-token mention: each mention gets weight 1/2
    offsets = [(0, 0), (0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (0, 0)]
    w = entity_weights(offsets, [(0, 3), (7, 15)])
    assert w == pytest.approx([0, 1 / 2, 0, 1 / 6, 1 / 6, 1 / 6, 0])
    assert sum(w) == pytest.approx(1.0)


# --- collate + pooling with the tiny model (CI) -----------------------------------------


def tiny_model() -> tuple[Any, MentionPoolingClassifier]:
    tokenizer, encoder = tiny_pretrained(tiny_cfg())
    return tokenizer, MentionPoolingClassifier(encoder, 16, 0.0).eval()


def test_collate_weights_point_at_the_right_tokens() -> None:
    tokenizer, _ = tiny_model()
    ex = example(
        "nike beats forecasts while adidas cuts", {"nike": "positive", "adidas": "negative"}
    )
    batch = collate_single_pass(single_pass_instances([ex]), tokenizer, 128)
    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    for row, name in enumerate(["nike", "adidas"]):
        selected = [t for t, w in zip(tokens, batch["pool_weights"][row], strict=True) if w > 0]
        assert selected == [name]
    assert batch["pool_index"].tolist() == [0, 0]
    assert batch["labels"].tolist() == [LABELS.index("positive"), LABELS.index("negative")]


def test_pooled_vector_equals_hand_computed_mean() -> None:
    tokenizer, model = tiny_model()
    ex = example(
        "acme shares rise while acme corp reports", {"acme": "positive"}, all_mentions=True
    )
    batch = collate_single_pass(single_pass_instances([ex]), tokenizer, 128)
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        pooled = model.pool(**inputs)
        hidden = model.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state[0]
    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    acme = [i for i, t in enumerate(tokens) if t == "acme"]
    assert len(acme) == 2
    expected = (hidden[acme[0]] + hidden[acme[1]]) / 2
    torch.testing.assert_close(pooled[0], expected)


def test_logits_do_not_depend_on_padding_or_batch() -> None:
    tokenizer, model = tiny_model()
    instances = single_pass_instances(DEV)
    with torch.no_grad():
        together = model(**_inputs(collate_single_pass(instances, tokenizer, 128)))
        alone = torch.cat(
            [model(**_inputs(collate_single_pass([inst], tokenizer, 128))) for inst in instances]
        )
    torch.testing.assert_close(together, alone, rtol=1e-5, atol=1e-5)


def _inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k != "labels"}


def test_too_long_headline_fails_loudly() -> None:
    tokenizer, _ = tiny_model()
    ex = example("nike beats " * 40, {"nike": "positive"})
    with pytest.raises(ValueError, match="max_length"):
        collate_single_pass(single_pass_instances([ex]), tokenizer, 32)


def test_mention_without_tokens_fails_loudly() -> None:
    tokenizer, _ = tiny_model()
    bad = Example(1, "nike beats", (Entity("ghost", "neutral", ((20, 25),)),))
    with pytest.raises(ValueError, match="covers no token"):
        collate_single_pass(single_pass_instances([bad]), tokenizer, 128)


def test_model_c_trains_deterministically_and_round_trips(tmp_path: Path) -> None:
    from entsent.experiment import load_checkpoint, run_training
    from entsent.training import make_loader_for, predict

    cfg = tiny_cfg(architecture="single_pass", save_checkpoint=True)
    a, b = run_tiny(cfg), run_tiny(cfg)
    assert [e.train_loss for e in a.epochs] == [e.train_loss for e in b.epochs]

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
    assert record["model_id"] == "C" and record["training"]["n_train_instances"] == len(TRAIN)
    model, tokenizer, loaded = load_checkpoint(tmp_path / record["checkpoint"]["path"])
    assert isinstance(model, MentionPoolingClassifier) and loaded == cfg
    loader = make_loader_for(cfg, single_pass_instances(DEV), tokenizer, shuffle=False)
    preds = [LABELS[p] for p in predict(model, loader, torch.device("cpu"))]
    lines = (tmp_path / "results" / "predictions" / f"{record['run_id']}.jsonl").read_text()
    assert preds == [json.loads(line)["pred"] for line in lines.splitlines()]


# --- the real distilroberta tokenizer (local cache only) --------------------------------


@pytest.fixture(scope="module")
def real_tokenizer() -> Any:
    from transformers import AutoTokenizer

    cfg = load_config(CONFIG_DIR / "c_final.yaml")
    return AutoTokenizer.from_pretrained(
        cfg.encoder.name, revision=cfg.encoder.revision, cache_dir=HF_CACHE, local_files_only=True
    )


def selected_text(tokenizer: Any, text: str, span: tuple[int, int]) -> str:
    enc = tokenizer(text, return_offsets_mapping=True)
    idx = mention_token_indices(enc["offset_mapping"], span)
    assert idx == list(range(idx[0], idx[-1] + 1)), "a mention's tokens must be contiguous"
    decoded: str = tokenizer.decode([enc["input_ids"][i] for i in idx])
    return decoded.strip()


EXACT_CASES = [
    ("Nike beats forecasts while Adidas cuts its outlook.", "Nike"),  # start, no leading space
    ("Nike beats forecasts while Adidas cuts its outlook.", "Adidas"),
    ("Markets slip, investors eye Infosys", "Infosys"),  # at the end
    ("Nike, Adidas shares fall", "Nike"),  # followed by punctuation
    ("Moody's cuts Zenith Steel UK's rating after weak results", "Zenith Steel UK"),  # multi-word
    ("Moody's cuts Zenith Steel UK's rating after weak results", "Moody's"),  # apostrophe
    ("Orion Labs and Dr. Reddy's lead the drug stocks", "Dr. Reddy's"),  # abbreviation
    ("Rupee at ₹83.2 as RBI steps in", "RBI"),  # non-ASCII before the mention
    ("\u2018Nifty\u2019 ends higher on IT gains", "Nifty"),  # curly quotes around the mention
    ("Auto, Pharma slip; Orion Motors, Sun Pharma lose ground", "Sun Pharma"),  # nested names
]
# All test headlines are made up (checked against SEntFiN); no dataset text is in the repo.


@needs_cache
@pytest.mark.parametrize(("text", "mention"), EXACT_CASES)
def test_real_tokenizer_exact_mentions(real_tokenizer: Any, text: str, mention: str) -> None:
    start = text.rindex(mention) if mention == "Sun Pharma" else text.index(mention)
    assert selected_text(real_tokenizer, text, (start, start + len(mention))) == mention


@needs_cache
@pytest.mark.parametrize(
    ("text", "mention"),
    [
        ("Acmes reading of the new rules worries analysts", "Acme"),
        ("Nifty50 hits a record as banks gain", "Nifty"),
    ],
)
def test_real_tokenizer_suffix_mentions(real_tokenizer: Any, text: str, mention: str) -> None:
    got = selected_text(real_tokenizer, text, (0, len(mention)))
    assert got.startswith(mention) and len(got) <= len(text.split()[0])  # mention + its suffix only


@needs_cache
def test_real_tokenizer_nested_names_do_not_share_tokens(real_tokenizer: Any) -> None:
    text = "Auto, Pharma slip; Orion Motors, Sun Pharma lose ground"
    enc = real_tokenizer(text, return_offsets_mapping=True)
    pharma = mention_token_indices(enc["offset_mapping"], (6, 12))
    sun_pharma = mention_token_indices(enc["offset_mapping"], (33, 43))
    assert not set(pharma) & set(sun_pharma)


# --- properties over every train and dev mention (raw data + cache) -------------------------


@needs_cache
@needs_data
def test_alignment_properties_on_train_and_dev(real_tokenizer: Any) -> None:
    """Every mention: >= 1 token, contiguous, full coverage of its non-space characters,
    no selected token entirely outside it, and no headline over max_length."""
    max_length = load_config(CONFIG_DIR / "c_final.yaml").max_length
    for ex in load_split("train") + load_split("dev"):
        enc = real_tokenizer(ex.text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        assert len(enc["input_ids"]) <= max_length, ex.id
        for ent in ex.entities:
            for a, b in ent.spans:
                idx = mention_token_indices(offsets, (a, b))
                assert idx == list(range(idx[0], idx[-1] + 1)), (ex.id, ent.name)
                covered = {c for i in idx for c in range(*offsets[i])}
                needed = {c for c in range(a, b) if not ex.text[c].isspace()}
                assert needed <= covered, (ex.id, ent.name)
                assert all(offsets[i][0] < b and offsets[i][1] > a for i in idx)


# --- the refactor must not change model B ---------------------------------------------------


B_CKPT = REPO / "checkpoints" / "b_final_s1_20261025T145523Z"


@pytest.mark.skipif(
    not (B_CKPT / "model.pt").exists() or not RAW_PATH.exists(),
    reason="needs the local b_final seed-1 checkpoint and raw data",
)
def test_saved_model_b_still_reproduces_its_dev_predictions() -> None:
    from entsent.experiment import load_checkpoint
    from entsent.training import make_loader_for, per_entity_instances, predict

    model, tokenizer, cfg = load_checkpoint(B_CKPT)
    loader = make_loader_for(cfg, per_entity_instances(load_split("dev")), tokenizer, shuffle=False)
    preds = [LABELS[p] for p in predict(model, loader, torch.device("cpu"))]
    lines = (REPO / "results" / "predictions" / f"{B_CKPT.name}.jsonl").read_text().splitlines()
    assert preds == [json.loads(line)["pred"] for line in lines]
