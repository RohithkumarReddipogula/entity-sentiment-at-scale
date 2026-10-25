"""M4: span -> token alignment statistics for model C on train and dev (never test).

Usage: uv run python scripts/check_alignment.py [--examples 8]
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from entsent.data import REPO_ROOT

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer

from entsent.config import CONFIG_DIR, load_config
from entsent.data import load_split
from entsent.single_pass import mention_token_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(CONFIG_DIR / "c_final.yaml")
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, revision=cfg.encoder.revision)
    counts: Counter[str] = Counter()
    tokens_per_mention: Counter[int] = Counter()
    extends: list[str] = []
    shared: list[str] = []
    spaced: list[str] = []
    max_len = 0

    for split in ("train", "dev"):
        for ex in load_split(split):
            enc = tokenizer(ex.text, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            max_len = max(max_len, len(enc["input_ids"]))
            counts["headlines"] += 1
            owner: dict[int, str] = {}
            for ent in ex.entities:
                counts["entities"] += 1
                counts["entities_with_2plus_mentions"] += len(ent.spans) > 1
                for a, b in ent.spans:
                    counts["mentions"] += 1
                    idx = mention_token_indices(offsets, (a, b))  # raises if no token
                    tokens_per_mention[len(idx)] += 1
                    lo, hi = offsets[idx[0]][0], offsets[idx[-1]][1]
                    raw = ex.text[a:b]
                    if raw != raw.strip():  # entity name with surrounding whitespace in the data
                        counts["mention_has_surrounding_whitespace"] += 1
                        if len(spaced) < args.examples:
                            spaced.append(f"{ex.id}: {raw!r}")
                        a += len(raw) - len(raw.lstrip())
                        b -= len(raw) - len(raw.rstrip())
                    if (lo, hi) == (a, b):
                        counts["exact_boundary"] += 1
                    else:
                        counts["token_extends_past_mention"] += 1
                        if len(extends) < args.examples:
                            extends.append(
                                f"{ex.id}: {ent.name!r} -> tokens cover {ex.text[lo:hi]!r}"
                            )
                    for i in idx:
                        if i in owner and owner[i] != ent.name:
                            counts["tokens_shared_by_two_entities"] += 1
                            if len(shared) < args.examples:
                                shared.append(
                                    f"{ex.id}: {owner[i]!r} and {ent.name!r} share "
                                    f"{ex.text[slice(*offsets[i])]!r}"
                                )
                        owner[i] = ent.name

    m = counts["mentions"]
    print(f"Headlines (train + dev): {counts['headlines']:,}")
    print(
        f"Entities: {counts['entities']:,}  (with 2+ mentions: "
        f"{counts['entities_with_2plus_mentions']:,})"
    )
    print(f"Mentions: {m:,}")
    print(
        f"  aligned exactly to token boundaries: {counts['exact_boundary']:,} "
        f"({counts['exact_boundary'] / m:.2%})"
    )
    print(
        f"  token extends past the mention:      {counts['token_extends_past_mention']:,} "
        f"({counts['token_extends_past_mention'] / m:.2%})"
    )
    print("  mentions with no token:              0 (would have raised)")
    print(
        f"  (of all mentions, {counts['mention_has_surrounding_whitespace']:,} have leading or "
        "trailing whitespace in the entity name; compared after trimming it)"
    )
    print(f"Tokens shared by two entities: {counts['tokens_shared_by_two_entities']:,}")
    print(f"Tokens per mention: {dict(sorted(tokens_per_mention.items()))}")
    print(f"Longest headline: {max_len} tokens (max_length {cfg.max_length})")
    if extends:
        print("\nExamples, token extends past mention:")
        print("\n".join(f"  {e}" for e in extends))
    if spaced:
        print("\nExamples, entity names with surrounding whitespace:")
        print("\n".join(f"  {s}" for s in spaced))
    if shared:
        print("\nExamples, shared tokens:")
        print("\n".join(f"  {s}" for s in shared))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
