"""M0: download SEntFiN 1.0, build the fixed train/dev/test split, regenerate DATASET.md stats.

Usage: uv run python scripts/make_data.py [--check]

--check rebuilds the split in memory and fails if it differs from the committed files,
instead of writing anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence

from entsent.data import (
    LABELS,
    REPO_ROOT,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    SOURCE_URL,
    SPLITS,
    SPLITS_DIR,
    CleaningEvent,
    download,
    load_examples,
    sha256_file,
)
from entsent.splits import FRACTIONS, SEED, SplitStats, make_split, split_stats

DATASET_MD = REPO_ROOT / "DATASET.md"


def render_stats(stats: dict[str, SplitStats]) -> str:
    def pct(n: int, d: int) -> str:
        return f"{n:,} ({n / d:.1%})"

    def row(title: str, cells: list[str]) -> str:
        return f"| {title} | " + " | ".join(cells) + " |\n"

    cols = list(stats.values())
    out = "| | " + " | ".join(stats) + " |\n|---|" + "---:|" * len(stats) + "\n"
    out += row("Headlines", [f"{s['headlines']:,}" for s in cols])
    out += row("Entities", [f"{s['entities']:,}" for s in cols])
    for lab in LABELS:
        cells = [pct(s["labels"][lab], s["entities"]) for s in cols]
        out += row(f"{lab.capitalize()} entities", cells)
    out += row(
        "Headlines with 2+ entities",
        [pct(s["multi_entity_headlines"], s["headlines"]) for s in cols],
    )
    out += row(
        "Mixed-label headlines", [pct(s["mixed_label_headlines"], s["headlines"]) for s in cols]
    )
    return out


def render_events(events: Sequence[CleaningEvent]) -> str:
    counts = Counter(e.action for e in events)
    out = (
        f"Repaired: {counts['repaired']}, dropped: {counts['dropped']}, "
        f"parsed with regex fallback: {counts['parsed_with_fallback']}.\n\n"
        "| ID | Action | Original | Result | Headline |\n|---:|---|---|---|---|\n"
    )
    for e in sorted(events, key=lambda e: (e.action, e.id)):
        cells = [str(e.id), e.action, e.original, e.result, e.text]
        out += "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |\n"
    return out


def replace_section(doc: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S
    )
    if not pattern.search(doc):
        raise ValueError(f"DATASET.md has no generated section {name!r}")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), doc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    raw = download()
    examples, events = load_examples(raw)
    split = make_split(examples)
    by_id = {ex.id: ex for ex in examples}

    if args.check:
        for name in SPLITS:
            committed = (SPLITS_DIR / f"{name}.txt").read_text().split()
            if [int(i) for i in committed] != split[name]:
                print(f"split {name} differs from committed file", file=sys.stderr)
                return 1
        print("splits reproduce exactly")
        return 0

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, SplitStats] = {"All": split_stats(examples)}
    for name in SPLITS:
        path = SPLITS_DIR / f"{name}.txt"
        path.write_text("".join(f"{i}\n" for i in split[name]))
        stats[name.capitalize()] = split_stats([by_id[i] for i in split[name]])

    manifest = {
        "dataset": "SEntFiN 1.0",
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "raw_sha256": SOURCE_SHA256,
        "seed": SEED,
        "fractions": FRACTIONS,
        "unit": "headline; identical normalized headlines kept in one split",
        "stratify_by": "label of single-label headlines x (1 vs 2+ entities); 'mixed' otherwise",
        "files": {
            name: {"count": len(split[name]), "sha256": sha256_file(SPLITS_DIR / f"{name}.txt")}
            for name in SPLITS
        },
        "stats": stats,
        "cleaning": dict(Counter(e.action for e in events)),
    }
    (SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    doc = DATASET_MD.read_text()
    doc = replace_section(doc, "stats", render_stats(stats))
    doc = replace_section(doc, "cleaning", render_events(events))
    DATASET_MD.write_text(doc)

    print(render_stats(stats))
    print(dict(Counter(e.action for e in events)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
