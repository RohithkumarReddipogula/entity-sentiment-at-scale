"""Checks on the committed split files. No raw data needed (runs in CI)."""

from __future__ import annotations

import json

from entsent.data import SPLITS, SPLITS_DIR, read_split_ids, sha256_file
from entsent.splits import FRACTIONS

N_HEADLINES = 10_753


def test_split_files_match_manifest_hashes() -> None:
    manifest = json.loads((SPLITS_DIR / "manifest.json").read_text())
    for name in SPLITS:
        entry = manifest["files"][name]
        assert sha256_file(SPLITS_DIR / f"{name}.txt") == entry["sha256"]
        assert len(read_split_ids(name)) == entry["count"]


def test_splits_are_disjoint_and_cover_all_headlines() -> None:
    ids = {name: read_split_ids(name) for name in SPLITS}
    for name, part in ids.items():
        assert len(part) == len(set(part)), f"duplicate IDs in {name}"
        assert part == sorted(part)
    train, dev, test = (set(ids[n]) for n in SPLITS)
    assert not train & dev and not train & test and not dev & test
    assert train | dev | test == set(range(1, N_HEADLINES + 1))


def test_split_sizes_are_70_15_15() -> None:
    for name, target in FRACTIONS.items():
        share = len(read_split_ids(name)) / N_HEADLINES
        assert abs(share - target) < 0.005, f"{name}: {share:.3f}"
