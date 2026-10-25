"""The README and model card: plain style, correct citations, numbers generated from results/."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "MODEL_CARD.md"]
HYPE = [
    "revolutionary", "cutting-edge", "cutting edge", "state-of-the-art", "state of the art",
    "seamless", "blazing", "game-changing", "game changer", "supercharge", "unlock", "leverage",
    "effortless", "world-class", "groundbreaking", "robust", "powerful", "delve", "harness",
]  # fmt: skip


@pytest.mark.parametrize("doc", DOCS)
def test_no_em_dashes(doc: str) -> None:
    text = (REPO / doc).read_text()
    assert "—" not in text, f"{doc} has an em dash"


@pytest.mark.parametrize("doc", DOCS)
def test_no_hype_words(doc: str) -> None:
    text = (REPO / doc).read_text().lower()
    found = [w for w in HYPE if re.search(rf"\b{re.escape(w)}\b", text)]
    assert not found, f"{doc}: {found}"


def test_readme_cites_meltwater_and_sentfin() -> None:
    text = (REPO / "README.md").read_text()
    url = "https://underthehood.meltwater.com/blog/2026/04/23/rethinking-entity-level-sentiment-at-scale/"
    assert url in text
    assert "Doing More With Less: Rethinking Entity-Level Sentiment at Scale" in text
    assert "https://doi.org/10.1002/asi.24634" in text and "73(9), 1314-1335" in text
    assert "MIT license" in text and "Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo" in text


def test_readme_results_block_is_generated_and_current() -> None:
    spec = importlib.util.spec_from_file_location("make_results", REPO / "scripts/make_results.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    begin, end = module.README_MARKERS
    current = (REPO / "README.md").read_text().split(begin, 1)[1].split(end, 1)[0]
    assert current == "\n" + module.readme_block(), "README results block is stale: make results"


def test_linked_files_exist() -> None:
    for doc in DOCS:
        for target in re.findall(r"\]\((?!https?://)([^)#]+)\)", (REPO / doc).read_text()):
            assert (REPO / target).exists(), f"{doc} links to missing {target}"
