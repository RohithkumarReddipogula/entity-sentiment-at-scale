"""SEntFiN 1.0 loader: download, parse, repair entity names, locate character spans, splits.

The raw CSV is never committed (see DATASET.md). It is downloaded at a pinned commit and
verified by SHA-256. Split membership lives in ``data/splits/{train,dev,test}.txt``.
"""

from __future__ import annotations

import ast
import hashlib
import re
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

Label = Literal["positive", "negative", "neutral"]
LABELS: tuple[Label, ...] = get_args(Label)
SplitName = Literal["train", "dev", "test"]
SPLITS: tuple[SplitName, ...] = get_args(SplitName)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "sentfin" / "SEntFiN.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

SOURCE_COMMIT = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
SOURCE_URL = f"https://raw.githubusercontent.com/pyRis/SEntFiN/{SOURCE_COMMIT}/SEntFiN.csv"
SOURCE_SHA256 = "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"

# The upstream annotation pipeline ran a find-and-replace that turned "Hold" into "neutral"
# and "sell" into "negative" inside entity names ("HSBC neutralings", "FTSE Rusnegative").
# We undo it, but only accept the result if the repaired name occurs verbatim in the headline.
_CORRUPTIONS: tuple[tuple[str, str], ...] = (("neutral", "Hold"), ("negative", "sell"))
_PAIR = re.compile(r"""['"](.+?)['"]: '(positive|negative|neutral)'""")


@dataclass(frozen=True)
class Entity:
    name: str
    label: Label
    spans: tuple[tuple[int, int], ...]  # every mention as [start, end) character offsets


@dataclass(frozen=True)
class Example:
    id: int
    text: str
    entities: tuple[Entity, ...]

    @property
    def is_multi(self) -> bool:
        return len(self.entities) >= 2

    @property
    def is_mixed(self) -> bool:
        return len({e.label for e in self.entities}) >= 2


@dataclass(frozen=True)
class CleaningEvent:
    id: int
    action: Literal["repaired", "dropped", "parsed_with_fallback"]
    original: str
    result: str
    text: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path = RAW_PATH) -> Path:
    """Download the pinned raw CSV if missing, then verify its checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE_URL, tmp)
        tmp.rename(path)
    digest = sha256_file(path)
    if digest != SOURCE_SHA256:
        raise ValueError(f"{path}: sha256 {digest} != expected {SOURCE_SHA256}")
    return path


def parse_decisions(raw: str) -> tuple[dict[str, Label], bool]:
    """Parse the ``Decisions`` column. Returns (entity -> label, used_fallback).

    Most rows are valid Python dict literals; a few contain unescaped quotes
    (``'Osian's Art Fund'``) and are parsed with a regex instead.
    """
    try:
        value = ast.literal_eval(raw)
        fallback = False
    except (SyntaxError, ValueError):
        value = dict(_PAIR.findall(raw))
        fallback = True
    if not isinstance(value, dict) or not value:
        raise ValueError(f"cannot parse decisions: {raw!r}")
    out: dict[str, Label] = {}
    for name, label in value.items():
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r} in {raw!r}")
        out[str(name)] = label
    return out, fallback


def repair_name(name: str, text: str) -> str | None:
    """Return ``name`` if it occurs in ``text``, a repaired name if that occurs, else None."""
    if name in text:
        return name
    candidates = {name}
    for corrupt, original in _CORRUPTIONS:
        candidates |= {c.replace(corrupt, original) for c in candidates}
    for candidate in sorted(candidates):
        if candidate != name and candidate in text:
            return candidate
    return None


def find_spans(names: Sequence[str], text: str) -> dict[str, tuple[tuple[int, int], ...]]:
    """Locate every mention of each entity name in ``text``.

    Matches are case-sensitive and must start at a word boundary (so "Jet" does not match
    inside "SpiceJet"); the end may run into a suffix ("Sebis", "Nifty50") because the
    headlines often drop apostrophes. If no boundary match exists, any substring match is
    used. Mentions lying inside a longer entity's mention are removed ("Pharma" inside
    "Sun Pharma"), so spans of different entities never overlap.
    """
    found: dict[str, list[tuple[int, int]]] = {}
    for name in names:
        bounded = [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(name), text)]
        anywhere = [(m.start(), m.end()) for m in re.finditer(re.escape(name), text)]
        found[name] = bounded or anywhere

    def inside_longer(name: str, span: tuple[int, int]) -> bool:
        return any(
            len(other) > len(name) and s <= span[0] and span[1] <= e
            for other, spans in found.items()
            if other != name
            for s, e in spans
        )

    return {
        name: tuple(sp for sp in spans if not inside_longer(name, sp))
        for name, spans in found.items()
    }


def build_example(
    row_id: int, text: str, decisions: dict[str, Label]
) -> tuple[Example, list[CleaningEvent]]:
    events: list[CleaningEvent] = []
    resolved: dict[str, Label] = {}
    for name, label in decisions.items():
        fixed = repair_name(name, text)
        if fixed is None:
            events.append(CleaningEvent(row_id, "dropped", name, "name not in text", text))
            continue
        if fixed != name:
            events.append(CleaningEvent(row_id, "repaired", name, fixed, text))
        resolved[fixed] = label
    spans = find_spans(list(resolved), text)
    entities: list[Entity] = []
    for name, label in resolved.items():
        if not spans[name]:
            reason = "only occurs inside a longer entity"
            events.append(CleaningEvent(row_id, "dropped", name, reason, text))
            continue
        entities.append(Entity(name, label, spans[name]))
    return Example(row_id, text, tuple(entities)), events


def load_examples(path: Path = RAW_PATH) -> tuple[list[Example], list[CleaningEvent]]:
    """Load and clean every headline. Headlines left with no entity are dropped."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"S No.": int, "Title": str, "Decisions": str})
    examples: list[Example] = []
    events: list[CleaningEvent] = []
    for row_id, text, raw in zip(df["S No."], df["Title"], df["Decisions"], strict=True):
        decisions, fallback = parse_decisions(raw)
        if fallback:
            events.append(CleaningEvent(row_id, "parsed_with_fallback", raw, repr(decisions), text))
        example, row_events = build_example(int(row_id), text, decisions)
        events.extend(row_events)
        if example.entities:
            examples.append(example)
        else:
            events.append(CleaningEvent(row_id, "dropped", text, "no entities left", text))
    return examples, events


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def read_split_ids(name: SplitName, splits_dir: Path = SPLITS_DIR) -> list[int]:
    lines = (splits_dir / f"{name}.txt").read_text().split()
    return [int(x) for x in lines]


def write_split_ids(ids: Iterable[int], path: Path) -> None:
    path.write_text("".join(f"{i}\n" for i in sorted(ids)))


def load_split(
    name: SplitName,
    *,
    allow_test: bool = False,
    raw_path: Path = RAW_PATH,
    splits_dir: Path = SPLITS_DIR,
) -> list[Example]:
    """Examples of one split, in ID order. The test split is sealed until M6."""
    if name == "test" and not allow_test:
        raise PermissionError("the test split is used once, in M6: pass allow_test=True there")
    by_id = {ex.id: ex for ex in load_examples(raw_path)[0]}
    return [by_id[i] for i in read_split_ids(name, splits_dir)]
