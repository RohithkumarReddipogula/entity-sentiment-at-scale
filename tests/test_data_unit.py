"""Unit tests for parsing, name repair and span finding. No raw data needed (runs in CI)."""

from __future__ import annotations

import pytest

from entsent.data import build_example, find_spans, load_split, parse_decisions, repair_name


def test_parse_valid_literal() -> None:
    decisions, fallback = parse_decisions("{'Nike': 'positive', 'Adidas': 'negative'}")
    assert decisions == {"Nike": "positive", "Adidas": "negative"}
    assert not fallback


def test_parse_unescaped_apostrophe_uses_fallback() -> None:
    decisions, fallback = parse_decisions("{'Sebi': 'neutral', 'Osian's Art Fund': 'neutral'}")
    assert decisions == {"Sebi": "neutral", "Osian's Art Fund": "neutral"}
    assert fallback


def test_parse_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        parse_decisions("{'Nike': 'bullish'}")


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("HSBC", "HSBC Holdings board", "HSBC"),
        ("HSBC neutralings", "Analyst joins the HSBC Holdings board", "HSBC Holdings"),
        ("FTSE Rusnegative", "FTSE Russell adds two new indexes", "FTSE Russell"),
        ("Frank Rusnegative Co", "deal for Frank Russell Co", "Frank Russell Co"),
        ("Acme neutralings", "Acme Group shares fall", None),
        ("Missing Co", "Nothing here", None),
    ],
)
def test_repair_name(name: str, text: str, expected: str | None) -> None:
    assert repair_name(name, text) == expected


def test_spans_respect_left_word_boundary() -> None:
    text = "Jet, SpiceJet shares soar on Etihad deal"
    spans = find_spans(["Jet", "SpiceJet"], text)
    assert spans["Jet"] == ((0, 3),)
    assert spans["SpiceJet"] == ((5, 13),)


def test_spans_allow_dropped_apostrophe_suffix() -> None:
    text = "Acmes reading of the new rules worries analysts"
    assert find_spans(["Acme"], text)["Acme"] == ((0, 4),)


def test_spans_keep_every_mention() -> None:
    text = "Infosys up; analysts back Infosys"
    assert find_spans(["Infosys"], text)["Infosys"] == ((0, 7), (26, 33))


def test_spans_drop_mention_inside_longer_entity() -> None:
    text = "Auto, Pharma slip; Orion Motors, Sun Pharma lose ground"
    spans = find_spans(["Pharma", "Sun Pharma"], text)
    assert spans["Pharma"] == ((6, 12),)
    assert spans["Sun Pharma"] == ((33, 43),)


def test_build_example_repairs_and_drops() -> None:
    text = "WNS Holdings names a new finance chief"
    example, events = build_example(
        1, text, {"WNS neutralings": "neutral", "Ghost Corp": "negative"}
    )
    assert [(e.name, e.spans) for e in example.entities] == [("WNS Holdings", ((0, 12),))]
    assert [(ev.action, ev.original) for ev in events] == [
        ("repaired", "WNS neutralings"),
        ("dropped", "Ghost Corp"),
    ]


def test_example_flags() -> None:
    text = "Nike beats forecasts while Adidas cuts its outlook."
    mixed, _ = build_example(1, text, {"Nike": "positive", "Adidas": "negative"})
    same, _ = build_example(2, text, {"Nike": "neutral", "Adidas": "neutral"})
    single, _ = build_example(3, text, {"Nike": "positive"})
    assert mixed.is_multi and mixed.is_mixed
    assert same.is_multi and not same.is_mixed
    assert not single.is_multi and not single.is_mixed


def test_test_split_is_sealed() -> None:
    with pytest.raises(PermissionError):
        load_split("test")
