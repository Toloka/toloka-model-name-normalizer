"""The golden corpus: every raw string of the survey (docs/model-name-canonicaliser/CORPUS.md) with its input
kind and the hand-verified or reviewed expectation. Every row, every field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer import CallableResolver, ModelName, Normalizer, Rules, errors
from toloka_model_name_normalizer.model import DESCRIPTIVE_FIELDS

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text("utf-8"))
ROWS = [row for row in GOLDEN["rows"] if row["input_kind"] != "label"]
IDS = [f"{row['input_kind']}|{row.get('provider') or ''}|{row['input']}" for row in ROWS]


@pytest.fixture(scope="module")
def normalizer() -> Normalizer:
    return Normalizer(Rules.default())


def run(normalizer: Normalizer, row: dict) -> ModelName:
    kind = row["input_kind"]
    if kind == "pair":
        return normalizer.parse_pair(row["provider"], row["input"])
    if kind in ("text", "slug"):
        return normalizer.parse(row["input"])
    if kind == "canonical":
        return normalizer.parse_canonical(row["input"])
    if kind == "routed":
        return normalizer.parse_routed(row["input"])
    if kind == "text+resolver":
        candidates = tuple(row["candidates"])
        resolver = CallableResolver(lambda bare, c=candidates: c, fingerprint="pricing-keys-2026-09-02")
        return Normalizer(normalizer.rules, resolver=resolver).parse(row["input"])
    raise AssertionError(f"unknown input kind {kind!r}")


@pytest.mark.parametrize("row", ROWS, ids=IDS)
def test_golden_row(normalizer: Normalizer, row: dict) -> None:
    if row["error"]:
        with pytest.raises(getattr(errors, row["error"]["type"])):
            run(normalizer, row)
        return
    expected = row["expected"]
    name = run(normalizer, row)
    assert (name.vendor, name.model, name.route, list(name.routes)) == (
        expected["vendor"],
        expected["model"],
        expected["route"],
        expected["routes"],
    )
    assert name.canonical == expected["canonical"]
    assert name.routed == expected["routed"]
    assert name.slug == expected["slug"]
    assert {key: getattr(name, key) for key in DESCRIPTIVE_FIELDS} == expected["fields"]
    assert list(name.unresolved) == expected["unresolved"]
    assert name.provenance.lookup is expected["lookup"]
    assert name.provenance.input_kind == (
        "text" if row["input_kind"] in ("slug", "text+resolver") else row["input_kind"]
    )


def test_corpus_coverage_matches_fixture() -> None:
    """The counters the design quotes come from this very fixture."""
    coverage = json.loads((Path(__file__).parent / "data" / "coverage.json").read_text("utf-8"))
    identities: dict[str, str] = {}
    for row in GOLDEN["rows"]:
        expected = row.get("expected")
        if expected and not row["classes"][0].startswith("review_"):
            identities.setdefault(expected["canonical"], row["tier"])
    assert sum(1 for tier in identities.values() if tier == "A") == coverage["identities"]["identities_A"]
    assert sum(1 for tier in identities.values() if tier == "B") == coverage["identities"]["identities_B"]
    assert len(GOLDEN["rows"]) == coverage["rows"]


def test_every_tier_a_identity_has_vendor_and_family(normalizer: Normalizer) -> None:
    seen = set()
    for row in ROWS:
        if row.get("tier") != "A" or not row["expected"]:
            continue
        name = run(normalizer, row)
        assert name.vendor and name.family, row["input"]
        seen.add(name.canonical)
    assert len(seen) == 60
