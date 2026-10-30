"""Pricing keys stay stable: every key of the engine's pricing table (tests/data/pricing_keys.json, recorded
with the identity the 2026.09.12.1 default gave it through ``parse_canonical``) keeps that identity, except
the named change classes below, each listed key by key:

- alias fold: a vendor spelling that now folds into its OpenRouter slug;
- version dot: an Anthropic, Z.ai or AI21 minor version spelled with a dash;
- route-named vendor: a vendor segment that is now a route, so the key no longer reads as canonical.

No two keys share an identity unless they shared one before, apart from the dash and dot spellings of one
model that the version dot merges and the vendor spellings that the alias fold merges, both on purpose.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from toloka_model_name_normalizer import ModelNameError, Normalizer, Rules

ROWS = json.loads((Path(__file__).parent / "data" / "pricing_keys.json").read_text("utf-8"))["rows"]
NORM = Normalizer(Rules.default())

ALIAS_FOLD: dict[str, tuple[str, str]] = {
    "moonshot/kimi-k2.5": ("moonshotai", "kimi-k2.5"),
}
VERSION_DOT: dict[str, tuple[str, str]] = {
    "anthropic/claude-sonnet-4-5": ("anthropic", "claude-sonnet-4.5"),
    "anthropic/claude-sonnet-4-6": ("anthropic", "claude-sonnet-4.6"),
}
ROUTE_NAMED_VENDOR: dict[str, str] = {
    "azure_ai/cohere-command-a-plus-05-2026": "MalformedName",
}
# the dash and dot spellings of one model, which the version dot makes one identity
EXPECTED_MERGES: set[frozenset[str]] = {
    frozenset({"anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"}),
    frozenset({"anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"}),
    # one model, two routes with their own prices (Moonshot's API and OpenRouter): the alias fold makes the
    # model one identity, so a price stays keyed by the key as spelled, never by the canonical identity
    frozenset({"moonshot/kimi-k2.5", "moonshotai/kimi-k2.5"}),
}

Outcome = tuple[str, str] | str


def recorded(row: dict) -> Outcome:
    return row["error"] if row["error"] else (row["vendor"], row["model"])


def today(key: str) -> Outcome:
    try:
        name = NORM.parse_canonical(key)
    except ModelNameError as exc:
        return type(exc).__name__
    return (name.vendor, name.model)


def changes() -> dict[str, tuple[Outcome, Outcome]]:
    return {
        row["key"]: (recorded(row), today(row["key"])) for row in ROWS if recorded(row) != today(row["key"])
    }


def test_the_change_classes_have_their_exact_counts() -> None:
    assert (len(ALIAS_FOLD), len(VERSION_DOT), len(ROUTE_NAMED_VENDOR)) == (1, 2, 1)
    keys = {row["key"] for row in ROWS}
    assert set(ALIAS_FOLD) | set(VERSION_DOT) | set(ROUTE_NAMED_VENDOR) <= keys
    changed = changes()
    assert {key: new for key, (_, new) in changed.items() if key in ALIAS_FOLD} == ALIAS_FOLD
    assert {key: new for key, (_, new) in changed.items() if key in VERSION_DOT} == VERSION_DOT
    assert {key: new for key, (_, new) in changed.items() if key in ROUTE_NAMED_VENDOR} == ROUTE_NAMED_VENDOR


def test_every_other_key_keeps_its_recorded_identity() -> None:
    named = set(ALIAS_FOLD) | set(VERSION_DOT) | set(ROUTE_NAMED_VENDOR)
    unexpected = {key: change for key, change in changes().items() if key not in named}
    listing = "\n".join(f"  {key}: {old} -> {new}" for key, (old, new) in sorted(unexpected.items()))
    assert not unexpected, f"{len(unexpected)} pricing keys changed outside the named classes:\n{listing}"


def test_no_two_keys_fold_into_one_identity_unless_named() -> None:
    before: dict[str, Outcome] = {row["key"]: recorded(row) for row in ROWS}
    by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in ROWS:
        outcome = today(row["key"])
        if isinstance(outcome, tuple):
            by_identity[outcome].add(row["key"])
    merges = {
        frozenset(keys)
        for keys in by_identity.values()
        if len(keys) > 1 and len({before[key] for key in keys}) > 1
    }
    assert merges == EXPECTED_MERGES
