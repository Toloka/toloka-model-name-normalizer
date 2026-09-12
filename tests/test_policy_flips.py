"""The golden fixture under every policy value: each flip has a predictable effect on each row, asserted row
by row (no row is skipped because it raised)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer import (
    MalformedName,
    ModelName,
    ModelNameError,
    Normalizer,
    Rules,
    UnknownRoute,
    UnknownVendor,
    UnresolvedFields,
)

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text("utf-8"))
ROWS = [row for row in GOLDEN["rows"] if row["input_kind"] in ("pair", "text", "canonical", "routed", "slug")]
DEFAULT = Normalizer(Rules.default())


def run(normalizer: Normalizer, row: dict) -> ModelName:
    kind = row["input_kind"]
    if kind == "pair":
        return normalizer.parse_pair(row["provider"], row["input"])
    if kind == "canonical":
        return normalizer.parse_canonical(row["input"])
    if kind == "routed":
        return normalizer.parse_routed(row["input"])
    return normalizer.parse(row["input"])


def outcome(normalizer: Normalizer, row: dict) -> ModelName | type[ModelNameError]:
    try:
        return run(normalizer, row)
    except ModelNameError as exc:
        return type(exc)


FLIPS = [
    ("case", "fold"),
    ("case", "reject"),
    ("bare_name", "error"),
    ("bare_name", "provider_as_vendor"),
    ("unknown_provider", "error"),
    ("leading_slash", "reject"),
    ("implied_vendor_mismatch", "name_wins"),
    ("fields", "strict"),
]


def expected_under(policy: str, value: str, row: dict, default: ModelName | type[ModelNameError]) -> object:
    """What the flipped policy must do to this row, given the default outcome."""
    if not isinstance(default, type):
        canonical = default.canonical
        if policy == "case" and value == "fold":
            return canonical.lower()
        if policy == "case" and value == "reject":
            return MalformedName if canonical != canonical.lower() else canonical
        if policy == "unknown_provider" and value == "error":
            undeclared = row["input_kind"] == "pair" and row["provider"].lower() not in DEFAULT.rules.routes
            return UnknownRoute if undeclared else canonical
        if policy == "leading_slash" and value == "reject":
            return MalformedName if row["input"].startswith("/") else canonical
        if policy == "fields" and value == "strict":
            return UnresolvedFields if default.unresolved else canonical
        return canonical
    if default is UnknownVendor and policy == "bare_name" and value == "provider_as_vendor":
        # a bare name that arrived with a route (the pair form, or a peelable route prefix in a text) takes
        # the route as its vendor; a bare text without any route still has nothing to take
        if row["input_kind"] == "pair" and "/" not in row["input"]:
            return f"{row['provider'].lower()}/{row['input']}"
        first, _, rest = row["input"].lstrip("/").partition("/")
        route = DEFAULT.rules.routes.get(first.lower())
        if route is not None and route.peelable and rest and "/" not in rest:
            return f"{route.name}/{rest}"
        return UnknownVendor
    if default is MalformedName and policy == "implied_vendor_mismatch" and value == "name_wins":
        # the only MalformedName the flip can lift: a vendor/model name under a route implying another vendor
        if row["input_kind"] == "pair":
            route = DEFAULT.rules.routes.get(row["provider"].lower())
            name = row["input"]
            if route is not None and route.implies_vendor and "/" in name:
                vendor, _, model = name.partition("/")
                if vendor.lower() != route.implies_vendor and "/" not in model:
                    return name
        return MalformedName
    return default


@pytest.mark.parametrize("policy, value", FLIPS, ids=[f"{p}={v}" for p, v in FLIPS])
def test_every_row_behaves_as_the_flip_predicts(policy: str, value: str) -> None:
    flipped = Normalizer(Rules.default().with_policy(**{policy: value}))
    checked = 0
    for row in ROWS:
        default = outcome(DEFAULT, row)
        want = expected_under(policy, value, row, default)
        got = outcome(flipped, row)
        got_value = got if isinstance(got, type) else got.canonical
        assert got_value == want, (policy, value, row["input_kind"], row.get("provider"), row["input"])
        checked += 1
    assert checked == len(ROWS)
