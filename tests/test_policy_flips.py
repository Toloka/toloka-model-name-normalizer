"""The golden fixture under every policy value: each flip has a predictable effect on each row, asserted row
by row (no row is skipped because it raised)."""

from __future__ import annotations

import copy
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
    inferred = not isinstance(default, type) and default.provenance.vendor_source == "family"
    if inferred and policy == "bare_name" and value == "error":
        return UnknownVendor  # the refusal comes before the families table
    if not isinstance(default, type) and not (inferred and policy == "bare_name"):
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
    if (default is UnknownVendor or inferred) and policy == "bare_name" and value == "provider_as_vendor":
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
                    return DEFAULT.parse_canonical(name).canonical
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


# ---------------------------------------------------------------------------
# Switches back to the 2026.09.12 reading: each new behaviour of the default has a configuration switch that
# restores the earlier reading on the named rows. Every test shows the default reading first, then the switch.
# ---------------------------------------------------------------------------


def with_override(override: dict) -> Normalizer:
    return Normalizer(Rules.from_dict(override, base=Rules.default()))


def identity(name: ModelName) -> tuple[str, str, str | None]:
    return (name.vendor, name.model, name.route)


def test_family_vendor_off_never_guesses_a_bare_name() -> None:
    assert DEFAULT.parse("claude-opus-4-7").canonical == "anthropic/claude-opus-4.7"
    assert DEFAULT.parse("azure/gpt-5.5").canonical == "openai/gpt-5.5"
    off = Normalizer(Rules.default().with_policy(family_vendor="off"))
    with pytest.raises(UnknownVendor):
        off.parse("claude-opus-4-7")
    with pytest.raises(UnknownVendor):
        off.parse("azure/gpt-5.5")
    assert off.parse("xai/grok-4").vendor == "x-ai", "spelled vendors still fold"


def test_an_empty_minor_dot_list_keeps_the_dash() -> None:
    assert DEFAULT.parse_pair("anthropic", "claude-opus-4-6").model == "claude-opus-4.6"
    flipped = with_override({"canonical": {"minor_dot": []}})
    assert identity(flipped.parse_pair("anthropic", "claude-opus-4-6")) == (
        "anthropic",
        "claude-opus-4-6",
        "anthropic",
    )
    assert flipped.parse_canonical("z-ai/glm-4-7").model == "glm-4-7"


def test_decimal_p_off_keeps_the_fireworks_spelling() -> None:
    assert DEFAULT.parse("fireworks_ai/glm-5p1").model == "glm-5.1"
    route = copy.deepcopy(dict(Rules.default().document["routes"]["fireworks_ai"]))
    route["decimal_p"] = False
    flipped = with_override({"routes": {"fireworks_ai": route}})
    assert flipped.parse("fireworks_ai/glm-5p1").model == "glm-5p1"


def test_vendor_case_preserve_keeps_the_spelled_case() -> None:
    assert DEFAULT.parse("deepinfra/Qwen/Qwen3-32B").vendor == "qwen"
    flipped = Normalizer(Rules.default().with_policy(vendor_case="preserve"))
    assert identity(flipped.parse("deepinfra/Qwen/Qwen3-32B")) == ("Qwen", "Qwen3-32B", "deepinfra")


def test_vendor_case_fold_runs_before_case_reject() -> None:
    norm = Normalizer(Rules.default().with_policy(case="reject"))
    assert norm.parse_canonical("Qwen/qwen3-8b").canonical == "qwen/qwen3-8b"
    with pytest.raises(MalformedName):
        norm.parse_canonical("qwen/Qwen3-8B")


@pytest.mark.parametrize(
    "route, text, new, old",
    [
        (
            "bedrock",
            "bedrock/us.anthropic.claude-sonnet-4-6",
            ("anthropic", "claude-sonnet-4.6", "bedrock"),
            ("bedrock", "us.anthropic.claude-sonnet-4-6", None),
        ),
        (
            "azure",
            "azure/gpt-5.5",
            ("openai", "gpt-5.5", "azure"),
            ("azure", "gpt-5.5", None),
        ),
        (
            "vertex_ai",
            "vertex_ai/claude-opus-4-7@default",
            ("anthropic", "claude-opus-4.7", "vertex_ai"),
            ("vertex_ai", "claude-opus-4-7@default", None),
        ),
    ],
)
def test_remove_routes_restores_the_canonical_reading(
    route: str, text: str, new: tuple[str, str, str | None], old: tuple[str, str, str | None]
) -> None:
    assert identity(DEFAULT.parse(text)) == new
    assert identity(with_override({"remove_routes": [route]}).parse(text)) == old


def test_an_alias_mapped_to_itself_keeps_the_spelling() -> None:
    assert DEFAULT.parse("xai/grok-4").vendor == "x-ai"
    assert with_override({"vendor_aliases": {"xai": "xai"}}).parse("xai/grok-4").vendor == "xai"
    assert DEFAULT.parse("replicate/meta/llama-3-70b").vendor == "meta-llama"
    flipped = with_override({"family_vendor_aliases": {"meta/llama": "meta"}})
    assert flipped.parse("replicate/meta/llama-3-70b").vendor == "meta"


def test_the_separators_without_the_at_sign_restore_the_family() -> None:
    assert DEFAULT.parse_canonical("mistralai/codestral@2405").family == "codestral"
    flipped = with_override({"fields": {"separators": "[-/\\s()]+"}})
    assert flipped.parse_canonical("mistralai/codestral@2405").family == "codestral@2405"
