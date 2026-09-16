"""Renderings: the routed round trip, the slug and its documented divergences from model_id_slug."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer import MalformedName, Normalizer, Rules, UnknownRoute

norm = Normalizer(Rules.default())
GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text("utf-8"))
ROUTED_ROWS = [row for row in GOLDEN["rows"] if row.get("expected") and row["expected"]["routes"]]
DECLARED = [row for row in ROUTED_ROWS if row["expected"]["routes"][0] in norm.rules.routes]
UNDECLARED = [row for row in ROUTED_ROWS if row["expected"]["routes"][0] not in norm.rules.routes]


@pytest.mark.parametrize("row", DECLARED, ids=[row["expected"]["routed"] for row in DECLARED])
def test_routed_round_trips_identity_and_chain(row: dict) -> None:
    expected = row["expected"]
    name = norm.parse_routed(expected["routed"])
    assert (name.vendor, name.model, list(name.routes)) == (
        expected["vendor"],
        expected["model"],
        expected["routes"],
    )
    assert name.routed == expected["routed"]


@pytest.mark.parametrize("row", UNDECLARED, ids=[row["expected"]["routed"] for row in UNDECLARED])
def test_undeclared_provider_renders_but_only_the_pair_reads_it_back(row: dict) -> None:
    """(xai, x-ai/grok-4) renders xai/x-ai/grok-4; the wire form of an undeclared provider is not a declared
    route, so parse_routed refuses it and parse_pair is the way back."""
    expected = row["expected"]
    with pytest.raises(UnknownRoute):
        norm.parse_routed(expected["routed"])
    name = norm.parse_pair(row["provider"], row["input"])
    assert (name.vendor, name.model, list(name.routes)) == (
        expected["vendor"],
        expected["model"],
        expected["routes"],
    )


def test_route_less_value_renders_canonical_and_slug() -> None:
    name = norm.parse_canonical("openai/gpt-6-astra")
    assert name.routed == "openai/gpt-6-astra"
    assert name.slug == "openai_gpt-6-astra"


def test_slug_is_the_identity_whatever_the_route() -> None:
    """The route is a property of the value, never part of the slug (owner decision 2026-09-12)."""
    for name in (
        norm.parse_pair("openrouter", "openai/gpt-6-astra"),
        norm.parse("platform_openai/gpt-6-astra"),
        norm.parse_routed("litellm/openrouter/openai/gpt-6-astra"),
        norm.parse_canonical("openai/gpt-6-astra"),
    ):
        assert name.slug == "openai_gpt-6-astra"
    assert norm.parse_pair("meta", "muse-spark-1.2").slug == "meta_muse-spark-1.2"
    assert norm.parse_pair("anthropic", "claude-fable-5.1").slug == "anthropic_claude-fable-5.1"
    assert norm.parse_pair("meta", "muse-spark-1.2").routed == "meta/muse-spark-1.2"


ENGINE = Normalizer(Rules.from_dict({"render": {"route_in_slug": True}}, base=Rules.default()))


def test_route_in_slug_agrees_with_model_id_slug_on_the_engine_shapes() -> None:
    cases = {
        ("openrouter", "anthropic/claude-opus-4.7"): "openrouter__anthropic_claude-opus-4.7",
        ("openai", "gpt-5.5"): "openai__gpt-5.5",
        ("nova", "amazon-nova-micro-1_0"): "nova__amazon-nova-micro-1_0",
        ("xai", "x-ai/grok-4"): "xai__x-ai_grok-4",
        ("anthropic", "anthropic/claude-sonnet-4.7-20260101"): "anthropic__claude-sonnet-4.7-20260101",
        ("meta", "muse-spark-1.2"): "meta__muse-spark-1.2",
    }
    for (provider, name), expected in cases.items():
        assert ENGINE.parse_pair(provider, name).slug == expected
    assert ENGINE.parse("platform_openai/gpt-6-astra").slug == "platform_openai__gpt-6-astra"


def test_route_in_slug_never_invents_a_route() -> None:
    # the engine's model_id_slug("", name) says unknown__<name>; the library never writes a placeholder
    assert ENGINE.parse_canonical("openai/gpt-6-astra").slug == "openai_gpt-6-astra"
    assert ENGINE.parse("anthropic/claude-3.7-sonnet:thinking").slug == "anthropic_claude-3.7-sonnet_thinking"


def test_slug_documented_divergences_from_model_id_slug() -> None:
    # The engine renders the raw pair (openai__openai_gpt-5.5); the library renders the identity.
    assert ENGINE.parse_pair("openai", "openai/gpt-5.5").slug == "openai__gpt-5.5"
    assert ENGINE.parse_pair("qwen", "qwen/qwen3-next").slug == "qwen__qwen3-next"


def test_unsafe_characters_collapse() -> None:
    assert norm.parse_pair("nova", "Nova Pro v3.4 (Preview)").slug == "nova_Nova_Pro_v3.4_Preview"
    assert norm.parse("anthropic/claude-3.7-sonnet:thinking").slug == "anthropic_claude-3.7-sonnet_thinking"
    assert ENGINE.parse_pair("nova", "Nova Pro v3.4 (Preview)").slug == "nova__Nova_Pro_v3.4_Preview"


def test_slugs_are_never_parsed() -> None:
    for slug in (
        "openrouter__openai_gpt-5.4",
        "meta__muse-spark-1.2",
        "claude-code__anthropic_claude-sonnet-4-6",
        "oracle__adhoc",
    ):
        with pytest.raises(MalformedName):
            norm.parse(slug)
        with pytest.raises(MalformedName):
            norm.parse_pair("openrouter", slug)
