"""Equality is exact tuple equality on (vendor, model, route); everything else is excluded."""

from __future__ import annotations

from toloka_model_name_normalizer import Normalizer, Rules

norm = Normalizer(Rules.default())


def test_equality_is_transitive_over_route_values() -> None:
    a = norm.parse_routed("openrouter/openai/gpt-6-astra")
    b = norm.parse_canonical("openai/gpt-6-astra")
    c = norm.parse_routed("platform_openai/gpt-6-astra")
    assert a != b and b != c and a != c
    assert a.same_model(b) and b.same_model(c) and a.same_model(c)
    assert len({a, b, c}) == 3
    assert {a: 1, b: 2, c: 3}[norm.parse("openrouter/openai/gpt-6-astra")] == 1


def test_descriptive_fields_and_chain_do_not_take_part() -> None:
    long_chain = norm.parse_routed("litellm/openrouter/openai/gpt-6-astra")
    short_chain = norm.parse_routed("litellm/openai/gpt-6-astra")
    assert long_chain == short_chain
    assert hash(long_chain) == hash(short_chain)
    assert long_chain.routed != short_chain.routed
    assert not long_chain.same_wire(short_chain)
    assert long_chain.same_wire(norm.parse_routed("litellm/openrouter/openai/gpt-6-astra"))


def test_none_route_is_a_regular_value() -> None:
    assert norm.parse_canonical("openai/gpt-6-astra") == norm.parse("openai/gpt-6-astra")
    assert norm.parse_canonical("openai/gpt-6-astra") != norm.parse_pair("openai", "gpt-6-astra")


def test_same_model_ignores_case_only_when_the_identity_does() -> None:
    assert not norm.parse("OpenAI/GPT-6-Astra").same_model(norm.parse("openai/gpt-6-astra"))
    folded = Normalizer(Rules.default().with_policy(case="fold"))
    assert folded.parse("OpenAI/GPT-6-Astra").same_model(folded.parse("openai/gpt-6-astra"))


def test_tags_and_fields() -> None:
    name = norm.parse_pair("openrouter", "openai/gpt-6-astra")
    assert name.tags() == ["model:openai/gpt-6-astra", "model_vendor:openai", "model_family:gpt"]
    fields = name.fields()
    assert fields["generation"] == "6" and fields["variant"] == "astra" and fields["tier"] is None
    assert set(fields) == {
        "vendor",
        "family",
        "generation",
        "tier",
        "variant",
        "snapshot",
        "size",
        "stage",
        "capability",
        "api_version",
    }
    as_dict = name.to_dict()
    assert as_dict["canonical"] == "openai/gpt-6-astra" and as_dict["provenance"]["input_kind"] == "pair"
