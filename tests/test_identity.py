"""The identity grammar: one entry point per carrier, the policies, the errors."""

from __future__ import annotations

import pytest

from toloka_model_name_normalizer import (
    AmbiguousModelName,
    CallableResolver,
    MalformedName,
    Normalizer,
    Rules,
    UnknownRoute,
    UnknownVendor,
)


@pytest.fixture(scope="module")
def norm() -> Normalizer:
    return Normalizer(Rules.default())


class TestCarriers:
    def test_canonical_string_has_no_route_even_for_a_direct_vendor_route_name(
        self, norm: Normalizer
    ) -> None:
        name = norm.parse("openai/gpt-6-astra")
        assert (name.vendor, name.model, name.route, name.routes) == ("openai", "gpt-6-astra", None, ())
        assert norm.parse_canonical("openai/gpt-6-astra") == name

    def test_pair_form_keeps_the_provider_as_route(self, norm: Normalizer) -> None:
        name = norm.parse_pair("openai", "gpt-6-astra")
        assert (name.vendor, name.model, name.route) == ("openai", "gpt-6-astra", "openai")
        assert name.routed == "openai/gpt-6-astra"
        assert name.slug == "openai__gpt-6-astra"

    def test_routed_reads_a_direct_vendor_route(self, norm: Normalizer) -> None:
        assert norm.parse_routed("openai/gpt-6-astra").route == "openai"

    def test_peelable_route_is_peeled_by_the_generic_parser(self, norm: Normalizer) -> None:
        name = norm.parse("platform_openai/gpt-6-astra")
        assert (name.vendor, name.route, name.routed) == (
            "openai",
            "platform_openai",
            "platform_openai/gpt-6-astra",
        )

    def test_nested_routes_keep_the_chain(self, norm: Normalizer) -> None:
        name = norm.parse("litellm/openrouter/anthropic/claude-opus-5")
        assert name.routes == ("litellm", "openrouter")
        assert name.route == "litellm"
        assert name.routed == "litellm/openrouter/anthropic/claude-opus-5"
        assert norm.parse_routed(name.routed).routes == name.routes

    def test_direct_vendor_route_is_not_peeled_after_a_nestable_route(self, norm: Normalizer) -> None:
        name = norm.parse("litellm/openai/gpt-6-astra")
        assert (name.vendor, name.routes) == ("openai", ("litellm",))

    def test_pair_with_implying_route_and_bare_name(self, norm: Normalizer) -> None:
        assert norm.parse_pair("meta", "muse-spark-1.2").canonical == "meta/muse-spark-1.2"
        assert norm.parse_pair("anthropic", "claude-opus-4-6").generation == "4.6"

    def test_unknown_provider_becomes_a_direct_route(self, norm: Normalizer) -> None:
        name = norm.parse_pair("xai", "x-ai/grok-4")
        assert (name.vendor, name.route, name.routed) == ("x-ai", "xai", "xai/x-ai/grok-4")

    def test_idempotence_guard(self, norm: Normalizer) -> None:
        assert norm.parse_pair("openai", "openai/gpt-6-astra").canonical == "openai/gpt-6-astra"
        assert (
            norm.parse_pair("openrouter", "openrouter/openai/gpt-6-astra").canonical == "openai/gpt-6-astra"
        )

    def test_lookup_wins_in_every_entry_point(self, norm: Normalizer) -> None:
        for name in (norm.parse("openrouter/auto"), norm.parse_canonical("openrouter/auto")):
            assert (name.vendor, name.model, name.route) == ("openrouter", "auto", None)
            assert name.provenance.lookup is True
        pair = norm.parse_pair("openrouter", "auto")
        assert (pair.vendor, pair.route, pair.routed) == ("openrouter", "openrouter", "openrouter/auto")
        wire = norm.parse_routed("openrouter/auto")
        assert (wire.vendor, wire.model, wire.route, wire.provenance.lookup) == (
            "openrouter",
            "auto",
            "openrouter",
            True,
        )
        assert wire == pair
        assert norm.parse_routed("litellm/openrouter/auto").routes == ("litellm", "openrouter")

    def test_routed_refuses_an_undeclared_first_segment(self, norm: Normalizer) -> None:
        with pytest.raises(UnknownRoute):
            norm.parse_routed("deepseek/deepseek-v4")

    def test_case_is_preserved_by_default(self, norm: Normalizer) -> None:
        assert norm.parse_pair("nova", "Nova Pro v3").canonical == "nova/Nova Pro v3"
        assert norm.parse("OpenAI/GPT-6-Astra").vendor == "OpenAI"
        assert norm.parse_pair("OpenRouter", "openai/gpt-6-astra").route == "openrouter"


class TestMalformed:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "/",
            "openrouter/",
            "foo/bar/baz",
            "openrouter__openai_gpt-5.4",
            "openai/gpt-6-astra​",
            "openrouter/platform_openai/gpt-6-astra",
            "~openai/gpt-latest",
            "anthropic/claude-opus-4.6/4.7",
        ],
    )
    def test_generic_parser_rejects(self, norm: Normalizer, text: str) -> None:
        with pytest.raises((MalformedName, UnknownVendor)):
            norm.parse(text)

    @pytest.mark.parametrize(
        "provider, name",
        [
            ("nova", ""),
            ("openai", ""),
            ("openrouter/", "x/y"),
            ("Open Router", "x/y"),
            ("~openai", "gpt-4o"),
            ("openrouter", "openrouter__openai_gpt-5.4"),
            ("openai", "anthropic/claude-opus-4-6"),
            ("", "x/y"),
        ],
    )
    def test_pair_rejects(self, norm: Normalizer, provider: str, name: str) -> None:
        with pytest.raises(MalformedName):
            norm.parse_pair(provider, name)

    def test_canonical_rejects_a_peelable_route_as_vendor(self, norm: Normalizer) -> None:
        with pytest.raises(MalformedName):
            norm.parse_canonical("platform_openai/gpt-6-astra")

    def test_repeated_route_is_rejected(self, norm: Normalizer) -> None:
        with pytest.raises(MalformedName):
            norm.parse_routed("litellm/litellm/openrouter/openai/gpt-6-astra")

    def test_non_string_input_is_a_type_error(self, norm: Normalizer) -> None:
        with pytest.raises(TypeError):
            norm.parse(42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            norm.parse_pair(None, "x/y")  # type: ignore[arg-type]

    def test_inner_spaces_and_parentheses_stay_legal(self, norm: Normalizer) -> None:
        assert norm.parse_canonical("nova/Nova Pro v3.4 (Suez)").model == "Nova Pro v3.4 (Suez)"


class TestResolver:
    def test_bare_name_without_resolver(self, norm: Normalizer) -> None:
        with pytest.raises(UnknownVendor):
            norm.parse("muse-spark-1.1")

    def test_single_candidate_resolves_and_records_the_fingerprint(self) -> None:
        resolver = CallableResolver(
            lambda bare: ["meta/muse-spark-1.1"] if bare == "muse-spark-1.1" else [], "cat-v1"
        )
        name = Normalizer(Rules.default(), resolver).parse("muse-spark-1.1")
        assert name.canonical == "meta/muse-spark-1.1"
        assert name.provenance.resolver_fingerprint == "cat-v1"

    def test_pair_form_does_not_consult_the_resolver(self) -> None:
        resolver = CallableResolver(lambda bare: ["amazon", "nova"], "cat-v1")
        name = Normalizer(Rules.default(), resolver).parse_pair("nova", "nova-pro-v1")
        assert name.canonical == "nova/nova-pro-v1"
        assert name.provenance.resolver_fingerprint is None

    def test_several_candidates_are_ambiguous(self) -> None:
        resolver = CallableResolver(lambda bare: ["moonshot", "moonshotai"])
        with pytest.raises(AmbiguousModelName) as info:
            Normalizer(Rules.default(), resolver).parse("kimi-k2.5")
        assert info.value.candidates == ("moonshot", "moonshotai")

    def test_zero_candidates_is_unknown(self) -> None:
        resolver = CallableResolver(lambda bare: [])
        with pytest.raises(UnknownVendor):
            Normalizer(Rules.default(), resolver).parse("some-unknown-model")


class TestPolicies:
    def test_fold(self) -> None:
        norm = Normalizer(Rules.default().with_policy(case="fold"))
        assert norm.parse("OpenAI/GPT-6-Astra") == norm.parse("openai/gpt-6-astra")

    def test_reject_mixed_case(self) -> None:
        norm = Normalizer(Rules.default().with_policy(case="reject"))
        with pytest.raises(MalformedName):
            norm.parse("OpenAI/GPT-6-Astra")
        assert norm.parse("openai/gpt-6-astra").vendor == "openai"

    def test_leading_slash_reject(self) -> None:
        norm = Normalizer(Rules.default().with_policy(leading_slash="reject"))
        with pytest.raises(MalformedName):
            norm.parse("/openrouter/openai/gpt-6-astra")
        assert Normalizer(Rules.default()).parse("/openrouter/openai/gpt-6-astra").route == "openrouter"

    def test_bare_name_provider_as_vendor(self) -> None:
        norm = Normalizer(Rules.default().with_policy(bare_name="provider_as_vendor"))
        assert norm.parse_pair("openrouter", "agent-model").canonical == "openrouter/agent-model"
        with pytest.raises(UnknownVendor):
            norm.parse("agent-model")

    def test_bare_name_error_ignores_the_resolver(self) -> None:
        resolver = CallableResolver(lambda bare: ["openai"])
        norm = Normalizer(Rules.default().with_policy(bare_name="error"), resolver)
        with pytest.raises(UnknownVendor):
            norm.parse("gpt-4o")

    def test_unknown_provider_error(self) -> None:
        norm = Normalizer(Rules.default().with_policy(unknown_provider="error"))
        with pytest.raises(UnknownRoute):
            norm.parse_pair("xai", "x-ai/grok-4")

    def test_implied_vendor_mismatch_name_wins(self) -> None:
        norm = Normalizer(Rules.default().with_policy(implied_vendor_mismatch="name_wins"))
        name = norm.parse_pair("openai", "anthropic/claude-opus-4-6")
        assert (name.vendor, name.route) == ("anthropic", "openai")


class TestOverrideTables:
    def test_bare_names_patterns_in_file_order(self) -> None:
        rules = Rules.from_dict(
            {"bare_names": {"*llama*": "meta-llama", "deepseek*": "deepseek", "gpt*": "openai"}},
            base=Rules.default(),
        )
        norm = Normalizer(rules)
        assert norm.parse("deepseek-r1-distill-llama-70b").vendor == "meta-llama"
        assert norm.parse("gpt-4o").canonical == "openai/gpt-4o"
        assert norm.parse_pair("openrouter", "gpt-4o").routed == "openrouter/openai/gpt-4o"

    def test_vendor_aliases_fold_the_vendor_only(self) -> None:
        rules = Rules.from_dict(
            {"vendor_aliases": {"xai": "x-ai", "moonshot": "moonshotai"}}, base=Rules.default()
        )
        norm = Normalizer(rules)
        assert norm.parse("xai/grok-4.6") == norm.parse("x-ai/grok-4.6")
        assert norm.parse("moonshot/kimi-k2.5").canonical == "moonshotai/kimi-k2.5"

    def test_determinism(self) -> None:
        norm = Normalizer(Rules.default())
        assert norm.parse("openrouter/openai/gpt-6-astra") == norm.parse("openrouter/openai/gpt-6-astra")
        assert (
            norm.parse("openrouter/openai/gpt-6-astra").to_dict()
            == norm.parse("openrouter/openai/gpt-6-astra").to_dict()
        )
