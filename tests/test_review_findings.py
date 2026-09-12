"""Acceptance tests of the code review of 2026-09-12 (docs/review-claude-fable-5.1.md), one block each."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer import (
    CallableResolver,
    MalformedName,
    ModelNameError,
    Normalizer,
    Rules,
    RulesError,
    UnknownVendor,
)
from toloka_model_name_normalizer.model import DESCRIPTIVE_FIELDS

norm = Normalizer(Rules.default())
GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text("utf-8"))


class TestF1ClosedSchemaForOverrides:
    @pytest.mark.parametrize(
        "override",
        [
            {"fields": {"vocabulary": {"tier": ["giga"]}}},
            {"fields": {"vocab": {"tiers": ["giga"]}}},
            {"fields": {"suffix": {"x": {"field": "variant"}}}},
            {"fields": {"vendors": {"acme": {}}}},
            {"fields": {"suffixes": {"x": {"field": "variant", "colour": "red"}}}},
            {"fields": {"dates": {"iso": {"tokens": ["YYYY", "MM"], "strict": True}}}},
            {"routes": {"nebius": {"vendor": "nebius"}}},
            {"lookup": {"a/b": {"vendor": "a", "model": "b", "why": "t", "price": "1"}}},
        ],
    )
    def test_unknown_nested_keys_are_rejected(self, override: dict) -> None:
        with pytest.raises(RulesError, match="unknown key"):
            Rules.from_dict(override, base=Rules.default())

    @pytest.mark.parametrize(
        "override",
        [
            {"fields": "x"},
            {"fields": {"vocab": "x"}},
            {"fields": {"vocab": {"tier": "giga"}}},
            {"remove_routes": "mock"},
            {"remove_routes": 5},
            {"routes": "openrouter"},
            {"policy": "fold"},
            {"lookup": "x"},
            {"fields": {"vendor": {"acme": "x"}}},
            {"fields": {"vocab": {"compounds": "x"}}},
            {"fields": {"shapes": {"size": "^x$"}}},
            {"fields": {"vocab": {"tier": ["Giga"]}}},
            {"schema_version": "1"},
        ],
    )
    def test_wrong_types_are_rejected(self, override: dict) -> None:
        with pytest.raises(RulesError):
            Rules.from_dict(override, base=Rules.default())


class TestF2CompoundsOnTokenBoundaries:
    def test_substring_is_not_a_compound(self) -> None:
        name = norm.parse("openai/o3-deep-researcher")
        assert (name.capability, name.variant) == (None, "deep-researcher")
        for text in ("openai/o3-deep-researcher", "x-ai/grok-5-multi-agents", "x-ai/grok-encode-faster"):
            assert "\x00" not in json.dumps(norm.parse(text).to_dict())
        assert norm.parse("x-ai/grok-5-multi-agents").variant == "multi-agents"

    def test_whole_token_compounds_still_match(self) -> None:
        assert norm.parse("openai/o3-deep-research").capability == "deep-research"
        assert norm.parse("x-ai/grok-code-fast-1").capability == "code-fast"
        assert norm.parse("x-ai/grok-4.1-fast-non-reasoning").variant == "non-reasoning"


class TestF3PeelableRouteNeverAVendor:
    @pytest.mark.parametrize(
        "provider, name",
        [
            ("openrouter", "platform_openai/gpt-6-astra"),
            ("openrouter", "litellm/gpt-6-astra"),
            ("xai", "openrouter/x"),
        ],
    )
    def test_pair_refuses(self, provider: str, name: str) -> None:
        with pytest.raises(MalformedName):
            norm.parse_pair(provider, name)

    def test_resolver_candidate_refuses(self) -> None:
        for candidate in ("openrouter/gpt-x", "platform_openai/gpt"):
            resolver = CallableResolver(lambda bare, c=candidate: [c])
            with pytest.raises(MalformedName):
                Normalizer(Rules.default(), resolver).parse("gpt-x")

    def test_direct_vendor_route_names_stay_valid_vendors(self) -> None:
        assert norm.parse_pair("openrouter", "openai/gpt-6-astra").vendor == "openai"
        assert norm.parse_pair("openrouter", "meta/muse-spark-1.1").vendor == "meta"


class TestF4FingerprintCanonicalForm:
    def test_bare_names_order_is_part_of_the_identity_fingerprint(self) -> None:
        base = Rules.default()
        a = Rules.from_dict({"bare_names": {"*llama*": "meta-llama", "deepseek*": "deepseek"}}, base=base)
        b = Rules.from_dict({"bare_names": {"deepseek*": "deepseek", "*llama*": "meta-llama"}}, base=base)
        assert a.identity_fingerprint != b.identity_fingerprint
        assert Normalizer(a).parse("deepseek-r1-distill-llama-70b").vendor == "meta-llama"
        assert Normalizer(b).parse("deepseek-r1-distill-llama-70b").vendor == "deepseek"

    def test_lookup_key_case_does_not_change_the_fingerprint(self) -> None:
        base = Rules.default()
        lower = Rules.from_dict(
            {"lookup": {"acme/x": {"vendor": "acme", "model": "x", "why": "t"}}}, base=base
        )
        upper = Rules.from_dict(
            {"lookup": {"ACME/X": {"vendor": "acme", "model": "x", "why": "t"}}}, base=base
        )
        assert lower.identity_fingerprint == upper.identity_fingerprint


class TestF5RulesErrorsOnly:
    def test_extends_false_with_partial_fields(self) -> None:
        doc = json.loads(json.dumps(dict(Rules.default().document)))
        doc["extends"] = False
        doc["fields"] = {"separators": "[-/]+"}
        rules = Rules.from_dict(doc, base=Rules.default())
        assert rules.fields.tier == ()
        assert Normalizer(rules).parse("openai/gpt-6-astra").family == "gpt"

    def test_malformed_toml_is_a_rules_error(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.toml"
        path.write_text("this is = not toml = [\n", "utf-8")
        with pytest.raises(RulesError, match="not valid TOML"):
            Rules.load(path)

    def test_cli_exits_2_on_a_bad_rules_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from toloka_model_name_normalizer.__main__ import main

        for content in ("this is = not toml = [\n", 'routes = "x"\n'):
            path = tmp_path / "rules.toml"
            path.write_text(content, "utf-8")
            assert main(["--rules", str(path), "parse", "openai/gpt-6-astra"]) == 2
            assert "rules error" in capsys.readouterr().err


class TestF6RouteNamesFolded:
    def test_override_with_uppercase_route_name_applies(self) -> None:
        rules = Rules.from_dict({"routes": {"OpenRouter": {"nestable": True}}}, base=Rules.default())
        assert [name for name in rules.routes if name.lower() == "openrouter"] == ["openrouter"]
        assert rules.routes["openrouter"].nestable is True

    def test_implies_vendor_is_folded(self) -> None:
        rules = Rules.from_dict({"routes": {"acme": {"implies_vendor": "Acme"}}}, base=Rules.default())
        assert rules.routes["acme"].peelable is False

    def test_case_collision_inside_one_document_is_rejected(self) -> None:
        with pytest.raises(RulesError, match="collide"):
            Rules.from_dict({"routes": {"nebius": {}, "Nebius": {}}}, base=Rules.default())


class TestF7LookupOnTwoSegmentRemainder:
    def test_every_carrier_agrees(self) -> None:
        rules = Rules.from_dict(
            {"lookup": {"openrouter/acme/thing": {"vendor": "acme", "model": "thing-x", "why": "t"}}},
            base=Rules.default(),
        )
        n = Normalizer(rules)
        results = [
            n.parse("openrouter/acme/thing"),
            n.parse_pair("openrouter", "acme/thing"),
            n.parse_routed("openrouter/acme/thing"),
            n.parse_routed("litellm/openrouter/acme/thing"),
        ]
        assert {r.canonical for r in results} == {"acme/thing-x"}
        assert all(r.provenance.lookup for r in results)
        assert results[3].routes == ("litellm", "openrouter")


class TestF8FamilyIsNeverEmpty:
    @pytest.mark.parametrize(
        "text", ["openai/-", "openai/(", "openai/:thinking", "openai/---", "openai/gpt:a:b"]
    )
    def test_names_without_a_letter_or_digit_before_the_suffix_are_malformed(self, text: str) -> None:
        with pytest.raises(MalformedName):
            norm.parse(text)

    def test_digit_first_names_take_the_first_token_as_family(self) -> None:
        assert norm.parse("openai/2024").family == "2024"

    def test_no_golden_input_yields_an_empty_family_tag(self) -> None:
        for row in GOLDEN["rows"]:
            if not row.get("expected") or row["input_kind"] not in ("pair", "text"):
                continue
            name = (
                norm.parse_pair(row["provider"], row["input"])
                if row["input_kind"] == "pair"
                else norm.parse(row["input"])
            )
            assert name.family, row["input"]
            assert name.tags()[2] != "model_family:"

    def test_lookup_cannot_clear_family(self) -> None:
        with pytest.raises(RulesError, match="family"):
            Rules.from_dict({"lookup": {"a/b": {"family": "", "why": "t"}}}, base=Rules.default())


class TestF10DashMinorVersionStopsAtASnapshot:
    def test_generation_and_snapshot_are_separated(self) -> None:
        name = norm.parse("anthropic/claude-3-20240229")
        assert (name.generation, name.snapshot) == ("3", "20240229")
        name = norm.parse("mistralai/mistral-small-3-2512")
        assert (name.generation, name.snapshot) == ("3", "2512")

    def test_the_real_dash_minor_versions_still_merge(self) -> None:
        assert norm.parse("anthropic/claude-3-5-sonnet-20241022").generation == "3.5"
        assert norm.parse("mistralai/mistral-medium-3-5").generation == "3.5"


class TestF11LookupValuesValidated:
    @pytest.mark.parametrize(
        "spec",
        [
            {"vendor": "acme", "model": "a/b", "why": "t"},
            {"vendor": "~x", "model": "y", "why": "t"},
            {"vendor": "acme", "model": "", "why": "t"},
            {"vendor": "acme", "model": "openrouter__x", "why": "t"},
        ],
    )
    def test_bad_identity_values(self, spec: dict) -> None:
        with pytest.raises(RulesError):
            Rules.from_dict({"lookup": {"acme/thing": spec}}, base=Rules.default())

    def test_bad_keys_and_values(self) -> None:
        with pytest.raises(RulesError):
            Rules.from_dict({"lookup": {"": {"tier": "pro", "why": "t"}}}, base=Rules.default())
        with pytest.raises(RulesError):
            Rules.from_dict({"lookup": {"a/b": {"tier": 5, "why": "t"}}}, base=Rules.default())
        with pytest.raises(RulesError, match="canonical"):
            Rules.from_dict(
                {"lookup": {"openrouter/openai/gpt-6-astra": {"tier": "giga", "why": "t"}}},
                base=Rules.default(),
            )

    def test_lookup_identity_follows_the_case_policy(self) -> None:
        spec = {"vendor": "X", "model": "Y-z", "why": "t"}
        folded = Normalizer(
            Rules.from_dict({"policy": {"case": "fold"}, "lookup": {"x/y": spec}}, base=Rules.default())
        )
        assert folded.parse("x/y").canonical == "x/y-z"
        rejecting = Normalizer(
            Rules.from_dict({"policy": {"case": "reject"}, "lookup": {"x/y": spec}}, base=Rules.default())
        )
        with pytest.raises(MalformedName):
            rejecting.parse("x/y")


class TestF12DegenerateRulesRejectedOrHarmless:
    @pytest.mark.parametrize(
        "override",
        [
            {"fields": {"shapes": {"snapshot8": ["^\\d{6}$"]}}},
            {"fields": {"shapes": {"snapshot4": ["^\\d{3}$"]}}},
            {"fields": {"separators": "x*"}},
            {"render": {"unsafe": ""}},
        ],
    )
    def test_rejected_at_load(self, override: dict) -> None:
        with pytest.raises(RulesError):
            Rules.from_dict(override, base=Rules.default())


class TestF13Edges:
    def test_route_without_model_is_malformed(self) -> None:
        with pytest.raises(MalformedName):
            norm.parse_routed("openrouter")

    def test_repeated_non_nestable_route_is_named_as_such(self) -> None:
        with pytest.raises(MalformedName, match="repeated route"):
            norm.parse_routed("openrouter/openrouter/openai/gpt-6-astra")

    def test_provider_never_tolerates_a_leading_slash(self) -> None:
        with pytest.raises(MalformedName):
            norm.parse_pair("/openrouter", "openai/gpt-6-astra")

    def test_outer_whitespace_is_an_error_not_trimmed(self) -> None:
        with pytest.raises(MalformedName):
            norm.parse_pair("openrouter", " openai/gpt-6-astra ")
        with pytest.raises(MalformedName):
            norm.parse("openai/gpt-6-astra ")
        with pytest.raises(MalformedName):
            norm.parse(" openai/gpt-6-astra")

    def test_second_colon_is_malformed(self) -> None:
        with pytest.raises(MalformedName):
            norm.parse("x-ai/grok-4:nitro:free")

    def test_resolver_without_fingerprint_attribute(self) -> None:
        class Bare:
            def candidates(self, bare_name: str) -> list[str]:
                return ["openai"]

        name = Normalizer(Rules.default(), Bare()).parse("gpt-4o")  # type: ignore[arg-type]
        assert name.canonical == "openai/gpt-4o" and name.provenance.resolver_fingerprint is None

    def test_callable_resolver_rejects_a_string_result(self) -> None:
        with pytest.raises(TypeError):
            Normalizer(Rules.default(), CallableResolver(lambda b: "meta/muse")).parse("muse")  # type: ignore[arg-type,return-value]

    def test_error_messages_carry_the_grammar_and_the_routes(self) -> None:
        with pytest.raises(MalformedName, match="segment grammar"):
            norm.parse("öpenai/gpt")
        with pytest.raises(ModelNameError, match="openrouter"):
            norm.parse_routed("deepseek/deepseek-v4")


class TestF14PromisedTests:
    @pytest.mark.parametrize(
        "policy, value",
        [
            ("case", "fold"),
            ("case", "reject"),
            ("bare_name", "error"),
            ("bare_name", "provider_as_vendor"),
            ("unknown_provider", "error"),
            ("leading_slash", "reject"),
            ("implied_vendor_mismatch", "name_wins"),
            ("fields", "strict"),
        ],
    )
    def test_golden_rows_under_every_policy_flip_fail_only_with_model_name_errors(
        self, policy: str, value: str
    ) -> None:
        n = Normalizer(Rules.default().with_policy(**{policy: value}))
        for row in GOLDEN["rows"]:
            if row["input_kind"] not in ("pair", "text", "canonical", "routed", "slug"):
                continue
            try:
                if row["input_kind"] == "pair":
                    name = n.parse_pair(row["provider"], row["input"])
                elif row["input_kind"] == "canonical":
                    name = n.parse_canonical(row["input"])
                elif row["input_kind"] == "routed":
                    name = n.parse_routed(row["input"])
                else:
                    name = n.parse(row["input"])
            except ModelNameError:
                continue
            assert name.vendor and name.model and name.family
            if row["expected"] and value != "fold":
                assert name.same_model(norm.parse_canonical(row["expected"]["canonical"]))

    def test_every_slug_is_rejected_by_every_entry_point(self) -> None:
        slugs = [row["input"] for row in GOLDEN["rows"] if row["input_kind"] == "slug"]
        assert slugs
        for slug in slugs:
            with pytest.raises(MalformedName):
                norm.parse(slug)
            with pytest.raises(MalformedName):
                norm.parse_pair("openrouter", slug)
            with pytest.raises((MalformedName, UnknownVendor)):
                norm.parse_canonical(f"acme/{slug}")

    def test_coverage_fill_and_error_counters_match_the_fixture(self) -> None:
        coverage = json.loads((Path(__file__).parent / "data" / "coverage.json").read_text("utf-8"))
        fill: dict[str, dict[str, int]] = {}
        seen: set[str] = set()
        errors: dict[str, int] = {}
        for row in GOLDEN["rows"]:
            if row.get("error"):
                key = f"error_{row['input_kind']}_{row['error']['type']}"
                errors[key] = errors.get(key, 0) + 1
            expected = row.get("expected")
            if not expected or row["classes"][0].startswith("review_") or expected["canonical"] in seen:
                continue
            seen.add(expected["canonical"])
            tier = fill.setdefault(row["tier"], {})
            tier["family"] = tier.get("family", 0) + 1
            for field in DESCRIPTIVE_FIELDS[1:]:
                if expected["fields"][field]:
                    tier[field] = tier.get(field, 0) + 1
            if expected["fields"]["generation"] is None:
                tier["no_generation"] = tier.get("no_generation", 0) + 1
        assert fill == coverage["fill"]
        assert errors == coverage["errors"]

    def test_second_snapshot_is_reported_not_kept(self) -> None:
        name = norm.parse("deepseek/deepseek-r1-0324-0528")
        assert name.snapshot == "0324" and name.variant is None and "second-snapshot:0528" in name.unresolved
