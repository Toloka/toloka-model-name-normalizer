"""Rules: loading, schema validation, merge semantics, fingerprints, the environment variable."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer import (
    RULES_ENV_VAR,
    Normalizer,
    Rules,
    RulesError,
    UnknownVendor,
    default_normalizer,
    parse,
    reset_default,
)


def doc() -> dict:
    return copy.deepcopy(dict(Rules.default().document))


def test_default_loads_and_reports_versions() -> None:
    rules = Rules.default()
    assert rules.schema_version == 2
    assert rules.version
    assert rules.identity_fingerprint.startswith("sha256:") and rules.fields_fingerprint.startswith("sha256:")
    assert rules.identity_fingerprint != rules.fields_fingerprint


class TestFingerprints:
    def test_stable_across_load_paths(self, tmp_path: Path) -> None:
        default = Rules.default()
        again = Rules.from_dict(doc())
        assert (again.identity_fingerprint, again.fields_fingerprint) == (
            default.identity_fingerprint,
            default.fields_fingerprint,
        )
        empty_override = tmp_path / "empty.toml"
        empty_override.write_text("", "utf-8")
        loaded = Rules.load(empty_override)
        assert (loaded.identity_fingerprint, loaded.fields_fingerprint) == (
            default.identity_fingerprint,
            default.fields_fingerprint,
        )

    def test_why_and_version_do_not_participate(self) -> None:
        default = Rules.default()
        changed = doc()
        changed["version"] = "9999.01.01"
        changed["lookup"]["openrouter/auto"]["why"] = "edited"
        rules = Rules.from_dict(changed)
        assert (rules.identity_fingerprint, rules.fields_fingerprint) == (
            default.identity_fingerprint,
            default.fields_fingerprint,
        )

    def test_vocabulary_changes_only_the_fields_fingerprint(self) -> None:
        default = Rules.default()
        rules = Rules.from_dict({"fields": {"vocab": {"tier": ["giga"]}}}, base=default)
        assert rules.identity_fingerprint == default.identity_fingerprint
        assert rules.fields_fingerprint != default.fields_fingerprint

    def test_route_alias_and_separator_change_only_the_identity_fingerprint(self) -> None:
        default = Rules.default()
        for override in (
            {"routes": {"acmehost": {}}},
            {"vendor_aliases": {"acme-ai": "acme"}},
            {"fields": {"separators": "[-/]+"}},
        ):
            rules = Rules.from_dict(override, base=default)
            assert rules.identity_fingerprint != default.identity_fingerprint, override
            assert rules.fields_fingerprint == default.fields_fingerprint, override

    def test_lookup_identity_and_field_halves_land_on_their_side(self) -> None:
        default = Rules.default()
        identity = Rules.from_dict(
            {"lookup": {"openrouter/x": {"vendor": "openrouter", "model": "x", "why": "t"}}}, base=default
        )
        assert identity.identity_fingerprint != default.identity_fingerprint
        assert identity.fields_fingerprint == default.fields_fingerprint
        fields = Rules.from_dict({"lookup": {"acme/y": {"tier": "pro", "why": "t"}}}, base=default)
        assert fields.identity_fingerprint == default.identity_fingerprint
        assert fields.fields_fingerprint != default.fields_fingerprint


class TestSchema:
    @pytest.mark.parametrize(
        "override, message",
        [
            ({"catalogue": {}}, "unknown key"),
            ({"schema_version": 7}, "unsupported schema_version"),
            ({"lookup": {"a/b": {"vendor": "a", "model": "b"}}}, "why"),
            ({"lookup": {"a/b": {"vendor": "a", "why": "t"}}}, "vendor and model together"),
            ({"bare_names": {"gpt-4o": "openai"}}, "no wildcard"),
            ({"routes": {"nebius": {"implies_vendor": "platform_openai"}}}, "peelable route"),
            ({"policy": {"case": "shout"}}, "policy.case"),
            ({"render": {"unsafe": "["}}, "invalid regex"),
            ({"fields": {"shapes": {"size": ["("]}}}, "invalid regex"),
            ({"fields": {"suffixes": {"x": {"field": "stage"}}}}, "suffixes.x.field"),
            ({"fields": {"vendor": {"acme": {"four_digit_snapshot": "DDMM"}}}}, "MMDD or YYMM"),
            ({"fields": {"dates": {"odd": {"tokens": ["MM"]}}}}, "tokens"),
            ({"vendor_aliases": {"openrouter": "x"}}, "also a route"),
            ({"policy": {"colour": "blue"}}, "unknown key"),
        ],
    )
    def test_rejects(self, override: dict, message: str) -> None:
        with pytest.raises(RulesError, match=message):
            Rules.from_dict(override, base=Rules.default())

    def test_extends_false_requires_a_complete_document(self) -> None:
        with pytest.raises(RulesError):
            Rules.from_dict({"extends": False, "schema_version": 1, "version": "x"}, base=Rules.default())
        full = doc()
        full["extends"] = False
        rules = Rules.from_dict(full, base=Rules.default())
        assert rules.version == Rules.default().version

    def test_direct_vendor_route_may_imply_itself(self) -> None:
        rules = Rules.from_dict({"routes": {"cohere": {"implies_vendor": "cohere"}}}, base=Rules.default())
        assert rules.routes["cohere"].peelable is False


class TestMerge:
    def test_route_replaces_and_remove_routes_drops(self) -> None:
        rules = Rules.from_dict(
            {"remove_routes": ["mock"], "routes": {"gemini": {"implies_vendor": "gemini"}}},
            base=Rules.default(),
        )
        assert "mock" not in rules.routes
        assert rules.routes["gemini"].implies_vendor == "gemini" and not rules.routes["gemini"].peelable

    def test_vocabulary_extends_without_duplicates(self) -> None:
        rules = Rules.from_dict({"fields": {"vocab": {"tier": ["pro", "giga"]}}}, base=Rules.default())
        assert rules.fields.tier.count("pro") == 1 and "giga" in rules.fields.tier

    def test_vendor_switch_merges_per_key(self) -> None:
        rules = Rules.from_dict(
            {"fields": {"vendor": {"mistralai": {"four_digit_snapshot": "MMDD"}}}}, base=Rules.default()
        )
        switches = rules.fields.switches("mistralai")
        assert switches.dash_minor_version is True and switches.four_digit_snapshot == "MMDD"

    def test_scalars_overwrite(self) -> None:
        rules = Rules.from_dict(
            {"version": "2027.01.01", "policy": {"case": "fold"}, "render": {"route_in_slug": True}},
            base=Rules.default(),
        )
        assert (rules.version, rules.policy.case, rules.render.route_in_slug) == ("2027.01.01", "fold", True)

    def test_load_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.toml"
        path.write_text('version = "site-1"\n[vendor_aliases]\nacme-ai = "acme"\n', "utf-8")
        rules = Rules.load(path)
        assert rules.version == "site-1" and rules.vendor_aliases["acme-ai"] == "acme"
        assert dict(rules.vendor_aliases).items() >= dict(Rules.default().vendor_aliases).items()
        assert Normalizer(rules).parse("acme-ai/thing-1").vendor == "acme"


class TestEnvironment:
    def test_from_env_reads_the_named_file_lazily(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "rules.toml"
        path.write_text('version = "from-env"\n', "utf-8")
        monkeypatch.setenv(RULES_ENV_VAR, str(path))
        reset_default()
        try:
            assert Rules.from_env().version == "from-env"
            assert parse("openai/gpt-6-astra").provenance.rules_version == "from-env"
            assert default_normalizer().rules.version == "from-env"
            monkeypatch.delenv(RULES_ENV_VAR)
            assert default_normalizer().rules.version == "from-env", "built once, not re-read"
        finally:
            reset_default()
        assert default_normalizer().rules.version == Rules.default().version

    def test_explicit_normalizer_never_reads_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "rules.toml"
        path.write_text('version = "from-env"\n', "utf-8")
        monkeypatch.setenv(RULES_ENV_VAR, str(path))
        assert Normalizer().rules.version == Rules.default().version

    def test_document_round_trips_through_json(self) -> None:
        assert json.loads(json.dumps(dict(Rules.default().document)))["schema_version"] == 2


# ---------------------------------------------------------------------------
# schema 2: families, family-scoped vendor aliases, canonical spelling, route keys, vendor policies
# ---------------------------------------------------------------------------

ALLOWED_VENDORS = json.loads((Path(__file__).parent / "data" / "allowed_vendors.json").read_text("utf-8"))

# A complete schema-1 document of the 2026.09.12 shape: no group, route key or policy key of schema 2.
SCHEMA_1_STANDALONE = {
    "extends": False,
    "schema_version": 1,
    "version": "standalone-1",
    "routes": {
        "openrouter": {},
        "openai": {"implies_vendor": "openai"},
        "anthropic": {"implies_vendor": "anthropic"},
    },
    "vendor_aliases": {},
    "bare_names": {},
    "policy": {
        "unknown_provider": "vendor",
        "bare_name": "resolver",
        "case": "preserve",
        "leading_slash": "tolerate",
        "implied_vendor_mismatch": "error",
        "fields": "best_effort",
    },
    "render": {
        "route_sep": "__",
        "unsafe": "[^A-Za-z0-9._-]+",
        "unsafe_replacement": "_",
        "route_in_slug": False,
    },
    "fields": {"separators": "[-/\\s()]+"},
}

# A schema-1 override in the shape of a deployment file written for the 2026.09.12 default: vendor aliases
# for the native API spellings and lookups whose vendor is a name the new default declares as a route.
SCHEMA_1_OVERRIDE = {
    "schema_version": 1,
    "version": "site-1",
    "vendor_aliases": {"xai": "x-ai", "moonshot": "moonshotai", "mistral": "mistralai"},
    "lookup": {
        "azure_ai/cohere-command-a-plus-05-2026": {
            "vendor": "azure_ai",
            "model": "cohere-command-a-plus-05-2026",
            "why": "the deployment keeps its Azure deployment name as the identity",
        },
        "openrouter/azure_ai/cohere-command-a-plus-05-2026": {
            "vendor": "azure_ai",
            "model": "cohere-command-a-plus-05-2026",
            "why": "the same identity behind OpenRouter",
        },
    },
}


class TestSchemaVersionTwo:
    def test_default_is_schema_2_and_the_new_rule_set_version(self) -> None:
        rules = Rules.default()
        assert (rules.schema_version, rules.version) == (2, "2026.10.30.1")
        assert rules.policy.vendor_case == "fold"
        assert rules.policy.family_vendor == "infer"

    def test_a_standalone_schema_1_document_loads_and_reads_as_before(self) -> None:
        rules = Rules.from_dict(copy.deepcopy(SCHEMA_1_STANDALONE), base=Rules.default())
        assert rules.schema_version == 1
        assert (rules.policy.vendor_case, rules.policy.family_vendor) == ("preserve", "off")
        norm = Normalizer(rules)
        name = norm.parse("xai/grok-4")
        assert (name.vendor, name.model, name.route) == ("xai", "grok-4", None)
        assert name.provenance.vendor_source == "spelled"
        assert norm.parse_pair("anthropic", "claude-opus-4-6").model == "claude-opus-4-6"
        assert norm.parse("Qwen/Qwen3-32B").vendor == "Qwen"
        assert norm.parse("bedrock/us.anthropic.claude-sonnet-4-6").vendor == "bedrock"
        with pytest.raises(UnknownVendor):
            norm.parse("claude-opus-4-7")

    def test_a_schema_1_override_with_aliases_and_route_named_lookup_vendors_loads(self) -> None:
        rules = Rules.from_dict(copy.deepcopy(SCHEMA_1_OVERRIDE), base=Rules.default())
        assert rules.schema_version == 2, "the merged schema_version is the higher of base and override"
        norm = Normalizer(rules)
        for name in (
            norm.parse_canonical("azure_ai/cohere-command-a-plus-05-2026"),
            norm.parse("azure_ai/cohere-command-a-plus-05-2026"),
        ):
            assert (name.vendor, name.model, name.route) == (
                "azure_ai",
                "cohere-command-a-plus-05-2026",
                None,
            )
            assert name.provenance.vendor_source == "lookup"
        routed = norm.parse_routed("openrouter/azure_ai/cohere-command-a-plus-05-2026")
        assert (routed.vendor, routed.model, routed.route) == (
            "azure_ai",
            "cohere-command-a-plus-05-2026",
            "openrouter",
        )
        assert norm.parse("xai/grok-4.6").vendor == "x-ai"
        assert norm.parse("moonshot/kimi-k2.5").vendor == "moonshotai"

    def test_the_merged_schema_version_is_the_higher_one(self) -> None:
        assert Rules.from_dict({"schema_version": 1}, base=Rules.default()).schema_version == 2
        standalone = Rules.from_dict(copy.deepcopy(SCHEMA_1_STANDALONE), base=Rules.default())
        assert Rules.from_dict({"schema_version": 2}, base=standalone).schema_version == 2
        assert Rules.from_dict({"version": "x"}, base=standalone).schema_version == 1

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"families": {"claude": 5}}, "families.claude"),
            ({"families": {"claude": "not a segment"}}, "families.claude"),
            ({"families": {"claude": "openrouter"}}, "peelable route"),
            ({"family_vendor_aliases": {"meta/llama": 5}}, "family_vendor_aliases.meta/llama"),
            ({"family_vendor_aliases": {"meta/llama": "azure"}}, "peelable route"),
            ({"canonical": {"minor_dot": "anthropic"}}, "canonical.minor_dot"),
            ({"canonical": {"colour": []}}, "in canonical"),
            ({"routes": {"bedrock": {"strip": ["("]}}}, "invalid regex"),
            ({"routes": {"bedrock": {"strip": ["x*"]}}}, "empty string"),
            ({"routes": {"bedrock": {"strip": "x"}}}, "routes.bedrock.strip"),
            ({"routes": {"bedrock": {"dotted_vendor": "yes"}}}, "routes.bedrock.dotted_vendor"),
            ({"routes": {"bedrock": {"regions": "us"}}}, "routes.bedrock.regions"),
            ({"routes": {"bedrock": {"skip_segments": [1]}}}, "routes.bedrock.skip_segments"),
            (
                {"routes": {"fireworks_ai": {"skip_paths": "accounts/*/models"}}},
                "routes.fireworks_ai.skip_paths",
            ),
            ({"routes": {"fireworks_ai": {"decimal_p": 1}}}, "routes.fireworks_ai.decimal_p"),
            ({"routes": {"perplexity": {"also_vendor": "true"}}}, "routes.perplexity.also_vendor"),
            ({"policy": {"vendor_case": "shout"}}, "policy.vendor_case"),
            ({"policy": {"family_vendor": "guess"}}, "policy.family_vendor"),
        ],
    )
    def test_rejects_invalid_new_groups(self, override: dict, message: str) -> None:
        with pytest.raises(RulesError, match=message):
            Rules.from_dict(override, base=Rules.default())

    def test_valid_new_groups_load(self) -> None:
        rules = Rules.from_dict(
            {
                "families": {"sonar": "perplexity", "newline": "acme"},
                "family_vendor_aliases": {"acme/newline": "acme-labs"},
                "canonical": {"minor_dot": ["anthropic"]},
                "routes": {"acmehost": {"strip": ["-hosted$"], "skip_segments": ["v1"]}},
            },
            base=Rules.default(),
        )
        norm = Normalizer(rules)
        assert norm.parse("newline-1").vendor == "acme"
        assert norm.parse("acme/newline-2").vendor == "acme-labs"
        name = norm.parse("acmehost/v1/newline-3-hosted")
        assert (name.vendor, name.model, name.wire) == ("acme", "newline-3", "v1/newline-3-hosted")

    def test_families_merge_entry_by_entry(self) -> None:
        norm = Normalizer(Rules.from_dict({"families": {"newline": "acme"}}, base=Rules.default()))
        assert norm.parse("newline-1").vendor == "acme"
        assert norm.parse("claude-opus-4-7").vendor == "anthropic"

    def test_new_groups_move_only_the_identity_fingerprint(self) -> None:
        default = Rules.default()
        for override in (
            {"families": {"newline": "acme"}},
            {"family_vendor_aliases": {"meta/newline": "meta-llama"}},
            {"canonical": {"minor_dot": []}},
            {"policy": {"vendor_case": "preserve"}},
            {"policy": {"family_vendor": "off"}},
        ):
            rules = Rules.from_dict(override, base=default)
            assert rules.identity_fingerprint != default.identity_fingerprint, override
            assert rules.fields_fingerprint == default.fields_fingerprint, override

    def test_a_route_named_like_a_vendor_slug_is_also_a_vendor_or_a_direct_vendor_route(self) -> None:
        rules = Rules.default()
        slugs = set(ALLOWED_VENDORS["openrouter"]) | set(ALLOWED_VENDORS["not_on_openrouter"])
        for name in ("perplexity", "tencent", "databricks", "snowflake"):
            assert rules.routes[name].also_vendor is True, name
        offenders = sorted(
            name
            for name, route in rules.routes.items()
            if name in slugs and route.peelable and not route.also_vendor
        )
        assert offenders == []
