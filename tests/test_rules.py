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
    default_normalizer,
    parse,
    reset_default,
)


def doc() -> dict:
    return copy.deepcopy(dict(Rules.default().document))


def test_default_loads_and_reports_versions() -> None:
    rules = Rules.default()
    assert rules.schema_version == 1
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
            {"routes": {"nebius": {}}},
            {"vendor_aliases": {"xai": "x-ai"}},
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
        path.write_text('version = "site-1"\n[vendor_aliases]\nxai = "x-ai"\n', "utf-8")
        rules = Rules.load(path)
        assert rules.version == "site-1" and rules.vendor_aliases == {"xai": "x-ai"}
        assert Normalizer(rules).parse("xai/grok-4.6").vendor == "x-ai"


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
        assert json.loads(json.dumps(dict(Rules.default().document)))["schema_version"] == 1
