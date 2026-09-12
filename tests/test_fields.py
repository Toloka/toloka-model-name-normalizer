"""The descriptive fields: the hard cases of the design, suffixes, dates, the sort key, strict mode."""

from __future__ import annotations

import pytest

from toloka_model_name_normalizer import ModelName, Normalizer, Rules, UnresolvedFields, generation_key

norm = Normalizer(Rules.default())


def fields(text: str) -> dict[str, str | None]:
    name = norm.parse(text)
    return {key: value for key, value in name.fields().items() if key != "vendor"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("openai/gpt-6-astra", {"family": "gpt", "generation": "6", "variant": "astra"}),
        (
            "deepseek/deepseek-v4-flash-0731",
            {"family": "deepseek", "generation": "v4", "tier": "flash", "snapshot": "0731"},
        ),
        ("openai/gpt-4o", {"family": "gpt", "generation": "4o"}),
        ("openai/gpt-oss-120b", {"family": "gpt", "generation": None, "variant": "oss", "size": "120b"}),
        ("x-ai/grok-4.20", {"family": "grok", "generation": "4.20"}),
        ("amazon/nova-2-lite-v1", {"family": "nova", "generation": "2", "tier": "lite", "api_version": "v1"}),
        ("nova/nova-lite-v2", {"family": "nova", "generation": "v2", "tier": "lite", "api_version": None}),
        (
            "anthropic/claude-3-5-sonnet-20241022",
            {"family": "claude", "generation": "3.5", "tier": "sonnet", "snapshot": "20241022"},
        ),
        ("anthropic/claude-sonnet-4-6", {"generation": "4.6", "tier": "sonnet"}),
        ("qwen/qwen3.8-max", {"family": "qwen", "generation": "3.8", "tier": "max"}),
        ("moonshotai/kimi-k2.7-code", {"family": "kimi", "generation": "k2.7", "capability": "code"}),
        ("tencent/hy3", {"family": "hy", "generation": "3"}),
        ("minimax/minimax-m3", {"family": "minimax", "generation": "m3"}),
        ("cohere/command-a-plus", {"family": "command", "generation": "a", "tier": "plus"}),
        (
            "cohere/command-r7b-12-2024",
            {"family": "command", "generation": "r", "size": "7b", "snapshot": "2024-12"},
        ),
        ("openai/gpt-5.6-terra", {"variant": "terra", "tier": None}),
        ("openai/gpt-5.4-pro", {"variant": None, "tier": "pro"}),
        ("meta/muse-spark-1.1", {"family": "muse", "generation": "1.1", "variant": "spark"}),
        ("google/gemma-4-31b-it", {"family": "gemma", "generation": "4", "size": "31b", "capability": "it"}),
        ("openai/o3-mini-high", {"family": "o", "generation": "3", "tier": "mini", "variant": "high"}),
        ("google/gemini-2.0-flash-001", {"generation": "2.0", "tier": "flash", "api_version": "001"}),
        (
            "google/gemini-2.5-flash-lite-preview-09-2025",
            {"tier": "flash-lite", "stage": "preview", "snapshot": "2025-09"},
        ),
        ("openai/gpt-4o-2024-05-13", {"generation": "4o", "snapshot": "2024-05-13"}),
        ("qwen/qwen3.5-flash-02-23", {"generation": "3.5", "tier": "flash", "snapshot": "02-23"}),
        ("mistralai/mistral-large-2512", {"family": "mistral", "tier": "large", "snapshot": "2512"}),
        ("nvidia/nemotron-3-ultra-550b-a55b", {"generation": "3", "tier": "ultra", "size": "550b-a55b"}),
        ("mistralai/mixtral-8x22b-instruct", {"size": "8x22b", "capability": "instruct"}),
        (
            "x-ai/grok-4.1-fast-non-reasoning",
            {"tier": "fast", "variant": "non-reasoning", "capability": None},
        ),
        ("openai/o3-deep-research", {"family": "o", "generation": "3", "capability": "deep-research"}),
        (
            "azure_ai/cohere-command-a-plus-05-2026",
            {"family": "command", "generation": "a", "tier": "plus", "variant": None, "snapshot": "2026-05"},
        ),
    ],
)
def test_hard_cases(text: str, expected: dict[str, str | None]) -> None:
    actual = fields(text)
    assert {key: actual[key] for key in expected} == expected


class TestSuffixes:
    def test_thinking_is_a_capability_and_stays_in_the_identity(self) -> None:
        name = norm.parse("anthropic/claude-3.7-sonnet:thinking")
        assert name.model == "claude-3.7-sonnet:thinking"
        assert (name.capability, name.suffix, name.transport_suffix, name.tier) == (
            "thinking",
            "thinking",
            False,
            "sonnet",
        )

    def test_transport_suffix_is_flagged(self) -> None:
        name = norm.parse("x-ai/grok-4:nitro")
        assert (name.variant, name.transport_suffix) == ("nitro", True)

    def test_beta_suffix_is_not_a_stage(self) -> None:
        name = norm.parse("anthropic/claude-3.5-sonnet:beta")
        assert (name.stage, name.variant) == (None, "beta")

    def test_unknown_suffix_is_unresolved(self) -> None:
        name = norm.parse("x-ai/grok-4:turbocharged")
        assert name.variant == "turbocharged" and "unknown-suffix:turbocharged" in name.unresolved


class TestDates:
    def test_impossible_dates_are_unresolved(self) -> None:
        assert norm.parse("openai/gpt-6-astra-2026-13-45").snapshot is None
        assert "bad-date:2026-13-45" in norm.parse("openai/gpt-6-astra-2026-13-45").unresolved
        assert "bad-date:0000" in norm.parse("openai/gpt-6-astra-0000").unresolved
        assert "bad-date:20261345" in norm.parse("openai/gpt-6-astra-20261345").unresolved

    def test_second_snapshot_and_second_api_version_are_unresolved(self) -> None:
        assert "second-snapshot:0528" in norm.parse("deepseek/deepseek-r1-0324-0528").unresolved
        assert "second-api-version:002" in norm.parse("google/gemini-2.0-001-002").unresolved

    def test_second_version_token_goes_to_variant(self) -> None:
        name = norm.parse("nousresearch/hermes-3-llama-3.1-405b")
        assert (name.generation, name.variant, name.size) == ("3", "llama-3.1", "405b")
        assert name.unresolved == ("second-version:3.1",)


class TestGenerationKey:
    def test_ordering(self) -> None:
        values = ["4.20", "4.5", "k2.7", "k3", "a", "r", "01", "1", "4o", "4", None]
        assert sorted(values, key=generation_key) == [
            None,
            "a",
            "r",
            "01",
            "1",
            "k2.7",
            "k3",
            "4",
            "4o",
            "4.5",
            "4.20",
        ]

    def test_raw_token_breaks_the_tie(self) -> None:
        assert generation_key("01") != generation_key("1")
        assert generation_key("01")[:4] == generation_key("1")[:4]

    def test_on_the_value(self) -> None:
        assert norm.parse("x-ai/grok-4.20").generation_key() > norm.parse("x-ai/grok-4.5").generation_key()


class TestStrict:
    def test_strict_raises_on_unresolved(self) -> None:
        strict = Normalizer(Rules.default().with_policy(fields="strict"))
        with pytest.raises(UnresolvedFields) as info:
            strict.parse("nousresearch/hermes-3-llama-3.1-405b")
        assert info.value.unresolved == ("second-version:3.1",)
        assert strict.parse("openai/gpt-6-astra").generation == "6"


class TestLookupFields:
    def test_empty_string_clears_a_computed_value(self) -> None:
        name = norm.parse("azure_ai/cohere-command-a-plus-05-2026")
        assert name.variant is None and name.provenance.lookup is True

    def test_lookup_fields_apply_to_every_carrier(self) -> None:
        pair = norm.parse_pair("azure_ai", "cohere-command-a-plus-05-2026")
        assert (pair.family, pair.generation) == ("command", "a")


def test_value_is_frozen() -> None:
    name = norm.parse("openai/gpt-6-astra")
    with pytest.raises(AttributeError):
        name.vendor = "x"  # type: ignore[misc]
    assert isinstance(name, ModelName)
