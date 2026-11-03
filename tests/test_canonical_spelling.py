"""The canonical model spelling: one identity per model, whatever the provider's notation and deployment.

- Version notation: the vendors that write a minor version with a dot on OpenRouter (``canonical.minor_dot``:
  anthropic, z-ai, ai21) get the dot also when the name spells a dash (``claude-opus-4-6`` is
  ``claude-opus-4.6``); Fireworks' ``5p1`` is ``5.1``. An eight-digit snapshot is never a minor version, and
  Mistral keeps its dashes.
- Deployment detail is no identity: a Bedrock region prefix, a LiteLLM mode, region or commitment path
  segment, Vertex ``@default`` and ``-maas``, a Bedrock ``:N`` suffix and the Databricks ``databricks-``
  prefix leave the model.
- ``wire`` and ``routed`` keep what was sent, so ``same_wire`` tells two spellings of one identity apart.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from toloka_model_name_normalizer import MalformedName, ModelName, Normalizer, Provenance, Rules
from toloka_model_name_normalizer.__main__ import main

NORM = Normalizer(Rules.default())
ENGINE = Normalizer(Rules.from_dict({"render": {"route_in_slug": True}}, base=Rules.default()))


class TestVersionDot:
    @pytest.mark.parametrize(
        "dash, dot",
        [
            ("anthropic/claude-opus-4-6", "anthropic/claude-opus-4.6"),
            ("anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),
            ("z-ai/glm-4-7", "z-ai/glm-4.7"),
            ("ai21/jamba-1-5-large", "ai21/jamba-1.5-large"),
        ],
    )
    def test_dash_and_dot_spelling_are_one_canonical_identity(self, dash: str, dot: str) -> None:
        dashed, dotted = NORM.parse_canonical(dash), NORM.parse_canonical(dot)
        assert dashed == dotted
        assert dashed.canonical == dot
        assert dashed.same_model(dotted)
        assert not dashed.same_wire(dotted)
        assert (dashed.wire, dashed.routed) == (dash, dash)
        assert (dotted.wire, dotted.routed) == (dot, dot)
        assert dashed.route is None and dashed.routes == ()

    def test_every_entry_point_reads_the_dash_as_the_dot(self) -> None:
        assert NORM.parse("anthropic/claude-opus-4-6") == NORM.parse("anthropic/claude-opus-4.6")
        assert NORM.parse_pair("anthropic", "claude-opus-4-6") == NORM.parse_pair(
            "anthropic", "claude-opus-4.6"
        )
        assert NORM.parse_routed("openrouter/anthropic/claude-sonnet-4-5") == NORM.parse_routed(
            "openrouter/anthropic/claude-sonnet-4.5"
        )
        assert NORM.parse("claude-opus-4-6").canonical == "anthropic/claude-opus-4.6"

    def test_wire_and_routed_keep_the_spelled_name_behind_a_route(self) -> None:
        dashed = NORM.parse_pair("anthropic", "claude-opus-4-6")
        dotted = NORM.parse_pair("anthropic", "claude-opus-4.6")
        assert (dashed.model, dashed.wire, dashed.routed) == (
            "claude-opus-4.6",
            "claude-opus-4-6",
            "anthropic/claude-opus-4-6",
        )
        assert (dotted.wire, dotted.routed) == ("claude-opus-4.6", "anthropic/claude-opus-4.6")
        assert not dashed.same_wire(dotted)
        assert dashed.same_wire(NORM.parse_pair("anthropic", "claude-opus-4-6"))
        assert dashed.generation == dotted.generation == "4.6"

    def test_bare_name_takes_the_dot_once_its_vendor_is_known(self) -> None:
        name = NORM.parse("claude-sonnet-4-6")
        assert (name.canonical, name.wire, name.routed) == (
            "anthropic/claude-sonnet-4.6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-6",
        )

    def test_the_dot_also_before_a_vertex_snapshot(self) -> None:
        name = NORM.parse("vertex_ai/claude-opus-4-1@20250805")
        assert (name.canonical, name.wire) == (
            "anthropic/claude-opus-4.1@20250805",
            "claude-opus-4-1@20250805",
        )

    @pytest.mark.parametrize(
        "text, model",
        [
            ("anthropic/claude-3-5-sonnet-20240620", "claude-3.5-sonnet-20240620"),
            ("anthropic/claude-opus-4-1-20250805", "claude-opus-4.1-20250805"),
            ("anthropic/claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
            ("anthropic/claude-opus-4-20250514", "claude-opus-4-20250514"),
            ("anthropic/claude-3-haiku-20240307", "claude-3-haiku-20240307"),
            ("anthropic/claude-opus-4.6-2026-01-15", "claude-opus-4.6-2026-01-15"),
            ("anthropic/claude-opus-4-6-2026-01-15", "claude-opus-4.6-2026-01-15"),
        ],
    )
    def test_dated_and_eight_digit_snapshots_stay(self, text: str, model: str) -> None:
        assert NORM.parse_canonical(text).model == model
        assert NORM.parse_canonical(text).wire == text

    def test_only_the_listed_vendors_take_the_dot(self) -> None:
        assert NORM.parse_canonical("anthropic/claude-haiku-4-5").model == "claude-haiku-4.5"
        assert NORM.parse_canonical("mistralai/mistral-medium-3-5").model == "mistral-medium-3-5"
        assert NORM.parse("mistral/mistral-medium-3-5").canonical == "mistralai/mistral-medium-3-5"
        assert NORM.parse_canonical("x-ai/grok-4-1-fast").model == "grok-4-1-fast"
        assert NORM.parse_canonical("openai/gpt-4-0613").model == "gpt-4-0613"

    def test_fireworks_decimal_p(self) -> None:
        full = NORM.parse("fireworks_ai/accounts/fireworks/models/glm-5p1")
        short = NORM.parse("fireworks_ai/glm-5p1")
        assert full.canonical == short.canonical == "z-ai/glm-5.1"
        assert full == short
        assert (full.wire, short.wire) == ("accounts/fireworks/models/glm-5p1", "glm-5p1")
        assert (
            NORM.parse("fireworks_ai/accounts/fireworks/models/kimi-k2p5").canonical == "moonshotai/kimi-k2.5"
        )

    def test_default_slug_follows_the_identity_and_the_route_in_slug_follows_the_wire(self) -> None:
        assert NORM.parse_pair("anthropic", "claude-opus-4-6").slug == "anthropic_claude-opus-4.6"
        assert ENGINE.parse_pair("anthropic", "claude-opus-4-6").slug == "anthropic__claude-opus-4-6"
        assert ENGINE.parse_pair("anthropic", "claude-opus-4.6").slug == "anthropic__claude-opus-4.6"


class TestDeploymentDetail:
    BEDROCK = (
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )

    def test_bedrock_region_prefixes_give_one_identity(self) -> None:
        names = [NORM.parse_pair("bedrock", text) for text in self.BEDROCK]
        for name, text in zip(names, self.BEDROCK, strict=True):
            assert (name.vendor, name.model, name.route) == (
                "anthropic",
                "claude-sonnet-4.5-20250929-v1",
                "bedrock",
            )
            assert (name.wire, name.routed) == (text, f"bedrock/{text}")
        first, *rest = names
        for other in rest:
            assert other == first
            assert not other.same_wire(first)

    @pytest.mark.parametrize(
        "detailed, plain",
        [
            ("bedrock/converse/amazon.nova-pro-v1:0", "bedrock/amazon.nova-pro-v1:0"),
            (
                "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
                "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            ),
            ("bedrock/us-west-1/meta.llama3-70b-instruct-v1:0", "bedrock/meta.llama3-70b-instruct-v1:0"),
            ("bedrock/eu-central-1/1-month-commitment/anthropic.claude-v1", "bedrock/anthropic.claude-v1"),
            ("bedrock/*/6-month-commitment/cohere.command-text-v14", "bedrock/cohere.command-text-v14"),
            ("azure/eu/gpt-5.5", "azure/gpt-5.5"),
            ("azure/us/o3-mini-2025-01-31", "azure/o3-mini-2025-01-31"),
            ("azure/global-standard/gpt-4o-mini", "azure/gpt-4o-mini"),
            ("azure_ai/global/grok-3", "azure_ai/grok-3"),
            ("vertex_ai/claude-opus-4-7@default", "vertex_ai/claude-opus-4-7"),
            ("vertex_ai/zai-org/glm-5-maas", "vertex_ai/zai-org/glm-5"),
            ("databricks/databricks-claude-sonnet-4-5", "databricks/claude-sonnet-4-5"),
        ],
    )
    def test_deployment_detail_leaves_the_model_and_stays_in_the_wire(
        self, detailed: str, plain: str
    ) -> None:
        with_detail, without = NORM.parse(detailed), NORM.parse(plain)
        assert with_detail == without
        route = detailed.split("/", 1)[0]
        assert with_detail.routes == without.routes == (route,)
        assert with_detail.wire == detailed.split("/", 1)[1]
        assert with_detail.routed == detailed
        assert not with_detail.same_wire(without)

    @pytest.mark.parametrize(
        "provider, name, model",
        [
            ("bedrock", "amazon.nova-pro-v1:0", "nova-pro-v1"),
            ("bedrock", "mistral.mistral-large-2402-v1:0", "mistral-large-2402-v1"),
            ("bedrock", "openai.gpt-oss-120b-1:0", "gpt-oss-120b-1"),
        ],
    )
    def test_a_bedrock_revision_suffix_leaves_the_model(self, provider: str, name: str, model: str) -> None:
        value = NORM.parse_pair(provider, name)
        assert (value.model, value.wire, value.routed) == (model, name, f"{provider}/{name}")

    def test_vertex_maas_suffix_leaves_the_model(self) -> None:
        name = NORM.parse("vertex_ai/meta/llama-3.1-405b-instruct-maas")
        assert (name.canonical, name.wire) == (
            "meta-llama/llama-3.1-405b-instruct",
            "meta/llama-3.1-405b-instruct-maas",
        )

    @pytest.mark.parametrize(
        "text, model",
        [
            ("databricks/databricks-mpt-30b-instruct", "mpt-30b-instruct"),
            ("databricks/databricks-dbrx-instruct", "dbrx-instruct"),
        ],
    )
    def test_a_route_vendors_own_model_drops_the_prefix_through_every_entry_point(
        self, text: str, model: str
    ) -> None:
        route, name = text.split("/", 1)
        values = [
            NORM.parse(text),
            NORM.parse_canonical(text),
            NORM.parse_routed(text),
            NORM.parse_pair(route, name),
        ]
        assert {(value.vendor, value.model) for value in values} == {("databricks", model)}

    def test_databricks_prefix_alone_is_a_malformed_name(self) -> None:
        with pytest.raises(MalformedName):
            NORM.parse("databricks/databricks-")

    def test_slugs_of_a_bedrock_id(self) -> None:
        text = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert NORM.parse_pair("bedrock", text).slug == "anthropic_claude-sonnet-4.5-20250929-v1"
        assert (
            ENGINE.parse_pair("bedrock", text).slug == "bedrock__us.anthropic.claude-sonnet-4-5-20250929-v1_0"
        )


class TestPublicSurface:
    def test_wire_is_keyword_only_and_outside_equality(self) -> None:
        wire = {field.name: field for field in dataclasses.fields(ModelName)}["wire"]
        assert wire.kw_only is True
        assert wire.compare is False

    def test_vendor_source_is_part_of_the_provenance(self) -> None:
        assert "vendor_source" in {field.name for field in dataclasses.fields(Provenance)}

    def test_to_dict_carries_wire_and_vendor_source(self) -> None:
        data = NORM.parse_pair("anthropic", "claude-opus-4-6").to_dict()
        assert (data["model"], data["wire"], data["routed"]) == (
            "claude-opus-4.6",
            "claude-opus-4-6",
            "anthropic/claude-opus-4-6",
        )
        assert data["provenance"]["vendor_source"] == "route"

    def test_cli_json_carries_wire(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["pair", "bedrock", "us.anthropic.claude-sonnet-4-6"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert (out["canonical"], out["wire"]) == (
            "anthropic/claude-sonnet-4.6",
            "us.anthropic.claude-sonnet-4-6",
        )
        assert out["provenance"]["vendor_source"] == "spelled"
