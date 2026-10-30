"""Catalogue coverage: the packaged default reads the chat and responses ids of the LiteLLM 1.93.0 model
catalogue (tests/data/litellm_catalogue.json) with a plausible vendor and family, with no override file.

An id passes when it parses to a vendor in tests/data/allowed_vendors.json and to a family that is neither
empty, nor digits only, nor one of FORBIDDEN_FAMILIES. Every id that does not pass is listed in EXCLUDED
with its reason, and every listed id must keep failing: an exclusion that starts passing is stale and fails
the test.

The input is built the way LiteLLM sends it: the key itself when it carries a slash, else the pair (provider
route, key), where the provider route is the catalogue's provider without LiteLLM's suffixes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from toloka_model_name_normalizer import ModelName, ModelNameError, Normalizer, Rules

DATA = Path(__file__).parent / "data"
CATALOGUE = json.loads((DATA / "litellm_catalogue.json").read_text("utf-8"))["rows"]
_VENDORS = json.loads((DATA / "allowed_vendors.json").read_text("utf-8"))
ALLOWED_VENDORS = frozenset(_VENDORS["openrouter"]) | frozenset(_VENDORS["not_on_openrouter"])

MIN_SHARE = 0.97

# A family must name a model line. These tokens name a route or host, a region, or a vendor instead.
ROUTE_TOKENS = {
    "accounts",
    "azure",
    "azure_ai",
    "bedrock",
    "bedrock_mantle",
    "cf",
    "cloudflare",
    "commitment",
    "converse",
    "databricks",
    "deepinfra",
    "fireworks",
    "fireworks_ai",
    "groq",
    "hf",
    "invoke",
    "litellm",
    "litellm_proxy",
    "models",
    "nebius",
    "novita",
    "openrouter",
    "preset",
    "ps",
    "replicate",
    "routers",
    "sagemaker",
    "snowflake",
    "together",
    "together_ai",
    "vertex",
    "vertex_ai",
}
REGION_TOKENS = {"ap", "apac", "au", "ca", "eu", "global", "gov", "jp", "sa", "us"}
VENDOR_TOKENS = {
    "ai21",
    "alibaba",
    "amazon",
    "anthropic",
    "cohere",
    "google",
    "ibm",
    "meta",
    "microsoft",
    "mistralai",
    "moonshotai",
    "nvidia",
    "openai",
    "x",
    "xai",
    "z",
    "zai",
}
# not a model line: a fine-tune prefix, a tool endpoint, a router endpoint
PSEUDO_TOKENS = {"container", "ft", "model_router"}
FORBIDDEN_FAMILIES = frozenset(ROUTE_TOKENS | REGION_TOKENS | VENDOR_TOKENS | PSEUDO_TOKENS)

_PRICING_TIER = "LiteLLM pricing-tier pseudo key, not a model"
_FINE_TUNE = "an OpenAI fine-tune id (ft:...), out of scope"
_PRESET = "a Perplexity preset, not a model"
_META_ROUTER = "an OpenRouter router under a second openrouter/ segment, out of scope"
_JUMPSTART = "a SageMaker JumpStart id: the name carries no vendor and no family"
_IMAGE = "an image generation model listed without a mode, not a chat model"
_COMMUNITY = "a community re-upload or fine-tune whose publisher has no vendor slug"
_FIREWORKS = "fireworks_ai/accounts/fireworks/models/"

EXCLUDED: dict[str, str] = {
    "fallback_generalizations": "LiteLLM's fallback entry, no provider and not a model",
    "fireworks-ai-4.1b-to-16b": _PRICING_TIER,
    "fireworks-ai-56b-to-176b": _PRICING_TIER,
    "fireworks-ai-above-16b": _PRICING_TIER,
    "fireworks-ai-default": _PRICING_TIER,
    "fireworks-ai-moe-up-to-56b": _PRICING_TIER,
    "fireworks-ai-up-to-4b": _PRICING_TIER,
    "fireworks-ai-embedding-150m-to-350m": _PRICING_TIER,
    "fireworks-ai-embedding-up-to-150m": _PRICING_TIER,
    "together-ai-21.1b-41b": _PRICING_TIER,
    "together-ai-4.1b-8b": _PRICING_TIER,
    "together-ai-41.1b-80b": _PRICING_TIER,
    "together-ai-8.1b-21b": _PRICING_TIER,
    "together-ai-81.1b-110b": _PRICING_TIER,
    "together-ai-up-to-4b": _PRICING_TIER,
    "azure/container": "the Azure container tool endpoint, not a model",
    "openai/container": "the OpenAI container tool endpoint, not a model",
    "azure_ai/model_router": "Azure AI Foundry's model router, not a model",
    "ft:gpt-3.5-turbo": _FINE_TUNE,
    "ft:gpt-3.5-turbo-0125": _FINE_TUNE,
    "ft:gpt-3.5-turbo-0613": _FINE_TUNE,
    "ft:gpt-3.5-turbo-1106": _FINE_TUNE,
    "ft:gpt-4-0613": _FINE_TUNE,
    "ft:gpt-4.1-2025-04-14": _FINE_TUNE,
    "ft:gpt-4.1-mini-2025-04-14": _FINE_TUNE,
    "ft:gpt-4.1-nano-2025-04-14": _FINE_TUNE,
    "ft:gpt-4o-2024-08-06": _FINE_TUNE,
    "ft:gpt-4o-2024-11-20": _FINE_TUNE,
    "ft:gpt-4o-mini-2024-07-18": _FINE_TUNE,
    "ft:o4-mini-2025-04-16": _FINE_TUNE,
    "perplexity/preset/advanced-deep-research": _PRESET,
    "perplexity/preset/deep-research": _PRESET,
    "perplexity/preset/fast-search": _PRESET,
    "perplexity/preset/pro-search": _PRESET,
    "openrouter/openrouter/auto": _META_ROUTER,
    "openrouter/openrouter/bodybuilder": _META_ROUTER,
    "openrouter/openrouter/free": _META_ROUTER,
    "sagemaker/meta-textgeneration-llama-2-13b-f": _JUMPSTART,
    "sagemaker/meta-textgeneration-llama-2-70b-b-f": _JUMPSTART,
    "sagemaker/meta-textgeneration-llama-2-7b-f": _JUMPSTART,
    _FIREWORKS + "flux-1-dev": _IMAGE,
    _FIREWORKS + "flux-1-dev-controlnet-union": _IMAGE,
    _FIREWORKS + "flux-1-schnell": _IMAGE,
    "replicateopenai/gpt-oss-20b": "a typo in the catalogue: route and vendor glued into one segment",
    "cloudflare/@hf/thebloke/codellama-7b-instruct-awq": _COMMUNITY,
    "tensormesh/lukealonso/GLM-5.1-NVFP4-MTP": _COMMUNITY,
    "chatdolphin": _COMMUNITY,
    "llamagate/llava-7b": _COMMUNITY,
    "llamagate/openthinker-7b": _COMMUNITY,
    "ovhcloud/llava-v1.6-mistral-7b-hf": _COMMUNITY,
    _FIREWORKS + "dobby-mini-unhinged-plus-llama-3-1-8b": _COMMUNITY,
    _FIREWORKS + "dobby-unhinged-llama-3-3-70b-new": _COMMUNITY,
    _FIREWORKS + "fare-20b": _COMMUNITY,
    _FIREWORKS + "nous-capybara-7b-v1p9": _COMMUNITY,
    _FIREWORKS + "openchat-3p5-0106-7b": _COMMUNITY,
    _FIREWORKS + "openhermes-2-mistral-7b": _COMMUNITY,
    _FIREWORKS + "openhermes-2p5-mistral-7b": _COMMUNITY,
    _FIREWORKS + "openorca-7b": _COMMUNITY,
    _FIREWORKS + "toppy-m-7b": _COMMUNITY,
}

_LITELLM_SUFFIXES = {"bedrock_converse": "bedrock", "fireworks_ai-embedding-models": "fireworks_ai"}
_HAS_LETTER = re.compile(r"[a-z]")

NORMALIZER = Normalizer(Rules.default())


def provider_route(provider: str) -> str:
    """The catalogue's provider as the route LiteLLM sends: ``vertex_ai-anthropic_models`` is ``vertex_ai``,
    ``bedrock_converse`` is ``bedrock``; ``cohere_chat`` stays."""
    if provider.startswith("vertex_ai-"):
        return "vertex_ai"
    return _LITELLM_SUFFIXES.get(provider, provider)


def read(row: dict) -> ModelName:
    if "/" in row["id"]:
        return NORMALIZER.parse(row["id"])
    return NORMALIZER.parse_pair(provider_route(row["provider"]), row["id"])


def problem(row: dict) -> str | None:
    """Why the id does not pass, or ``None`` when it does."""
    try:
        name = read(row)
    except ModelNameError as exc:
        return f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # a bug, but report it with the others instead of stopping at the first
        return f"unexpected {type(exc).__name__}: {exc}"
    family = name.family.lower()
    if name.vendor not in ALLOWED_VENDORS:
        return f"vendor {name.vendor!r} is not an allowed vendor slug (family {family!r})"
    if not _HAS_LETTER.search(family) or family in FORBIDDEN_FAMILIES:
        return f"family {family!r} is not a model line (vendor {name.vendor!r})"
    return None


def outcomes() -> dict[str, str | None]:
    return {row["id"]: problem(row) for row in CATALOGUE}


def test_the_default_reads_at_least_97_percent_of_the_catalogue() -> None:
    results = outcomes()
    passing = sum(1 for reason in results.values() if reason is None)
    share = passing / len(results)
    assert share >= MIN_SHARE, (
        f"{passing} of {len(results)} ids pass ({share:.1%}), the bar is {MIN_SHARE:.0%}"
    )


def test_every_failing_id_is_a_named_exclusion() -> None:
    unexpected = {key: reason for key, reason in outcomes().items() if reason and key not in EXCLUDED}
    listing = "\n".join(f"  {key}: {reason}" for key, reason in sorted(unexpected.items()))
    assert not unexpected, f"{len(unexpected)} ids fail without a named exclusion:\n{listing}"


def test_every_named_exclusion_is_a_catalogue_id_that_still_fails() -> None:
    results = outcomes()
    unknown = sorted(key for key in EXCLUDED if key not in results)
    assert not unknown, f"exclusions that are not catalogue ids: {unknown}"
    stale = sorted(key for key in EXCLUDED if results[key] is None)
    assert not stale, f"excluded ids that now pass, remove them from EXCLUDED: {stale}"
