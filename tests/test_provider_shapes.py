"""Provider shapes: how the packaged default reads the model references of the common providers.

Every expectation is written from the providers' own ids and the OpenRouter vendor slugs, never copied from
the parser's output. Each row gives the exact vendor, model, route chain, family, ``wire`` (what was sent
after the route chain), ``routed`` (what was sent, whole) and ``provenance.vendor_source``.

``vendor_source`` names where the vendor came from:

- ``spelled``: written in the reference (a vendor segment or a dotted ``vendor.`` prefix), kept as written
  apart from the vendor case fold;
- ``alias``: written, then changed by a vendor alias or a family-scoped vendor alias;
- ``route``: implied by the route (``implies_vendor``), or an undeclared provider taken as the vendor;
- ``family``: inferred from the model's family through the families table;
- ``lookup``, ``pattern``, ``resolver``: a lookup entry, a ``bare_names`` pattern, the caller's resolver.

The model is the canonical spelling: the version dot for the vendors that write it (``claude-opus-4-6`` is
``claude-opus-4.6``), Fireworks' ``5p1`` as ``5.1``, and no deployment detail (region, mode and commitment
segments, Vertex ``@default`` and ``-maas``, a Bedrock ``:N`` suffix, the Databricks ``databricks-`` prefix).
The model's case and a vendor token inside the name stay as spelled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest

from toloka_model_name_normalizer import CallableResolver, ModelName, Normalizer, Rules, UnknownVendor

NATIVE = "native API with an implied vendor"
NATIVE_SPELLED = "native API spelled as a vendor"
HOST_VENDOR = "host route with vendor and model"
ROUTE_BARE = "route with a bare name"
BARE = "bare name alone"
DOTTED = "dotted vendor id"
VERTEX = "Vertex AI"
HOST_PATH = "host path prefix"
NAME_TOKEN = "vendor or platform token in the name"
ROUTE_VENDOR = "route that is also a vendor"
REGION_MODE = "regional and mode path segments"
PROXY = "proxy pair"

SHAPES = (
    NATIVE,
    NATIVE_SPELLED,
    HOST_VENDOR,
    ROUTE_BARE,
    BARE,
    DOTTED,
    VERTEX,
    HOST_PATH,
    NAME_TOKEN,
    ROUTE_VENDOR,
    REGION_MODE,
    PROXY,
)


@dataclass(frozen=True)
class Row:
    shape: str
    entry: str  # pair, text (parse), canonical (parse_canonical), routed (parse_routed)
    provider: str | None
    text: str
    vendor: str
    model: str
    routes: tuple[str, ...]
    family: str
    wire: str
    source: str

    @property
    def sent(self) -> str:
        """What was sent, whole: ``routed`` must reproduce it."""
        return f"{self.provider}/{self.text}" if self.entry == "pair" else self.text

    @property
    def id(self) -> str:
        where = f"({self.provider}, {self.text})" if self.entry == "pair" else f"{self.entry} {self.text}"
        return f"{self.shape}: {where}"


def P(
    shape: str,
    provider: str,
    name: str,
    vendor: str,
    model: str,
    routes: tuple[str, ...],
    family: str,
    wire: str,
    source: str,
) -> Row:
    return Row(shape, "pair", provider, name, vendor, model, routes, family, wire, source)


def T(
    shape: str,
    text: str,
    vendor: str,
    model: str,
    routes: tuple[str, ...],
    family: str,
    wire: str,
    source: str,
    entry: str = "text",
) -> Row:
    return Row(shape, entry, None, text, vendor, model, routes, family, wire, source)


# fmt: off
ROWS: list[Row] = [
    # --- native API with an implied vendor --------------------------------------------------------------
    P(NATIVE, "anthropic", "claude-opus-4-6", "anthropic", "claude-opus-4.6", ("anthropic",), "claude", "claude-opus-4-6", "route"),  # noqa: E501
    P(NATIVE, "anthropic", "claude-opus-4-1-20250805", "anthropic", "claude-opus-4.1-20250805", ("anthropic",), "claude", "claude-opus-4-1-20250805", "route"),  # noqa: E501
    P(NATIVE, "anthropic", "claude-sonnet-4-5-20250929", "anthropic", "claude-sonnet-4.5-20250929", ("anthropic",), "claude", "claude-sonnet-4-5-20250929", "route"),  # noqa: E501
    P(NATIVE, "anthropic", "claude-3-7-sonnet-20250219", "anthropic", "claude-3.7-sonnet-20250219", ("anthropic",), "claude", "claude-3-7-sonnet-20250219", "route"),  # noqa: E501
    P(NATIVE, "anthropic", "claude-haiku-4-5", "anthropic", "claude-haiku-4.5", ("anthropic",), "claude", "claude-haiku-4-5", "route"),  # noqa: E501
    P(NATIVE, "anthropic", "claude-sonnet-4-20250514", "anthropic", "claude-sonnet-4-20250514", ("anthropic",), "claude", "claude-sonnet-4-20250514", "route"),  # noqa: E501
    P(NATIVE, "openai", "gpt-5.5", "openai", "gpt-5.5", ("openai",), "gpt", "gpt-5.5", "route"),
    P(NATIVE, "openai", "o3-mini", "openai", "o3-mini", ("openai",), "o", "o3-mini", "route"),
    P(NATIVE, "openai", "gpt-4o-2024-08-06", "openai", "gpt-4o-2024-08-06", ("openai",), "gpt", "gpt-4o-2024-08-06", "route"),  # noqa: E501
    T(NATIVE, "gemini/gemini-3.5-flash", "google", "gemini-3.5-flash", ("gemini",), "gemini", "gemini-3.5-flash", "route"),  # noqa: E501
    P(NATIVE, "gemini", "gemini-2.5-pro", "google", "gemini-2.5-pro", ("gemini",), "gemini", "gemini-2.5-pro", "route"),  # noqa: E501
    T(NATIVE, "codestral/codestral-2405", "mistralai", "codestral-2405", ("codestral",), "codestral", "codestral-2405", "route"),  # noqa: E501
    T(NATIVE, "chatgpt/gpt-5.2", "openai", "gpt-5.2", ("chatgpt",), "gpt", "gpt-5.2", "route"),
    P(NATIVE, "cohere_chat", "command-r-plus", "cohere", "command-r-plus", ("cohere_chat",), "command", "command-r-plus", "route"),  # noqa: E501
    T(NATIVE, "amazon-nova/nova-pro-v1", "amazon", "nova-pro-v1", ("amazon-nova",), "nova", "nova-pro-v1", "route"),  # noqa: E501
    P(NATIVE, "meta", "muse-spark-1.2", "meta", "muse-spark-1.2", ("meta",), "muse", "muse-spark-1.2", "route"),  # noqa: E501
    P(NATIVE, "deepseek", "deepseek-v4-flash", "deepseek", "deepseek-v4-flash", ("deepseek",), "deepseek", "deepseek-v4-flash", "route"),  # noqa: E501
    P(NATIVE, "ai21", "jamba-mini-1.7", "ai21", "jamba-mini-1.7", ("ai21",), "jamba", "jamba-mini-1.7", "route"),  # noqa: E501
    # --- native API spelled as a vendor -----------------------------------------------------------------
    T(NATIVE_SPELLED, "xai/grok-4", "x-ai", "grok-4", (), "grok", "xai/grok-4", "alias"),
    T(NATIVE_SPELLED, "xai/grok-4-1-fast-reasoning", "x-ai", "grok-4-1-fast-reasoning", (), "grok", "xai/grok-4-1-fast-reasoning", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "xai/grok-code-fast-1", "x-ai", "grok-code-fast-1", (), "grok", "xai/grok-code-fast-1", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "zai/glm-4.7", "z-ai", "glm-4.7", (), "glm", "zai/glm-4.7", "alias"),
    T(NATIVE_SPELLED, "zai/glm-4.5-air", "z-ai", "glm-4.5-air", (), "glm", "zai/glm-4.5-air", "alias"),
    T(NATIVE_SPELLED, "zai/glm-5", "z-ai", "glm-5", (), "glm", "zai/glm-5", "alias"),
    T(NATIVE_SPELLED, "moonshot/kimi-k2.5", "moonshotai", "kimi-k2.5", (), "kimi", "moonshot/kimi-k2.5", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "moonshot/kimi-k2-thinking-turbo", "moonshotai", "kimi-k2-thinking-turbo", (), "kimi", "moonshot/kimi-k2-thinking-turbo", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "moonshot/moonshot-v1-128k", "moonshotai", "moonshot-v1-128k", (), "moonshot", "moonshot/moonshot-v1-128k", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "mistral/mistral-large-2512", "mistralai", "mistral-large-2512", (), "mistral", "mistral/mistral-large-2512", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "mistral/codestral-2508", "mistralai", "codestral-2508", (), "codestral", "mistral/codestral-2508", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "mistral/mistral-medium-3-5", "mistralai", "mistral-medium-3-5", (), "mistral", "mistral/mistral-medium-3-5", "alias"),  # noqa: E501
    T(NATIVE_SPELLED, "deepseek/deepseek-v3.2", "deepseek", "deepseek-v3.2", (), "deepseek", "deepseek/deepseek-v3.2", "spelled"),  # noqa: E501
    T(NATIVE_SPELLED, "minimax/MiniMax-M2.1", "minimax", "MiniMax-M2.1", (), "minimax", "minimax/MiniMax-M2.1", "spelled"),  # noqa: E501
    T(NATIVE_SPELLED, "xai/grok-4", "x-ai", "grok-4", (), "grok", "xai/grok-4", "alias", entry="canonical"),
    T(NATIVE_SPELLED, "moonshot/kimi-k2.6", "moonshotai", "kimi-k2.6", (), "kimi", "moonshot/kimi-k2.6", "alias", entry="canonical"),  # noqa: E501
    # --- host route with vendor and model ---------------------------------------------------------------
    T(HOST_VENDOR, "openrouter/anthropic/claude-opus-4.6", "anthropic", "claude-opus-4.6", ("openrouter",), "claude", "anthropic/claude-opus-4.6", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "openrouter/z-ai/glm-4.6:exacto", "z-ai", "glm-4.6:exacto", ("openrouter",), "glm", "z-ai/glm-4.6:exacto", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "openrouter/openai/o3-mini-high", "openai", "o3-mini-high", ("openrouter",), "o", "openai/o3-mini-high", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "litellm/openrouter/anthropic/claude-opus-4.6", "anthropic", "claude-opus-4.6", ("litellm", "openrouter"), "claude", "anthropic/claude-opus-4.6", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "together_ai/deepseek-ai/DeepSeek-R1", "deepseek", "DeepSeek-R1", ("together_ai",), "deepseek", "deepseek-ai/DeepSeek-R1", "alias"),  # noqa: E501
    T(HOST_VENDOR, "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama", "Llama-3.3-70B-Instruct-Turbo", ("together_ai",), "llama", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "together_ai/zai-org/GLM-4.7", "z-ai", "GLM-4.7", ("together_ai",), "glm", "zai-org/GLM-4.7", "alias"),  # noqa: E501
    T(HOST_VENDOR, "deepinfra/Qwen/Qwen3-235B-A22B-Instruct-2507", "qwen", "Qwen3-235B-A22B-Instruct-2507", ("deepinfra",), "qwen", "Qwen/Qwen3-235B-A22B-Instruct-2507", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "deepinfra/NousResearch/Hermes-3-Llama-3.1-405B", "nousresearch", "Hermes-3-Llama-3.1-405B", ("deepinfra",), "hermes", "NousResearch/Hermes-3-Llama-3.1-405B", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "groq/openai/gpt-oss-20b", "openai", "gpt-oss-20b", ("groq",), "gpt", "openai/gpt-oss-20b", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "vercel_ai_gateway/xai/grok-3", "x-ai", "grok-3", ("vercel_ai_gateway",), "grok", "xai/grok-3", "alias"),  # noqa: E501
    T(HOST_VENDOR, "vercel_ai_gateway/anthropic/claude-3-5-sonnet-20241022", "anthropic", "claude-3.5-sonnet-20241022", ("vercel_ai_gateway",), "claude", "anthropic/claude-3-5-sonnet-20241022", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "novita/zai-org/glm-4.5", "z-ai", "glm-4.5", ("novita",), "glm", "zai-org/glm-4.5", "alias"),  # noqa: E501
    T(HOST_VENDOR, "replicate/ibm-granite/granite-3.3-8b-instruct", "ibm-granite", "granite-3.3-8b-instruct", ("replicate",), "granite", "ibm-granite/granite-3.3-8b-instruct", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "watsonx/ibm/granite-3-8b-instruct", "ibm-granite", "granite-3-8b-instruct", ("watsonx",), "granite", "ibm/granite-3-8b-instruct", "alias"),  # noqa: E501
    T(HOST_VENDOR, "baseten/MiniMaxAI/MiniMax-M2.5", "minimax", "MiniMax-M2.5", ("baseten",), "minimax", "MiniMaxAI/MiniMax-M2.5", "alias"),  # noqa: E501
    T(HOST_VENDOR, "scaleway/mistralai/mistral-medium-3.5-128b", "mistralai", "mistral-medium-3.5-128b", ("scaleway",), "mistral", "mistralai/mistral-medium-3.5-128b", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "hyperbolic/Qwen/QwQ-32B", "qwen", "QwQ-32B", ("hyperbolic",), "qwq", "Qwen/QwQ-32B", "spelled"),  # noqa: E501
    T(HOST_VENDOR, "novita/xiaomimimo/mimo-v2-flash", "xiaomi", "mimo-v2-flash", ("novita",), "mimo", "xiaomimimo/mimo-v2-flash", "alias"),  # noqa: E501
    T(HOST_VENDOR, "openrouter/qwen/qwen3.5-plus-02-15", "qwen", "qwen3.5-plus-02-15", ("openrouter",), "qwen", "qwen/qwen3.5-plus-02-15", "spelled", entry="routed"),  # noqa: E501
    # --- route with a bare name -------------------------------------------------------------------------
    T(ROUTE_BARE, "azure/gpt-5.5", "openai", "gpt-5.5", ("azure",), "gpt", "gpt-5.5", "family"),
    T(ROUTE_BARE, "azure/o3", "openai", "o3", ("azure",), "o", "o3", "family"),
    T(ROUTE_BARE, "azure/gpt-4o-mini-2024-07-18", "openai", "gpt-4o-mini-2024-07-18", ("azure",), "gpt", "gpt-4o-mini-2024-07-18", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure/command-r-plus", "cohere", "command-r-plus", ("azure",), "command", "command-r-plus", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure/mistral-large-2402", "mistralai", "mistral-large-2402", ("azure",), "mistral", "mistral-large-2402", "family"),  # noqa: E501
    P(ROUTE_BARE, "azure", "gpt-5.4-pro", "openai", "gpt-5.4-pro", ("azure",), "gpt", "gpt-5.4-pro", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/Llama-4-Scout-17B-16E-Instruct", "meta-llama", "Llama-4-Scout-17B-16E-Instruct", ("azure_ai",), "llama", "Llama-4-Scout-17B-16E-Instruct", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/Phi-4-mini-instruct", "microsoft", "Phi-4-mini-instruct", ("azure_ai",), "phi", "Phi-4-mini-instruct", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/grok-3-mini", "x-ai", "grok-3-mini", ("azure_ai",), "grok", "grok-3-mini", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/deepseek-v3.2", "deepseek", "deepseek-v3.2", ("azure_ai",), "deepseek", "deepseek-v3.2", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/claude-opus-4-6", "anthropic", "claude-opus-4.6", ("azure_ai",), "claude", "claude-opus-4-6", "family"),  # noqa: E501
    T(ROUTE_BARE, "azure_ai/kimi-k2.5", "moonshotai", "kimi-k2.5", ("azure_ai",), "kimi", "kimi-k2.5", "family"),  # noqa: E501
    T(ROUTE_BARE, "groq/llama-3.3-70b-versatile", "meta-llama", "llama-3.3-70b-versatile", ("groq",), "llama", "llama-3.3-70b-versatile", "family"),  # noqa: E501
    T(ROUTE_BARE, "dashscope/qwen3-max", "qwen", "qwen3-max", ("dashscope",), "qwen", "qwen3-max", "family"),
    T(ROUTE_BARE, "cerebras/gpt-oss-120b", "openai", "gpt-oss-120b", ("cerebras",), "gpt", "gpt-oss-120b", "family"),  # noqa: E501
    T(ROUTE_BARE, "github_copilot/claude-sonnet-4.5", "anthropic", "claude-sonnet-4.5", ("github_copilot",), "claude", "claude-sonnet-4.5", "family"),  # noqa: E501
    T(ROUTE_BARE, "sambanova/DeepSeek-V3.1", "deepseek", "DeepSeek-V3.1", ("sambanova",), "deepseek", "DeepSeek-V3.1", "family"),  # noqa: E501
    P(ROUTE_BARE, "volcengine", "glm-4-7-251222", "z-ai", "glm-4.7-251222", ("volcengine",), "glm", "glm-4-7-251222", "family"),  # noqa: E501
    # --- bare name alone --------------------------------------------------------------------------------
    T(BARE, "claude-opus-4-7", "anthropic", "claude-opus-4.7", (), "claude", "claude-opus-4-7", "family"),
    T(BARE, "claude-3-5-haiku-20241022", "anthropic", "claude-3.5-haiku-20241022", (), "claude", "claude-3-5-haiku-20241022", "family"),  # noqa: E501
    T(BARE, "claude-sonnet-4.6", "anthropic", "claude-sonnet-4.6", (), "claude", "claude-sonnet-4.6", "family"),  # noqa: E501
    T(BARE, "gemini-3.5-flash", "google", "gemini-3.5-flash", (), "gemini", "gemini-3.5-flash", "family"),
    T(BARE, "o3-mini", "openai", "o3-mini", (), "o", "o3-mini", "family"),
    T(BARE, "o1", "openai", "o1", (), "o", "o1", "family"),
    T(BARE, "gpt-5.5", "openai", "gpt-5.5", (), "gpt", "gpt-5.5", "family"),
    T(BARE, "DeepSeek-R1", "deepseek", "DeepSeek-R1", (), "deepseek", "DeepSeek-R1", "family"),
    T(BARE, "deepseek-v4-flash", "deepseek", "deepseek-v4-flash", (), "deepseek", "deepseek-v4-flash", "family"),  # noqa: E501
    T(BARE, "grok-4", "x-ai", "grok-4", (), "grok", "grok-4", "family"),
    T(BARE, "kimi-k2.5", "moonshotai", "kimi-k2.5", (), "kimi", "kimi-k2.5", "family"),
    T(BARE, "glm-4-7", "z-ai", "glm-4.7", (), "glm", "glm-4-7", "family"),
    T(BARE, "qwen3-max", "qwen", "qwen3-max", (), "qwen", "qwen3-max", "family"),
    T(BARE, "mistral-large-2512", "mistralai", "mistral-large-2512", (), "mistral", "mistral-large-2512", "family"),  # noqa: E501
    T(BARE, "command-a-03-2025", "cohere", "command-a-03-2025", (), "command", "command-a-03-2025", "family"),
    T(BARE, "llama-3.3-70b-instruct", "meta-llama", "llama-3.3-70b-instruct", (), "llama", "llama-3.3-70b-instruct", "family"),  # noqa: E501
    T(BARE, "muse-spark-1.2", "meta", "muse-spark-1.2", (), "muse", "muse-spark-1.2", "family"),
    T(BARE, "jamba-1.5-large", "ai21", "jamba-1.5-large", (), "jamba", "jamba-1.5-large", "family"),
    T(BARE, "MiniMax-M2.5", "minimax", "MiniMax-M2.5", (), "minimax", "MiniMax-M2.5", "family"),
    # --- dotted vendor id -------------------------------------------------------------------------------
    P(DOTTED, "bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic", "claude-sonnet-4.5-20250929-v1", ("bedrock",), "claude", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0", "anthropic", "claude-3.5-haiku-20241022-v1", ("bedrock",), "claude", "anthropic.claude-3-5-haiku-20241022-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "global.anthropic.claude-opus-4-6-v1", "anthropic", "claude-opus-4.6-v1", ("bedrock",), "claude", "global.anthropic.claude-opus-4-6-v1", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "eu.anthropic.claude-sonnet-4-6", "anthropic", "claude-sonnet-4.6", ("bedrock",), "claude", "eu.anthropic.claude-sonnet-4-6", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic", "claude-sonnet-4.5-20250929-v1", ("bedrock",), "claude", "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "apac.amazon.nova-pro-v1:0", "amazon", "nova-pro-v1", ("bedrock",), "nova", "apac.amazon.nova-pro-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "amazon.titan-text-premier-v1:0", "amazon", "titan-text-premier-v1", ("bedrock",), "titan", "amazon.titan-text-premier-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "mistral.mistral-large-2407-v1:0", "mistralai", "mistral-large-2407-v1", ("bedrock",), "mistral", "mistral.mistral-large-2407-v1:0", "alias"),  # noqa: E501
    P(DOTTED, "bedrock", "us.meta.llama3-1-70b-instruct-v1:0", "meta-llama", "llama3-1-70b-instruct-v1", ("bedrock",), "llama", "us.meta.llama3-1-70b-instruct-v1:0", "alias"),  # noqa: E501
    P(DOTTED, "bedrock", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen", "qwen3-coder-30b-a3b-v1", ("bedrock",), "qwen", "qwen.qwen3-coder-30b-a3b-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "moonshot.kimi-k2-thinking", "moonshotai", "kimi-k2-thinking", ("bedrock",), "kimi", "moonshot.kimi-k2-thinking", "alias"),  # noqa: E501
    P(DOTTED, "bedrock", "zai.glm-4.7", "z-ai", "glm-4.7", ("bedrock",), "glm", "zai.glm-4.7", "alias"),
    P(DOTTED, "bedrock", "openai.gpt-oss-120b-1:0", "openai", "gpt-oss-120b-1", ("bedrock",), "gpt", "openai.gpt-oss-120b-1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "ai21.jamba-1-5-large-v1:0", "ai21", "jamba-1.5-large-v1", ("bedrock",), "jamba", "ai21.jamba-1-5-large-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "writer.palmyra-x5-v1:0", "writer", "palmyra-x5-v1", ("bedrock",), "palmyra", "writer.palmyra-x5-v1:0", "spelled"),  # noqa: E501
    P(DOTTED, "bedrock", "claude-sonnet-4-5-20250929-v1:0", "anthropic", "claude-sonnet-4.5-20250929-v1", ("bedrock",), "claude", "claude-sonnet-4-5-20250929-v1:0", "family"),  # noqa: E501
    T(DOTTED, "oci/meta.llama-3.1-405b-instruct", "meta-llama", "llama-3.1-405b-instruct", ("oci",), "llama", "meta.llama-3.1-405b-instruct", "alias"),  # noqa: E501
    T(DOTTED, "oci/cohere.command-a-03-2025", "cohere", "command-a-03-2025", ("oci",), "command", "cohere.command-a-03-2025", "spelled"),  # noqa: E501
    T(DOTTED, "oci/xai.grok-4", "x-ai", "grok-4", ("oci",), "grok", "xai.grok-4", "alias"),
    T(DOTTED, "oci/openai.gpt-5-mini", "openai", "gpt-5-mini", ("oci",), "gpt", "openai.gpt-5-mini", "spelled"),  # noqa: E501
    T(DOTTED, "oci/google.gemini-2.5-pro", "google", "gemini-2.5-pro", ("oci",), "gemini", "google.gemini-2.5-pro", "spelled"),  # noqa: E501
    T(DOTTED, "bedrock_mantle/openai.gpt-oss-120b", "openai", "gpt-oss-120b", ("bedrock_mantle",), "gpt", "openai.gpt-oss-120b", "spelled"),  # noqa: E501
    T(DOTTED, "bedrock_mantle/google.gemma-4-31b", "google", "gemma-4-31b", ("bedrock_mantle",), "gemma", "google.gemma-4-31b", "spelled"),  # noqa: E501
    T(DOTTED, "bedrock_mantle/xai.grok-4.3", "x-ai", "grok-4.3", ("bedrock_mantle",), "grok", "xai.grok-4.3", "alias"),  # noqa: E501
    # --- Vertex AI --------------------------------------------------------------------------------------
    T(VERTEX, "vertex_ai/claude-opus-4-7@default", "anthropic", "claude-opus-4.7", ("vertex_ai",), "claude", "claude-opus-4-7@default", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/claude-opus-4-1@20250805", "anthropic", "claude-opus-4.1@20250805", ("vertex_ai",), "claude", "claude-opus-4-1@20250805", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/claude-3-5-sonnet@20240620", "anthropic", "claude-3.5-sonnet@20240620", ("vertex_ai",), "claude", "claude-3-5-sonnet@20240620", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/claude-sonnet-4@20250514", "anthropic", "claude-sonnet-4@20250514", ("vertex_ai",), "claude", "claude-sonnet-4@20250514", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/mistral-large@2407", "mistralai", "mistral-large@2407", ("vertex_ai",), "mistral", "mistral-large@2407", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/codestral-2@001", "mistralai", "codestral-2@001", ("vertex_ai",), "codestral", "codestral-2@001", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/jamba-1.5-large@001", "ai21", "jamba-1.5-large@001", ("vertex_ai",), "jamba", "jamba-1.5-large@001", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/gemini-3.1-pro-preview", "google", "gemini-3.1-pro-preview", ("vertex_ai",), "gemini", "gemini-3.1-pro-preview", "family"),  # noqa: E501
    P(VERTEX, "vertex_ai", "gemini-2.0-flash-001", "google", "gemini-2.0-flash-001", ("vertex_ai",), "gemini", "gemini-2.0-flash-001", "family"),  # noqa: E501
    T(VERTEX, "vertex_ai/meta/llama-3.1-405b-instruct-maas", "meta-llama", "llama-3.1-405b-instruct", ("vertex_ai",), "llama", "meta/llama-3.1-405b-instruct-maas", "alias"),  # noqa: E501
    T(VERTEX, "vertex_ai/zai-org/glm-5-maas", "z-ai", "glm-5", ("vertex_ai",), "glm", "zai-org/glm-5-maas", "alias"),  # noqa: E501
    T(VERTEX, "vertex_ai/deepseek-ai/deepseek-v3.2-maas", "deepseek", "deepseek-v3.2", ("vertex_ai",), "deepseek", "deepseek-ai/deepseek-v3.2-maas", "alias"),  # noqa: E501
    T(VERTEX, "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas", "qwen", "qwen3-next-80b-a3b-instruct", ("vertex_ai",), "qwen", "qwen/qwen3-next-80b-a3b-instruct-maas", "spelled"),  # noqa: E501
    T(VERTEX, "vertex_ai/openai/gpt-oss-120b-maas", "openai", "gpt-oss-120b", ("vertex_ai",), "gpt", "openai/gpt-oss-120b-maas", "spelled"),  # noqa: E501
    T(VERTEX, "vertex_ai/moonshotai/kimi-k2-thinking-maas", "moonshotai", "kimi-k2-thinking", ("vertex_ai",), "kimi", "moonshotai/kimi-k2-thinking-maas", "spelled"),  # noqa: E501
    T(VERTEX, "vertex_ai/minimaxai/minimax-m2-maas", "minimax", "minimax-m2", ("vertex_ai",), "minimax", "minimaxai/minimax-m2-maas", "alias"),  # noqa: E501
    T(VERTEX, "vertex_ai/xai/grok-4.20-reasoning", "x-ai", "grok-4.20-reasoning", ("vertex_ai",), "grok", "xai/grok-4.20-reasoning", "alias"),  # noqa: E501
    # --- host path prefix -------------------------------------------------------------------------------
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/glm-5p1", "z-ai", "glm-5.1", ("fireworks_ai",), "glm", "accounts/fireworks/models/glm-5p1", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/kimi-k2p5", "moonshotai", "kimi-k2.5", ("fireworks_ai",), "kimi", "accounts/fireworks/models/kimi-k2p5", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/deepseek-v3p1", "deepseek", "deepseek-v3.1", ("fireworks_ai",), "deepseek", "accounts/fireworks/models/deepseek-v3p1", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct", "meta-llama", "llama-v3.1-405b-instruct", ("fireworks_ai",), "llama", "accounts/fireworks/models/llama-v3p1-405b-instruct", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct", "qwen", "qwen2.5-coder-32b-instruct", ("fireworks_ai",), "qwen", "accounts/fireworks/models/qwen2p5-coder-32b-instruct", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/gpt-oss-120b", "openai", "gpt-oss-120b", ("fireworks_ai",), "gpt", "accounts/fireworks/models/gpt-oss-120b", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/accounts/fireworks/models/minimax-m2p1", "minimax", "minimax-m2.1", ("fireworks_ai",), "minimax", "accounts/fireworks/models/minimax-m2p1", "family"),  # noqa: E501
    T(HOST_PATH, "fireworks_ai/glm-5p2", "z-ai", "glm-5.2", ("fireworks_ai",), "glm", "glm-5p2", "family"),
    T(HOST_PATH, "cloudflare/@cf/meta/llama-3.2-3b-instruct", "meta-llama", "llama-3.2-3b-instruct", ("cloudflare",), "llama", "@cf/meta/llama-3.2-3b-instruct", "alias"),  # noqa: E501
    T(HOST_PATH, "cloudflare/@cf/qwen/qwq-32b", "qwen", "qwq-32b", ("cloudflare",), "qwq", "@cf/qwen/qwq-32b", "spelled"),  # noqa: E501
    T(HOST_PATH, "cloudflare/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "deepseek", "deepseek-r1-distill-qwen-32b", ("cloudflare",), "deepseek", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "alias"),  # noqa: E501
    T(HOST_PATH, "cloudflare/@cf/openai/gpt-oss-20b", "openai", "gpt-oss-20b", ("cloudflare",), "gpt", "@cf/openai/gpt-oss-20b", "spelled"),  # noqa: E501
    T(HOST_PATH, "cloudflare/@cf/zai-org/glm-4.7-flash", "z-ai", "glm-4.7-flash", ("cloudflare",), "glm", "@cf/zai-org/glm-4.7-flash", "alias"),  # noqa: E501
    T(HOST_PATH, "cloudflare/@cf/google/gemma-4-26b-a4b-it", "google", "gemma-4-26b-a4b-it", ("cloudflare",), "gemma", "@cf/google/gemma-4-26b-a4b-it", "spelled"),  # noqa: E501
    # --- vendor or platform token in the name -----------------------------------------------------------
    T(NAME_TOKEN, "databricks/databricks-claude-sonnet-4-5", "anthropic", "claude-sonnet-4.5", ("databricks",), "claude", "databricks-claude-sonnet-4-5", "family"),  # noqa: E501
    T(NAME_TOKEN, "databricks/databricks-claude-3-7-sonnet", "anthropic", "claude-3.7-sonnet", ("databricks",), "claude", "databricks-claude-3-7-sonnet", "family"),  # noqa: E501
    T(NAME_TOKEN, "databricks/databricks-gpt-5-mini", "openai", "gpt-5-mini", ("databricks",), "gpt", "databricks-gpt-5-mini", "family"),  # noqa: E501
    T(NAME_TOKEN, "databricks/databricks-gemini-2-5-pro", "google", "gemini-2-5-pro", ("databricks",), "gemini", "databricks-gemini-2-5-pro", "family"),  # noqa: E501
    T(NAME_TOKEN, "databricks/databricks-meta-llama-3-3-70b-instruct", "meta-llama", "meta-llama-3-3-70b-instruct", ("databricks",), "llama", "databricks-meta-llama-3-3-70b-instruct", "family"),  # noqa: E501
    T(NAME_TOKEN, "snowflake/openai-gpt-5-mini", "openai", "openai-gpt-5-mini", ("snowflake",), "gpt", "openai-gpt-5-mini", "family"),  # noqa: E501
    T(NAME_TOKEN, "snowflake/claude-sonnet-4-6", "anthropic", "claude-sonnet-4.6", ("snowflake",), "claude", "claude-sonnet-4-6", "family"),  # noqa: E501
    T(NAME_TOKEN, "snowflake/snowflake-llama-3.3-70b", "meta-llama", "snowflake-llama-3.3-70b", ("snowflake",), "llama", "snowflake-llama-3.3-70b", "family"),  # noqa: E501
    T(NAME_TOKEN, "cerebras/zai-glm-4.7", "z-ai", "zai-glm-4.7", ("cerebras",), "glm", "zai-glm-4.7", "family"),  # noqa: E501
    T(NAME_TOKEN, "gradient_ai/anthropic-claude-3.7-sonnet", "anthropic", "anthropic-claude-3.7-sonnet", ("gradient_ai",), "claude", "anthropic-claude-3.7-sonnet", "family"),  # noqa: E501
    T(NAME_TOKEN, "gradient_ai/openai-gpt-4o", "openai", "openai-gpt-4o", ("gradient_ai",), "gpt", "openai-gpt-4o", "family"),  # noqa: E501
    T(NAME_TOKEN, "gradient_ai/openai-o3-mini", "openai", "openai-o3-mini", ("gradient_ai",), "o", "openai-o3-mini", "family"),  # noqa: E501
    T(NAME_TOKEN, "gradient_ai/alibaba-qwen3-32b", "qwen", "alibaba-qwen3-32b", ("gradient_ai",), "qwen", "alibaba-qwen3-32b", "family"),  # noqa: E501
    T(NAME_TOKEN, "azure_ai/Meta-Llama-3.1-405B-Instruct", "meta-llama", "Meta-Llama-3.1-405B-Instruct", ("azure_ai",), "llama", "Meta-Llama-3.1-405B-Instruct", "family"),  # noqa: E501
    T(NAME_TOKEN, "azure_ai/cohere-command-a-plus-05-2026", "cohere", "cohere-command-a-plus-05-2026", ("azure_ai",), "command", "cohere-command-a-plus-05-2026", "family"),  # noqa: E501
    T(NAME_TOKEN, "azure_ai/AI21-Jamba-1.5-Large", "ai21", "AI21-Jamba-1.5-Large", ("azure_ai",), "jamba", "AI21-Jamba-1.5-Large", "family"),  # noqa: E501
    T(NAME_TOKEN, "sambanova/Meta-Llama-3.3-70B-Instruct", "meta-llama", "Meta-Llama-3.3-70B-Instruct", ("sambanova",), "llama", "Meta-Llama-3.3-70B-Instruct", "family"),  # noqa: E501
    T(NAME_TOKEN, "friendliai/meta-llama-3.1-8b-instruct", "meta-llama", "meta-llama-3.1-8b-instruct", ("friendliai",), "llama", "meta-llama-3.1-8b-instruct", "family"),  # noqa: E501
    # --- route that is also a vendor --------------------------------------------------------------------
    T(ROUTE_VENDOR, "perplexity/sonar-pro", "perplexity", "sonar-pro", (), "sonar", "perplexity/sonar-pro", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/sonar-reasoning-pro", "perplexity", "sonar-reasoning-pro", (), "sonar", "perplexity/sonar-reasoning-pro", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/sonar", "perplexity", "sonar", (), "sonar", "perplexity/sonar", "spelled"),
    T(ROUTE_VENDOR, "perplexity/sonar-pro", "perplexity", "sonar-pro", (), "sonar", "perplexity/sonar-pro", "spelled", entry="canonical"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/anthropic/claude-opus-4-7", "anthropic", "claude-opus-4.7", ("perplexity",), "claude", "anthropic/claude-opus-4-7", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/openai/gpt-5.1", "openai", "gpt-5.1", ("perplexity",), "gpt", "openai/gpt-5.1", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/perplexity/sonar", "perplexity", "sonar", ("perplexity",), "sonar", "perplexity/sonar", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "perplexity/llama-3.1-70b-instruct", "meta-llama", "llama-3.1-70b-instruct", ("perplexity",), "llama", "llama-3.1-70b-instruct", "family"),  # noqa: E501
    T(ROUTE_VENDOR, "tencent/hy3", "tencent", "hy3", (), "hy", "tencent/hy3", "spelled"),
    T(ROUTE_VENDOR, "tencent/hy3", "tencent", "hy3", (), "hy", "tencent/hy3", "spelled", entry="canonical"),
    T(ROUTE_VENDOR, "tencent/hunyuan-a13b-instruct", "tencent", "hunyuan-a13b-instruct", (), "hunyuan", "tencent/hunyuan-a13b-instruct", "spelled"),  # noqa: E501
    T(ROUTE_VENDOR, "tencent/deepseek-v4-flash", "deepseek", "deepseek-v4-flash", ("tencent",), "deepseek", "deepseek-v4-flash", "family"),  # noqa: E501
    T(ROUTE_VENDOR, "tencent/deepseek-v4-pro", "deepseek", "deepseek-v4-pro", ("tencent",), "deepseek", "deepseek-v4-pro", "family"),  # noqa: E501
    T(ROUTE_VENDOR, "databricks/dbrx-instruct", "databricks", "dbrx-instruct", (), "dbrx", "databricks/dbrx-instruct", "spelled"),  # noqa: E501
    # --- regional and mode path segments ----------------------------------------------------------------
    T(REGION_MODE, "azure/eu/gpt-5.5", "openai", "gpt-5.5", ("azure",), "gpt", "eu/gpt-5.5", "family"),
    T(REGION_MODE, "azure/us/gpt-5.4-2026-03-05", "openai", "gpt-5.4-2026-03-05", ("azure",), "gpt", "us/gpt-5.4-2026-03-05", "family"),  # noqa: E501
    T(REGION_MODE, "azure/global/gpt-5.1-codex", "openai", "gpt-5.1-codex", ("azure",), "gpt", "global/gpt-5.1-codex", "family"),  # noqa: E501
    T(REGION_MODE, "azure/global-standard/gpt-4o-mini", "openai", "gpt-4o-mini", ("azure",), "gpt", "global-standard/gpt-4o-mini", "family"),  # noqa: E501
    T(REGION_MODE, "azure/eu/o3-mini-2025-01-31", "openai", "o3-mini-2025-01-31", ("azure",), "o", "eu/o3-mini-2025-01-31", "family"),  # noqa: E501
    T(REGION_MODE, "azure_ai/global/grok-3-mini", "x-ai", "grok-3-mini", ("azure_ai",), "grok", "global/grok-3-mini", "family"),  # noqa: E501
    T(REGION_MODE, "bedrock/converse/amazon.nova-pro-v1:0", "amazon", "nova-pro-v1", ("bedrock",), "nova", "converse/amazon.nova-pro-v1:0", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic", "claude-3.5-sonnet-20240620-v1", ("bedrock",), "claude", "invoke/anthropic.claude-3-5-sonnet-20240620-v1:0", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/us-west-1/meta.llama3-70b-instruct-v1:0", "meta-llama", "llama3-70b-instruct-v1", ("bedrock",), "llama", "us-west-1/meta.llama3-70b-instruct-v1:0", "alias"),  # noqa: E501
    T(REGION_MODE, "bedrock/eu-central-1/1-month-commitment/anthropic.claude-v1", "anthropic", "claude-v1", ("bedrock",), "claude", "eu-central-1/1-month-commitment/anthropic.claude-v1", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/us-west-2/6-month-commitment/anthropic.claude-instant-v1", "anthropic", "claude-instant-v1", ("bedrock",), "claude", "us-west-2/6-month-commitment/anthropic.claude-instant-v1", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/*/1-month-commitment/cohere.command-text-v14", "cohere", "command-text-v14", ("bedrock",), "command", "*/1-month-commitment/cohere.command-text-v14", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/ap-northeast-1/moonshotai.kimi-k2.5", "moonshotai", "kimi-k2.5", ("bedrock",), "kimi", "ap-northeast-1/moonshotai.kimi-k2.5", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/us-gov-west-1/anthropic.claude-3-7-sonnet-20250219-v1:0", "anthropic", "claude-3.7-sonnet-20250219-v1", ("bedrock",), "claude", "us-gov-west-1/anthropic.claude-3-7-sonnet-20250219-v1:0", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/us-east-1/zai.glm-5", "z-ai", "glm-5", ("bedrock",), "glm", "us-east-1/zai.glm-5", "alias"),  # noqa: E501
    # ':2' is part of the version here (Mistral 7B Instruct v0.2): only the default ':0' leaves the model
    T(REGION_MODE, "bedrock/eu-west-3/mistral.mistral-7b-instruct-v0:2", "mistralai", "mistral-7b-instruct-v0:2", ("bedrock",), "mistral", "eu-west-3/mistral.mistral-7b-instruct-v0:2", "alias"),  # noqa: E501
    T(REGION_MODE, "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0", "anthropic", "claude-3.5-haiku-20241022-v1", ("bedrock",), "claude", "us.anthropic.claude-3-5-haiku-20241022-v1:0", "spelled"),  # noqa: E501
    T(REGION_MODE, "bedrock/us-gov-east-1/claude-sonnet-4-5-20250929-v1:0", "anthropic", "claude-sonnet-4.5-20250929-v1", ("bedrock",), "claude", "us-gov-east-1/claude-sonnet-4-5-20250929-v1:0", "family"),  # noqa: E501
    # --- proxy pair -------------------------------------------------------------------------------------
    P(PROXY, "openai", "azure_ai/cohere-command-a-plus-05-2026", "cohere", "cohere-command-a-plus-05-2026", ("openai", "azure_ai"), "command", "cohere-command-a-plus-05-2026", "family"),  # noqa: E501
    P(PROXY, "openai", "azure_ai/claude-opus-4-6", "anthropic", "claude-opus-4.6", ("openai", "azure_ai"), "claude", "claude-opus-4-6", "family"),  # noqa: E501
    P(PROXY, "openai", "azure/gpt-5.4-mini", "openai", "gpt-5.4-mini", ("openai", "azure"), "gpt", "gpt-5.4-mini", "family"),  # noqa: E501
    P(PROXY, "openai", "openrouter/anthropic/claude-opus-4.7", "anthropic", "claude-opus-4.7", ("openai", "openrouter"), "claude", "anthropic/claude-opus-4.7", "spelled"),  # noqa: E501
    P(PROXY, "openai", "openrouter/qwen/qwen3-max", "qwen", "qwen3-max", ("openai", "openrouter"), "qwen", "qwen/qwen3-max", "spelled"),  # noqa: E501
    P(PROXY, "litellm_proxy", "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic", "claude-sonnet-4.5-20250929-v1", ("litellm_proxy", "bedrock"), "claude", "anthropic.claude-sonnet-4-5-20250929-v1:0", "spelled"),  # noqa: E501
    P(PROXY, "litellm_proxy", "bedrock/us.anthropic.claude-sonnet-4-6", "anthropic", "claude-sonnet-4.6", ("litellm_proxy", "bedrock"), "claude", "us.anthropic.claude-sonnet-4-6", "spelled"),  # noqa: E501
    P(PROXY, "litellm_proxy", "vertex_ai/claude-opus-4-7@default", "anthropic", "claude-opus-4.7", ("litellm_proxy", "vertex_ai"), "claude", "claude-opus-4-7@default", "family"),  # noqa: E501
    P(PROXY, "litellm_proxy", "gemini/gemini-3.5-flash", "google", "gemini-3.5-flash", ("litellm_proxy", "gemini"), "gemini", "gemini-3.5-flash", "route"),  # noqa: E501
    P(PROXY, "litellm", "openrouter/x-ai/grok-4", "x-ai", "grok-4", ("litellm", "openrouter"), "grok", "x-ai/grok-4", "spelled"),  # noqa: E501
    T(PROXY, "openai/azure_ai/cohere-command-a-plus-05-2026", "cohere", "cohere-command-a-plus-05-2026", ("openai", "azure_ai"), "command", "cohere-command-a-plus-05-2026", "family", entry="routed"),  # noqa: E501
    T(PROXY, "litellm_proxy/bedrock/converse/amazon.nova-pro-v1:0", "amazon", "nova-pro-v1", ("litellm_proxy", "bedrock"), "nova", "converse/amazon.nova-pro-v1:0", "spelled", entry="routed"),  # noqa: E501
]
# fmt: on

NORM = Normalizer(Rules.default())


def run(normalizer: Normalizer, row: Row) -> ModelName:
    if row.entry == "pair":
        assert row.provider is not None
        return normalizer.parse_pair(row.provider, row.text)
    if row.entry == "canonical":
        return normalizer.parse_canonical(row.text)
    if row.entry == "routed":
        return normalizer.parse_routed(row.text)
    return normalizer.parse(row.text)


def observed(name: ModelName) -> dict[str, object]:
    return {
        "vendor": name.vendor,
        "model": name.model,
        "route": name.route,
        "routes": name.routes,
        "family": name.family,
        "wire": name.wire,
        "routed": name.routed,
        "vendor_source": name.provenance.vendor_source,
    }


def expected(row: Row) -> dict[str, object]:
    return {
        "vendor": row.vendor,
        "model": row.model,
        "route": row.routes[0] if row.routes else None,
        "routes": row.routes,
        "family": row.family,
        "wire": row.wire,
        "routed": row.sent,
        "vendor_source": row.source,
    }


def test_the_table_covers_every_shape_at_least_three_times() -> None:
    counts = Counter(row.shape for row in ROWS)
    assert len(ROWS) >= 150
    assert {shape: counts[shape] for shape in SHAPES if counts[shape] < 3} == {}
    assert set(counts) == set(SHAPES)
    for row in ROWS:
        assert row.sent == "/".join((*row.routes, row.wire)), f"inconsistent row {row.id}"


@pytest.mark.parametrize("row", ROWS, ids=[row.id for row in ROWS])
def test_provider_shape(row: Row) -> None:
    assert observed(run(NORM, row)) == expected(row)


# providers that stay undeclared: the pair form reads them as their own vendor, parse_routed refuses them
UNDECLARED_PROVIDERS = {"ai21", "deepseek"}
ROUND_TRIP = [row for row in ROWS if row.routes and row.routes[0] not in UNDECLARED_PROVIDERS]


@pytest.mark.parametrize("row", ROUND_TRIP, ids=[row.id for row in ROUND_TRIP])
def test_routed_reads_back_to_the_same_value(row: Row) -> None:
    name = run(NORM, row)
    again = NORM.parse_routed(name.routed)
    assert again == name
    assert (again.routes, again.wire, again.routed) == (name.routes, name.wire, name.routed)


FOLDS = [
    ("text", None, "xai/grok-4", "x-ai"),
    ("pair", "xai", "grok-4", "x-ai"),
    ("text", None, "zai/glm-4.7", "z-ai"),
    ("text", None, "together_ai/zai-org/GLM-4.7", "z-ai"),
    ("text", None, "moonshot/kimi-k2.5", "moonshotai"),
    ("text", None, "mistral/mistral-large-2512", "mistralai"),
    ("text", None, "together_ai/deepseek-ai/DeepSeek-R1", "deepseek"),
    ("text", None, "watsonx/ibm/granite-3-8b-instruct", "ibm-granite"),
    ("text", None, "deepinfra/Qwen/Qwen3-32B", "qwen"),
    ("text", None, "deepinfra/NousResearch/Hermes-3-Llama-3.1-70B", "nousresearch"),
    ("text", None, "meta/muse-spark-1.2", "meta"),
    ("canonical", None, "meta/muse-spark-1.2", "meta"),
    ("text", None, "replicate/meta/llama-3-70b", "meta-llama"),
    ("text", None, "vercel_ai_gateway/meta/llama-3.3-70b", "meta-llama"),
    ("text", None, "oci/meta.llama-3.1-8b-instruct", "meta-llama"),
    ("pair", "bedrock", "meta.llama3-70b-instruct-v1:0", "meta-llama"),
    ("canonical", None, "meta/llama-3.3-70b-instruct", "meta-llama"),
    ("canonical", None, "meta/codellama-34b-instruct", "meta-llama"),
]


@pytest.mark.parametrize(
    "entry, provider, text, vendor", FOLDS, ids=[f"{e}|{p or ''}|{t}" for e, p, t, _ in FOLDS]
)
def test_vendor_folds(entry: str, provider: str | None, text: str, vendor: str) -> None:
    """Every spelling of a vendor folds to its OpenRouter slug; ``meta`` folds to ``meta-llama`` for the Llama
    families only, so Meta's own Muse stays ``meta``."""
    row = Row("fold", entry, provider, text, vendor, "", (), "", "", "")
    assert run(NORM, row).vendor == vendor


def test_vendor_folds_make_one_identity() -> None:
    assert NORM.parse("xai/grok-4") == NORM.parse("x-ai/grok-4")
    assert NORM.parse("zai/glm-4.7") == NORM.parse("z-ai/glm-4.7")
    assert NORM.parse_canonical("meta/llama-3.3-70b-instruct") == NORM.parse_canonical(
        "meta-llama/llama-3.3-70b-instruct"
    )
    assert NORM.parse("deepinfra/Qwen/Qwen3-32B").same_model(NORM.parse("deepinfra/qwen/Qwen3-32B"))


class TestVendorSource:
    def test_lookup(self) -> None:
        assert NORM.parse("openrouter/auto").provenance.vendor_source == "lookup"

    def test_bare_name_pattern(self) -> None:
        rules = Rules.from_dict({"bare_names": {"acme-*": "acme"}}, base=Rules.default())
        name = Normalizer(rules).parse("acme-large-1")
        assert (name.vendor, name.provenance.vendor_source) == ("acme", "pattern")

    def test_resolver_answers_before_the_families_table(self) -> None:
        resolver = CallableResolver(lambda bare: ["acme/claude-opus-4-7"], "catalogue-1")
        name = Normalizer(Rules.default(), resolver).parse("claude-opus-4-7")
        assert (name.vendor, name.provenance.vendor_source) == ("acme", "resolver")

    def test_an_empty_resolver_answer_falls_through_to_the_families_table(self) -> None:
        resolver = CallableResolver(lambda bare: [], "catalogue-1")
        name = Normalizer(Rules.default(), resolver).parse("claude-opus-4-7")
        assert (name.canonical, name.provenance.vendor_source) == ("anthropic/claude-opus-4.7", "family")

    def test_an_unknown_family_is_still_an_unknown_vendor(self) -> None:
        with pytest.raises(UnknownVendor):
            NORM.parse("some-unknown-model")
        with pytest.raises(UnknownVendor):
            NORM.parse("azure/some-unknown-model")
