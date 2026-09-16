#!/usr/bin/env python3
"""Show what the normalizer makes of a list of inputs: identity, descriptive fields, renderings, or the error.

Run from the repository root with any Python 3.10+. The sources next to this script are used when the package
is not installed, so no PYTHONPATH and no network are needed; with uv, --no-sync skips the dependency sync:

    python3 scripts/show_outputs.py                                # the built-in inputs, plain text
    python3 scripts/show_outputs.py --markdown                     # the same as a Markdown table
    python3 scripts/show_outputs.py x-ai/grok-4.6 "pair:meta:muse-spark-1.2" "routed:openai/o3"
    uv run --no-sync python scripts/show_outputs.py

Input syntax on the command line: a plain string goes through ``parse()``; ``pair:<provider>:<name>``
through ``parse_pair()``; ``canonical:<text>`` and ``routed:<text>`` through those entry points.
``--rules PATH`` uses an override rules file merged over the packaged default.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

# Deliberately below the declared minimum: an older interpreter must get this message instead of a TypeError
# from the kw_only dataclass fields in model.py.
if sys.version_info < (3, 10):  # noqa: UP036
    sys.exit("toloka-model-name-normalizer needs Python 3.10 or later; this is " + sys.version.split()[0])

try:
    import toloka_model_name_normalizer  # noqa: F401  (installed package)
except ImportError:  # a checkout without an installed package: use the sources next to this script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toloka_model_name_normalizer import ModelName, ModelNameError, Normalizer, Rules  # noqa: E402
from toloka_model_name_normalizer.model import DESCRIPTIVE_FIELDS  # noqa: E402

# (kind, provider or None, input); one line per case, grouped the way the design discusses them
BUILT_IN: list[tuple[str, str | None, str]] = [
    # the frontier line-up as task.yaml pairs
    ("pair", "openrouter", "openai/gpt-6-astra"),
    ("pair", "openrouter", "openai/gpt-5.6-terra"),
    ("pair", "openrouter", "openai/gpt-5.4-pro"),
    ("pair", "openrouter", "anthropic/claude-opus-5"),
    ("pair", "openrouter", "anthropic/claude-fable-5.1"),
    ("pair", "anthropic", "claude-opus-4-6"),
    ("pair", "openrouter", "google/gemini-3.7-flash"),
    ("pair", "openrouter", "google/gemma-4-31b-it"),
    ("pair", "openrouter", "x-ai/grok-4.20"),
    ("pair", "openrouter", "deepseek/deepseek-v4-flash-0731"),
    ("pair", "openrouter", "moonshotai/kimi-k2.7-code"),
    ("pair", "openrouter", "qwen/qwen3.8-max"),
    ("pair", "meta", "muse-spark-1.2"),
    ("pair", "openrouter", "z-ai/glm-5.3"),
    ("pair", "openrouter", "minimax/minimax-m3"),
    ("pair", "openrouter", "tencent/hy3"),
    ("pair", "openrouter", "openai/gpt-oss-120b"),
    ("pair", "openrouter", "amazon/nova-2-lite-v1"),
    ("pair", "nova", "nova-pro-v1"),
    ("pair", "xai", "x-ai/grok-4"),
    # the same identity on different carriers
    ("text", None, "openai/gpt-6-astra"),
    ("canonical", None, "openai/gpt-6-astra"),
    ("routed", None, "openai/gpt-6-astra"),
    ("text", None, "openrouter/openai/gpt-6-astra"),
    ("text", None, "platform_openai/gpt-6-astra"),
    ("text", None, "litellm/openrouter/anthropic/claude-opus-5"),
    ("text", None, "/openrouter/openai/gpt-6-astra"),
    ("text", None, "openrouter/auto"),
    ("pair", "openrouter", "auto"),
    # hard cases of the field grammar
    ("text", None, "openai/gpt-4o"),
    ("text", None, "openai/o3-mini-high"),
    ("text", None, "anthropic/claude-3-5-sonnet-20241022"),
    ("text", None, "anthropic/claude-sonnet-4-6"),
    ("text", None, "cohere/command-a-plus"),
    ("text", None, "cohere/command-r7b-12-2024"),
    ("text", None, "google/gemini-2.5-flash-lite-preview-09-2025"),
    ("text", None, "openai/gpt-4o-2024-05-13"),
    ("text", None, "mistralai/mistral-large-2512"),
    ("text", None, "nvidia/nemotron-3-ultra-550b-a55b"),
    ("text", None, "anthropic/claude-3.7-sonnet:thinking"),
    ("text", None, "x-ai/grok-4:nitro"),
    ("text", None, "x-ai/grok-4.1-fast-non-reasoning"),
    ("text", None, "nousresearch/hermes-3-llama-3.1-405b"),
    ("text", None, "azure_ai/cohere-command-a-plus-05-2026"),
    ("text", None, "OpenAI/GPT-6-Astra"),
    # inputs the grammar refuses or cannot place
    ("text", None, "muse-spark-1.1"),
    ("text", None, "openrouter__openai_gpt-5.4"),
    ("text", None, "foo/bar/baz"),
    ("text", None, "openai/gpt-6-astra "),
    ("pair", "openai", "anthropic/claude-opus-4-6"),
    ("pair", "openrouter", "platform_openai/gpt-6-astra"),
    ("routed", None, "deepseek/deepseek-v4"),
    ("canonical", None, "platform_openai/gpt-6-astra"),
    ("text", None, "~openai/gpt-latest"),
]


def run(normalizer: Normalizer, kind: str, provider: str | None, text: str) -> ModelName:
    if kind == "pair":
        assert provider is not None
        return normalizer.parse_pair(provider, text)
    if kind == "canonical":
        return normalizer.parse_canonical(text)
    if kind == "routed":
        return normalizer.parse_routed(text)
    return normalizer.parse(text)


def describe_input(kind: str, provider: str | None, text: str) -> str:
    return f"({provider}, {text})" if kind == "pair" else text


def fields_line(name: ModelName) -> str:
    parts = [
        f"{key}={value}" for key, value in name.fields().items() if key != "vendor" and value is not None
    ]
    if name.transport_suffix:
        parts.append("transport_suffix=True")
    return " ".join(parts)


def print_text(normalizer: Normalizer, cases: Iterable[tuple[str, str | None, str]]) -> None:
    for kind, provider, text in cases:
        head = f"{kind:9s} {describe_input(kind, provider, text)}"
        try:
            name = run(normalizer, kind, provider, text)
        except ModelNameError as exc:
            print(f"{head}\n    -> {type(exc).__name__}: {exc}\n")
            continue
        routes = ",".join(name.routes) or "-"
        print(head)
        print(f"    -> {name.canonical}   route={name.route}  routes={routes}")
        print(f"       {fields_line(name)}")
        print(f"       routed={name.routed}   slug={name.slug}")
        if name.unresolved:
            print(f"       unresolved={list(name.unresolved)}")
        if name.provenance.lookup:
            print("       lookup=True")
        print()


def print_markdown(normalizer: Normalizer, cases: Iterable[tuple[str, str | None, str]]) -> None:
    print("| kind | input | canonical | route | fields | routed | slug or error |")
    print("|---|---|---|---|---|---|---|")
    for kind, provider, text in cases:
        shown = describe_input(kind, provider, text).replace("|", "\\|")
        try:
            name = run(normalizer, kind, provider, text)
        except ModelNameError as exc:
            print(f"| {kind} | `{shown}` | | | | | `{type(exc).__name__}` |")
            continue
        extra = f" unresolved={list(name.unresolved)}" if name.unresolved else ""
        print(
            f"| {kind} | `{shown}` | `{name.canonical}` | {name.route or ''} | {fields_line(name)}{extra} "
            f"| `{name.routed}` | `{name.slug}` |"
        )


def parse_cli_input(raw: str) -> tuple[str, str | None, str]:
    if raw.startswith("pair:"):
        _, provider, text = raw.split(":", 2)
        return "pair", provider, text
    for kind in ("canonical", "routed", "text"):
        if raw.startswith(kind + ":"):
            return kind, None, raw[len(kind) + 1 :]
    return "text", None, raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="*", help="inputs to show instead of the built-in list")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--rules", help="override rules file merged over the packaged default")
    args = parser.parse_args(argv)
    rules = Rules.load(args.rules) if args.rules else Rules.default()
    normalizer = Normalizer(rules)
    cases = [parse_cli_input(raw) for raw in args.inputs] or BUILT_IN
    identity, fields = rules.identity_fingerprint[:19], rules.fields_fingerprint[:19]
    print(f"rules {rules.version}  identity {identity}  fields {fields}")
    print(f"descriptive fields: {', '.join(DESCRIPTIVE_FIELDS)}\n")
    if args.markdown:
        print_markdown(normalizer, cases)
    else:
        print_text(normalizer, cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
