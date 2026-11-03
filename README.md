# toloka-model-name-normalizer

One identity and rule-derived descriptive fields for every LLM model reference. Every parsing rule is
configuration: the wheel ships one default rule set (Toloka's), and any deployment overrides it from outside
without a release of the library.

Status: alpha, not yet published on PyPI. The design notes and the survey behind the golden corpus are being
prepared for publication in this repository.

## What it does

A model is spelled many ways across a stack: `openai/gpt-6-astra` in a pricing table,
`openrouter/openai/gpt-6-astra` on the wire, `platform_openai/gpt-6-astra` on a gateway, `(openrouter,
openai/gpt-6-astra)` as a provider-plus-name pair in a run config, `openrouter__openai_gpt-6-astra` as the
engine's output directory name. The library gives all of them one identity, `ModelName(vendor, model, route)`, and derives
descriptive fields from the name by rule:

| Group | Fields |
|---|---|
| identity (equality, hash) | `vendor`, `model`, `route` (`routes` keeps the whole chain) |
| static, always filled | `vendor`, `family` |
| dynamic, `None` when absent | `generation`, `tier`, `variant`, `snapshot`, `size`, `stage`, `capability`, `api_version` |

`model` is the canonical spelling: the vendors that write a minor version with a dot get it also when a name
spells a dash (`claude-opus-4-6` is `claude-opus-4.6`), and deployment detail (a Bedrock region, a Vertex
`@default`, a LiteLLM path segment) is no part of it. `wire` keeps the name exactly as it was sent after the
route chain, so `routed` reproduces the input and a provider gets what it expects.

The library carries no model catalogue. A bare name (`claude-opus-4-7`, `azure/gpt-5.5`) takes the vendor of
its family from the `[families]` table, one row per model line, after the caller's resolver has had its say;
`provenance.vendor_source` records where every vendor came from (`spelled`, `alias`, `route`, `lookup`,
`pattern`, `resolver`, `family`), and `policy.family_vendor = "off"` restores the strict reading in which a
bare name resolves only through a resolver or the pair form. The guess is a family's vendor, so a third-party
fine-tune named after its base model (`airoboros-llama-2-70b`) reads as the base model's vendor; a consumer that
attributes cost or rank counts `vendor_source = "family"` rows apart.

## Providers the default reads

The default rules follow LiteLLM's provider names, the de-facto wire convention, and fold every vendor
spelling into its OpenRouter slug. With no override and no code:

| Shape | Example | Identity |
|---|---|---|
| native API | `(anthropic, claude-opus-4-6)`, `xai/grok-4`, `zai/glm-4.7` | `anthropic/claude-opus-4.6`, `x-ai/grok-4`, `z-ai/glm-4.7` |
| OpenRouter and other hosts | `together_ai/deepseek-ai/DeepSeek-R1`, `deepinfra/Qwen/Qwen3-32B` | `deepseek/DeepSeek-R1`, `qwen/Qwen3-32B` |
| Azure OpenAI and AI Foundry | `azure/gpt-5.5`, `azure_ai/Llama-4-Scout-17B-16E-Instruct` | `openai/gpt-5.5`, `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| AWS Bedrock, OCI | `(bedrock, us.anthropic.claude-sonnet-4-5-20250929-v1:0)`, `oci/meta.llama-3.1-405b-instruct` | `anthropic/claude-sonnet-4.5-20250929-v1`, `meta-llama/llama-3.1-405b-instruct` |
| Vertex AI | `vertex_ai/claude-opus-4-7@default`, `vertex_ai/zai-org/glm-5-maas` | `anthropic/claude-opus-4.7`, `z-ai/glm-5` |
| host paths | `fireworks_ai/accounts/fireworks/models/glm-5p1`, `cloudflare/@cf/meta/llama-3.1-8b-instruct` | `z-ai/glm-5.1`, `meta-llama/llama-3.1-8b-instruct` |
| a proxy in front of a route | `(openai, openrouter/anthropic/claude-opus-4.7)`, `(litellm_proxy, bedrock/…)` | the inner route continues the chain |

The test suite measures this against the LiteLLM 1.93.0 model catalogue (`tests/data/litellm_catalogue.json`):
at least 97% of its chat models read with an allowed vendor and a real family, and every id that does not is
named with its reason. The model's case and a vendor token inside the name stay as spelled
(`snowflake/openai-gpt-5-mini` is `openai/openai-gpt-5-mini`), and a glued family and version
(`llama3-3-70b`) stays apart from the dashed one (`llama-3.3-70b`).

## Quick start

```python
from toloka_model_name_normalizer import Normalizer, Rules

norm = Normalizer(Rules.default())

name = norm.parse_pair("openrouter", "openai/gpt-6-astra")  # the task.yaml form, never lossy
name.canonical  # "openai/gpt-6-astra"
name.wire  # "openai/gpt-6-astra", as sent after the route chain
name.routed  # "openrouter/openai/gpt-6-astra"
name.slug  # "openai_gpt-6-astra"; the route is a property, not part of the slug
name.family, name.generation, name.variant  # "gpt", "6", "astra"
name.tags()  # ["model:openai/gpt-6-astra", "model_vendor:openai", "model_family:gpt"]
name.fields()  # every descriptive field with an explicit None
name.provenance  # rules version, identity and fields fingerprints, resolver fingerprint, lookup flag

norm.parse_canonical("openai/gpt-6-astra")  # route None
norm.parse_routed("litellm/openrouter/anthropic/claude-opus-5")  # routes ("litellm", "openrouter")
norm.parse("platform_openai/gpt-6-astra")  # generic: peelable route, vendor openai
```

The carrier decides the grammar, so there is one entry point per carrier:

| Entry point | Use it for |
|---|---|
| `parse_pair(provider, name)` | `task.yaml` `model_config.<role>`, `ModelConfig`, certificates, config files |
| `parse_canonical(text)` | pricing keys, `model:` tags, collector ids |
| `parse_routed(text)` | the litellm wire string, replay provenance, gateway route ids |
| `parse(text)` | anything else: lookup, then routed for a peelable route, then canonical, then the resolver |

`openai/gpt-6-astra` is canonical (route `None`) in `parse` and `parse_canonical`, and the direct-route wire
form (route `openai`) only in `parse_routed` or through the pair.

## Rules are configuration

`Rules.default()` reads the packaged `rules.toml`. Override it with one file, merged over the default
(tables replace entry by entry, vocabulary lists extend, scalars overwrite; `extends = false` starts from an
empty rule set):

```python
rules = Rules.load("/etc/toloka/model-name-rules.toml")
```

or through the environment for the module-level functions:

```bash
export TOLOKA_MODEL_NAME_NORMALIZER_RULES=/etc/toloka/model-name-rules.toml
```

```toml
# an override file
version = "acme-2026.10"

[routes.nebius]                      # a new route: openrouter-style transport namespace
[vendor_aliases]
xai = "x-ai"                         # fold a misspelling into one vendor
[bare_names]
"gpt*" = "openai"                    # ordered wildcard patterns for names that carry no vendor
[policy]
case = "fold"
[fields.vocab]
tier = ["giga"]                      # extends the default list
[fields.vendor.acme]
four_digit_snapshot = "YYMM"
[lookup."acme/thing-1"]
vendor = "acme"
model = "thing-1"
why = "the one name the grammar cannot read"
```

Rule groups: `routes`, `vendor_aliases`, `family_vendor_aliases`, `families`, `bare_names`, `lookup`,
`canonical`, `policy`, `render`, `fields.separators`, `fields.suffixes`, `fields.shapes`, `fields.dates`,
`fields.vocab`, `fields.vendor.<vendor>`. A route carries `implies_vendor`, `nestable`, `also_vendor`,
`dotted_vendor`, `regions`, `skip_segments`, `skip_paths`, `strip` and `decimal_p`; `rules.toml` says what each
one does. Schema 2 added `families`, `family_vendor_aliases`, `canonical`, the route keys after `nestable` and
the policies `vendor_case` and `family_vendor`; a schema-1 file still loads, and without these keys it reads as
schema 1 did. To go back to the strict reading of release 0.1 over the new default: `policy.family_vendor =
"off"`, `canonical.minor_dot = []`, `policy.vendor_case = "preserve"`, a route restated or removed with
`remove_routes`, an alias mapped to itself (`xai = "xai"`). The schema is
closed and checked on the override file itself: an unknown key at any level, a wrong type, an unsupported
`schema_version`, a `lookup` entry without `why`, a regex that matches the empty string or a shape regex that
does not fit its kind's token layout is a `RulesError`. Rules files are trusted input all the same: a pathological
regex in a rules file can make parsing slow, so load rules only from sources you control.
Every rule set has two fingerprints, `identity_fingerprint` (what decides vendor, model, route, family) and
`fields_fingerprint` (everything else); every value carries both in its `provenance`, so consumers record
which rules produced a value.

## Resolver

```python
from toloka_model_name_normalizer import CallableResolver, Normalizer, Rules

catalogue = CallableResolver(
    lambda bare: ["meta/muse-spark-1.1"] if bare == "muse-spark-1.1" else [], fingerprint="catalogue-2026-09"
)
Normalizer(Rules.default(), resolver=catalogue).parse("muse-spark-1.1").canonical  # "meta/muse-spark-1.1"
```

The resolver answers before the families table. One candidate resolves, several raise `AmbiguousModelName`,
none falls through to the families table, which raises `UnknownVendor` when it has no row for the family (or
always, under `policy.family_vendor = "off"`); the resolver's fingerprint lands in the provenance.

## Command line

```bash
python -m toloka_model_name_normalizer parse openrouter/openai/gpt-6-astra
python -m toloka_model_name_normalizer pair meta muse-spark-1.2
python -m toloka_model_name_normalizer --rules override.toml rules
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

To eyeball what the library makes of any input (identity, fields, renderings or the error), run the demo script,
with the built-in list or your own inputs (`pair:<provider>:<name>`, `canonical:<text>`, `routed:<text>`, or a
plain string for the generic entry point; `--markdown` for a table). It falls back to the sources next to it
when the package is not installed, so it needs no `PYTHONPATH`, no `uv sync` and no network:

```bash
python3 scripts/show_outputs.py
python3 scripts/show_outputs.py x-ai/grok-4.6 "pair:meta:muse-spark-1.2"
uv run --no-sync python scripts/show_outputs.py       # with uv, without syncing the dev dependencies
```

`tests/data/golden.json` is the golden corpus generated from the survey (every raw string with its input kind
and expectation); the test suite asserts every field of every row. The maintainers regenerate it
from the survey corpus when the rules change and review the diff together with the change.

Python 3.10 and later; `tomli` is the single dependency below 3.11. Licence: Apache-2.0, see `LICENSE`.
