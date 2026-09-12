# toloka-model-name-normalizer

One identity and rule-derived descriptive fields for every LLM model reference. Every parsing rule is
configuration: the wheel ships one default rule set (Toloka's), and any deployment overrides it from outside
without a release of the library.

Status: alpha, not yet published. Design, survey and golden corpus: the workspace documents under
`docs/model-name-canonicaliser/` (`DESIGN.md`, `CORPUS.md`, `INVENTORY.md`, two design reviews).

## What it does

A model is spelled many ways across a stack: `openai/gpt-6-astra` in a pricing table,
`openrouter/openai/gpt-6-astra` on the wire, `platform_openai/gpt-6-astra` on a gateway, `(openrouter,
openai/gpt-6-astra)` as a provider-plus-name pair in a run config, `openrouter__openai_gpt-6-astra` as a
filesystem slug. The library gives all of them one identity, `ModelName(vendor, model, route)`, and derives
descriptive fields from the name by rule:

| Group | Fields |
|---|---|
| identity (equality, hash) | `vendor`, `model`, `route` (`routes` keeps the whole chain) |
| static, always filled | `vendor`, `family` |
| dynamic, `None` when absent | `generation`, `tier`, `variant`, `snapshot`, `size`, `stage`, `capability`, `api_version` |

The library carries no model catalogue: a bare name (`muse-spark-1.1`) is never guessed, it resolves only
through a caller-supplied resolver or the pair form, and fails loudly otherwise.

## Quick start

```python
from toloka_model_name_normalizer import Normalizer, Rules

norm = Normalizer(Rules.default())

name = norm.parse_pair("openrouter", "openai/gpt-6-astra")   # the task.yaml form, never lossy
name.canonical        # "openai/gpt-6-astra"
name.routed           # "openrouter/openai/gpt-6-astra"
name.slug             # "openrouter__openai_gpt-6-astra"
name.family, name.generation, name.variant   # "gpt", "6", "astra"
name.tags()           # ["model:openai/gpt-6-astra", "model_vendor:openai", "model_family:gpt"]
name.fields()         # every descriptive field with an explicit None
name.provenance       # rules version, identity and fields fingerprints, resolver fingerprint, lookup flag

norm.parse_canonical("openai/gpt-6-astra")                    # route None
norm.parse_routed("litellm/openrouter/anthropic/claude-opus-5")  # routes ("litellm", "openrouter")
norm.parse("platform_openai/gpt-6-astra")                     # generic: peelable route, vendor openai
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

Rule groups: `routes`, `vendor_aliases`, `bare_names`, `lookup`, `policy`, `render`, `fields.separators`,
`fields.suffixes`, `fields.shapes`, `fields.dates`, `fields.vocab`, `fields.vendor.<vendor>`. The schema is
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

catalogue = CallableResolver(lambda bare: ["meta/muse-spark-1.1"] if bare == "muse-spark-1.1" else [],
                             fingerprint="catalogue-2026-09")
Normalizer(Rules.default(), resolver=catalogue).parse("muse-spark-1.1").canonical   # "meta/muse-spark-1.1"
```

Zero candidates raise `UnknownVendor`, several raise `AmbiguousModelName`; the resolver's fingerprint lands in
the provenance.

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

`tests/data/golden.json` is the golden corpus generated from the survey (every raw string with its input kind
and expectation); the test suite asserts every field of every row. Regenerate it from
`docs/model-name-canonicaliser/fixtures/` in the workspace when the rules change, and review the diff.

Python 3.10 and later; `tomli` is the single dependency below 3.11. Licence: to be decided before the first
release (Apache-2.0 proposed).
