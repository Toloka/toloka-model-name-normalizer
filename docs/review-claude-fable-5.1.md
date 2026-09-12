# Code review: toloka-model-name-normalizer, first implementation

Reviewer: claude-fable-5.1
Date: 2026-09-12
Commit reviewed: `1cb34ff` (the single commit of the repository)
Design reviewed against: `docs/model-name-canonicaliser/DESIGN.md` in the workspace (718 lines, both review logs), plus
`fixtures/README.md`, `fixtures/make_fixture.py` and the compound block of `fixtures/proto.py`.

Files reviewed (every line): `pyproject.toml`, `README.md`, `.gitignore`,
`src/toloka_model_name_normalizer/{__init__,__main__,errors,fields,identity,model,normalizer,render,resolver,rules}.py`,
`src/toloka_model_name_normalizer/rules.toml`, `src/toloka_model_name_normalizer/py.typed`,
`tests/{test_cli,test_equality,test_fields,test_golden,test_identity,test_render,test_rules}.py`,
`tests/data/golden.json` (structure and all error, review, routed, pair and unresolved rows), `tests/data/coverage.json`.

## Verdict

Changes required before this is mergeable as a first alpha. The core is right: the identity grammar, the four entry
points, exact equality on `(vendor, model, route)`, the `routes` chain and the `routed` round trip, `same_model` and
`same_wire`, `tags()` and `fields()`, the provenance record, the identity and fields fingerprint split, and the
policies all behave as DESIGN.md says on the default rules, and every one of the 928 golden rows passes; ruff, ruff
format and strict mypy are clean, and an offline hatchling build ships `rules.toml`, `py.typed` and the console script.
Three findings are P1: the closed rules schema is not enforced inside `fields` overrides (typos are silently dropped,
which is exactly the misconfiguration the design says must fail loudly), the compound vocabulary is matched as a
substring and leaves a NUL byte in `variant` for plausible names, and `parse_pair` accepts a peelable route in vendor
position although the review log records that check as adopted for every entry point. Six P2 findings should land
before the first PyPI release; the fingerprint one (F4) in particular, because fingerprints are recorded in traces
and cannot be changed later without a migration.

## Findings

### F1 (P1) The closed schema is not enforced for `fields` overrides; list-valued keys are not type-checked

Location: `rules.py:327-346` (`_merge`), `rules.py:202` (only the top level of the override is key-checked),
`rules.py:446-495` (`_validate` runs on the merged document, after the override's unknown keys are gone).

Problem: `_merge` copies only the keys it knows out of an override's `fields` and `fields.vocab` tables. An unknown
key never reaches `_validate`, so a typo in a deployment override is silently ignored. DESIGN.md 213-216 and README
line 99 promise a `RulesError` for any unknown key. The vocabulary merge also iterates the override value without
checking that it is a list, so a string is spread into single characters.

Evidence (all against `base=Rules.default()`):

```
Rules.from_dict({"fields": {"vocabulary": {"tier": ["giga"]}}})   -> Rules built, no error
Rules.from_dict({"fields": {"vocab": {"tiers": ["giga"]}}})       -> Rules built, no error
Rules.from_dict({"fields": {"suffix": {"x": {"field": "variant"}}}}) -> Rules built, no error
Rules.from_dict({"fields": {"vendors": {"acme": {}}}})            -> Rules built, no error
Rules.from_dict({"fields": "x"})                                  -> Rules built, no error
Rules.from_dict({"fields": {"vocab": "x"}})                       -> Rules built, no error
Rules.from_dict({"fields": {"vocab": {"tier": "giga"}}}).fields.tier[-4:] -> ('g', 'i', 'g', 'a')
```

Proposed change: validate the override document itself before merging: walk it with the same allowed-key tables
`_validate` uses (`_TOP_KEYS`, `_FIELDS_KEYS`, `_VOCAB_KEYS`, `_VENDOR_SWITCHES`, the per-route and per-suffix key
sets) and type-check every list-valued key (`fields.vocab.{tier,stage,capability}`, `fields.shapes.<kind>`,
`fields.dates.<name>.tokens`, `letter_generations`, `remove_routes`) and every table-valued key. Alternatively make
`_merge` generic (deep-merge unknown keys through) so that `_validate` sees and rejects them.

Acceptance test: `pytest.raises(RulesError, match="unknown key")` for each of the six documents above;
`pytest.raises(RulesError)` for `{"fields": {"vocab": {"tier": "giga"}}}` and for `{"remove_routes": "mock"}`.

### F2 (P1) Compounds are matched as substrings of the joined token string; a NUL byte survives in `variant`

Location: `fields.py:144-150`.

Problem: `compound in joined` tests a substring of `"-".join(rest)`, not a run of whole tokens, and the `"\x00"`
replacement marker survives when the match falls inside a token. The classification is wrong (a capability that is not
there) and the value carries a control character into `fields()`, `to_dict()` and therefore Langfuse metadata.
Strict mode does not catch it because `unresolved` stays empty.

Evidence:

```
parse("openai/o3-deep-researcher").fields()  -> capability='deep-research', variant='\x00er', unresolved=()
parse("x-ai/grok-5-multi-agents").fields()   -> variant='multi-agent-\x00s'
parse("x-ai/grok-encode-faster").fields()    -> capability='code-fast', variant='en\x00er'
json.dumps(parse("openai/o3-deep-researcher").to_dict()["fields"]["variant"]) -> "\u0000er"
Normalizer(Rules.default().with_policy(fields="strict")).parse("x-ai/grok-5-multi-agents") -> no error
```

Proposed change: match compounds on token boundaries: for each compound, slide a window of
`len(compound.split("-"))` over `rest` and compare the token tuple; remove exactly those tokens. Drop the `\x00`
marker. The prototype (`fixtures/proto.py`, the `for comp, field in F["vocab"]["compounds"]` block) has the same code
and should be fixed with it; `tests/data/golden.json` contains no row with `\u0000`, so the fixture needs no
regeneration.

Acceptance test: `parse("openai/o3-deep-researcher").fields()` gives `capability=None, variant="deep-researcher"`;
`"\x00" not in json.dumps(parse(text).to_dict())` for the three inputs above; the golden rows `openai/o3-deep-research`
and `x-ai/grok-code-fast-1` are unchanged.

### F3 (P1) `parse_pair` accepts a peelable route in vendor position

Location: `identity.py:205-211` (declared-route branch), `identity.py:217-219` (unknown-provider branch),
`identity.py:111` (resolver candidates reach `_finish` unchecked). The check exists only in `parse_canonical`
(`identity.py:123-125`) and `parse_routed` (`identity.py:161-163`).

Design: DESIGN.md 360 ("a peelable route left in vendor position ... MalformedName"), 715 (Fable F17 disposition:
"repeated route and peelable-route-as-vendor rejected in every entry point"), 391 ("the identity parser has no
best-effort mode").

Evidence:

```
parse_pair("openrouter", "platform_openai/gpt-6-astra") -> vendor='platform_openai', canonical='platform_openai/gpt-6-astra',
                                                          routed='openrouter/platform_openai/gpt-6-astra'
parse_routed("openrouter/platform_openai/gpt-6-astra")   -> MalformedName (golden row; the same wire string)
parse_pair("openrouter", "litellm/gpt-6-astra")          -> vendor='litellm'
parse_pair("xai", "openrouter/x")                        -> vendor='openrouter'
resolver candidate "openrouter/gpt-x" for bare "gpt-x"   -> vendor='openrouter'
resolver candidate "platform_openai/gpt"                 -> vendor='platform_openai'
```

Proposed change: move the check into `_finish` (`identity.py:72-89`), which every entry point and the resolver path
already go through: `route = _route(rules, vendor); if route is not None and _peelable(rules, route): raise
MalformedName(...)`. Delete the two per-entry-point copies.

Acceptance test: `pytest.raises(MalformedName)` for the three pairs above and for a `CallableResolver` returning
`["openrouter/gpt-x"]`; every golden row still passes.

### F4 (P2) `identity_fingerprint` is blind to `bare_names` order and sensitive to lookup-key case

Location: `rules.py:510-511` (`_canonical_json` with `sort_keys=True`), `rules.py:514-534`.

Design: DESIGN.md 441-443 (canonical form: "lists in post-merge order"), 427-428 and 447-450 (same rules, same
result; consumers record the fingerprint to audit values). `bare_names` is an ordered table (first match in file
order, `identity.py:96-98`; `tests/test_identity.py:226-234`), but JSON serialisation with sorted keys erases the
order.

Evidence:

```
a = Rules.from_dict({"bare_names": {"*llama*": "meta-llama", "deepseek*": "deepseek"}}, base=D)
b = Rules.from_dict({"bare_names": {"deepseek*": "deepseek", "*llama*": "meta-llama"}}, base=D)
a.identity_fingerprint == b.identity_fingerprint                    -> True
Normalizer(a).parse("deepseek-r1-distill-llama-70b").vendor         -> 'meta-llama'
Normalizer(b).parse("deepseek-r1-distill-llama-70b").vendor         -> 'deepseek'
Rules.from_dict({"lookup": {"acme/x": ...}}) vs {"lookup": {"ACME/X": ...}}: different fingerprints, identical parses
```

Proposed change: in `fingerprints`, serialise `bare_names` as an ordered list of `[pattern, vendor]` pairs in
post-merge order, and lower-case lookup keys (and route names, see F6) before hashing. Land it before the first
fingerprint is recorded anywhere; DESIGN.md 516 makes a later change a trace migration.

Acceptance test: the two `bare_names` documents above hash differently; `acme/x` and `ACME/X` hash identically;
`TestFingerprints` still passes.

### F5 (P2) Rules loading leaks raw `KeyError`, `ValueError`, `TypeError` and `TOMLDecodeError`

Location: `rules.py:250` (`fields_doc["vocab"]`) and `rules.py:258` (`fields_doc["shapes"]`) versus `_validate`
`rules.py:456` and `rules.py:470`, which treat both as optional; `rules.py:319-326` and `rules.py:343-346` (`_merge`
calls `.update()` on whatever the override holds); `rules.py:187-188` (`tomllib.load` unguarded); `__main__.py:35-38`
(catches only `RulesError` and `OSError`).

Design: DESIGN.md 589-591 (`RulesError` is the error for a rules file), 460-461 ("fails loudly as RulesError").

Evidence:

```
extends=false document with complete policy + render and fields = {separators = "[-/]+"}   -> KeyError: 'vocab'
same plus fields.shapes = {}                                                              -> KeyError: 'vocab'
Rules.from_dict({"routes": "openrouter"}, base=D)  -> ValueError: dictionary update sequence element #0 has length 1
  (same for policy = "fold", lookup = "x", fields.vendor.acme = "x", fields.vocab.compounds = "x")
Rules.from_dict({"remove_routes": 5}, base=D)      -> TypeError: 'int' object is not iterable
Rules.load(<file containing: this is = not toml = [>) -> tomllib.TOMLDecodeError
main(["--rules", that_file, "parse", "openai/gpt"])  -> traceback, not exit code 2
main(["--rules", <file: routes = "x">, "parse", "openai/gpt"]) -> traceback (ValueError)
```

Proposed change: make `shapes` and `vocab` consistently optional (`fields_doc.get("shapes", {})`,
`fields_doc.get("vocab", {})`) or consistently required in `_validate`; type-check the override before `_merge`
(every table-valued key must be a `Mapping`, `remove_routes` a list of `str`); wrap `tomllib.load`/`loads` in
`Rules.load` and `Rules.default` and re-raise as `RulesError(f"{path}: {exc}")`.

Acceptance test: every input above raises `RulesError`; the CLI returns 2 and prints `rules error:` for the malformed
file and for `routes = "x"`.

### F6 (P2) Route names and `implies_vendor` are not case-normalised, unlike every other rule table

Location: `rules.py:220-227` (`_build` keeps route keys as written), `rules.py:380-400` (`_validate` compares as
written), `rules.py:87` (`peelable` compares as written), `identity.py:54-57` (`_route` lower-cases the input and
looks it up in that dict), `render.py:20`.

Design: DESIGN.md 110 and 417 ("rule tables are matched case-insensitively"), 213-214 (a duplicate route is
rejected). `vendor_aliases`, `bare_names`, `lookup` and `fields.vendor` are lower-cased at build (`rules.py:229, 274,
289, 291`); routes are not.

Evidence:

```
r = Rules.from_dict({"routes": {"OpenRouter": {"nestable": True}}}, base=D)
[k for k in r.routes if k.lower() == "openrouter"]        -> ['openrouter', 'OpenRouter']   (a case-duplicate route)
Normalizer(r).parse_routed("OpenRouter/openai/gpt").routes -> ('openrouter',)  (the base entry; the override is ignored)
Rules.from_dict({"routes": {"openai": {"implies_vendor": "OpenAI"}}}, base=D)
  -> RulesError "routes.platform_openai.implies_vendor names 'openai', which is itself a peelable route"
     (the right outcome for the wrong reason: "OpenAI" != "openai" makes routes.openai peelable)
```

Proposed change: lower-case route names and `implies_vendor` in `_build` and `_validate`, reject two routes that
collide after folding, and hash the folded names (F4).

Acceptance test: `{"routes": {"OpenRouter": {"nestable": True}}}` either raises `RulesError` or yields
`rules.routes["openrouter"].nestable is True`; `{"routes": {"acme": {"implies_vendor": "Acme"}}}` gives
`rules.routes["acme"].peelable is False`.

### F7 (P2) `parse_routed` skips the lookup stage when the remainder after peeling has two segments

Location: `identity.py:149-156` (lookup only when `len(parts) == 1`), `identity.py:159-169` (the two-segment branch
never consults `rules.lookup`).

Design: DESIGN.md 360 ("lookup on innermost_route/remainder after peeling"), 380 ("in every entry point"), 699
(Fable F1 disposition).

Evidence with an override `lookup."openrouter/acme/thing" = {vendor="acme", model="thing-x", why="t"}`:

```
parse("openrouter/acme/thing")                -> acme/thing-x, lookup=True
parse_pair("openrouter", "acme/thing")        -> acme/thing-x, lookup=True, route openrouter
parse_routed("openrouter/acme/thing")         -> acme/thing,   lookup=False
parse_routed("litellm/openrouter/acme/thing") -> acme/thing,   lookup=False
```

The shipped entries are single-segment (`openrouter/auto`), so the golden fixture does not notice.

Proposed change: after the peeling loop and before the two-segment branch, call
`_lookup(rules, f"{chain[-1]}/{'/'.join(parts)}", chain[0], tuple(chain))`.

Acceptance test: the four calls above agree on `acme/thing-x` with `provenance.lookup is True`.

### F8 (P2) `family` can be empty or non-alphabetic although the design declares it static and `[a-z]+`

Location: `fields.py:137-142` (family is the first token whatever it is, `""` when there is none),
`identity.py:41-51` (the model grammar admits `-`, `(`, `:thinking`), `model.py:117-123` (`tags()` emits it),
`rules.py:234` (a lookup entry may clear `family` to `""`).

Design: DESIGN.md 74 (`family: str`, "filled for 100% of the corpus"), 118-122 (grammar `[a-z]+`, "non-empty by
construction as long as the model name starts with a letter").

Evidence:

```
parse("openai/-").tags()          -> ['model:openai/-', 'model_vendor:openai', 'model_family:']   unresolved=('empty',)
parse("openai/(")  and  parse("openai/:thinking")  -> family '' (and the suffix is never classified: early return at fields.py:139)
parse("openai/2024").family -> '2024';  parse("a/1").family -> '1';  parse("openai/gpt.4").family -> 'gpt.4'
lookup entry family = "" -> tags() ends with 'model_family:'
```

Proposed change: decide one of (a) tighten the model grammar to require at least one ASCII alphanumeric character
(rejects `-`, `(`, `:thinking` as `MalformedName`) and document that a name starting with a digit takes its first
token as family, or (b) make `family: str | None`. In both cases refuse `family = ""` in lookup validation and update
DESIGN.md 119.

Acceptance test: `parse("openai/-")` raises `MalformedName` (option a); a property test over the golden inputs and
these three asserts that no `tags()` element is `model_family:` with an empty value.

### F9 (P2) Release-readiness gaps: licence, CI, version duplication, README references

Location: `pyproject.toml` (no `license`, no `[project.urls]`, `version = "0.1.0"` at line 3 duplicated by
`__init__.py:27`), `README.md:7-8` and `README.md:134-136` (point at workspace-internal paths), `README.md:138-139`
(licence to be decided), no CI configuration in `git ls-files`, sdist ships `tests/data/golden.json` (800 KB).

Design: DESIGN.md 56-60 ("py.typed and a strict type check in CI", conditional `tomli`), 675 (licence open).

Evidence: the offline hatchling 1.31 build (`/tmp/fable-code-review/dist`) produces a wheel whose METADATA carries no
`License` or `License-Expression`; `git ls-files` lists no `.github/` or other CI file; Python 3.10, the only
interpreter that exercises `import tomli as tomllib` (`rules.py:21-22`), has never been run against this code (the
`.mypy_cache/3.10` directory is mypy's target version, not an interpreter run).

Proposed change: `license = "Apache-2.0"` (or the licence Balázs decides) plus a `LICENSE` file; `[project.urls]`;
one source of truth for the version (`dynamic = ["version"]` with `[tool.hatch.version] path =
"src/toloka_model_name_normalizer/__init__.py"`); a CI workflow running ruff, mypy and pytest on 3.10 through 3.13; a
README that stands alone on PyPI (link to the public repository, not to `docs/model-name-canonicaliser/`); consider
dropping `tests` from the sdist or keeping it deliberately.

Acceptance test: METADATA has a licence; CI green on 3.10; `importlib.metadata.version("toloka-model-name-normalizer")
== toloka_model_name_normalizer.__version__`.

### F10 (P3) `dash_minor_version` merges a bare generation with a following snapshot token

Location: `fields.py:163-173` (only the first token's length is checked).

Evidence:

```
parse("anthropic/claude-3-20240229").fields()     -> generation '3.20240229', snapshot None
parse("mistralai/mistral-small-3-2512").fields()  -> generation '3.2512', snapshot None
parse("anthropic/claude-3-opus-20240229")         -> generation '3', snapshot '20240229'   (the tier separates them)
```

Proposed change: also require that the second token is not a snapshot shape
(`_shape(rules, rest[index + 1]) not in ("snapshot4", "snapshot8")`).

Acceptance test: the two inputs give generation `3` and snapshot `20240229` / `2512`; the golden rows
`claude-3-5-sonnet-20241022` and `mistral-medium-3-5` are unchanged.

### F11 (P3) Lookup entries bypass the identity grammar and the case policy; lookup values are not validated

Location: `identity.py:64-69` (`_lookup` returns an `Identity` without going through `_finish`), `rules.py:422-431`
(no value validation), `fields.py:253` (field half looked up by canonical only).

Evidence:

```
policy.case = "reject" + lookup."x/y" = {vendor="X", model="Y z/w"}: parse("x/y") -> vendor 'X', model 'Y z/w', no error
policy.case = "fold"   + the same entry:                              parse("x/y").canonical -> 'X/Y'
lookup."acme/thing" = {vendor="acme", model="a/b"}: parse("acme/thing").canonical -> 'acme/a/b';
  parse_canonical("acme/a/b") -> MalformedName (the value cannot be read back)
lookup."a/b".tier = 5 accepted (ModelName.tier becomes an int); lookup."a/b".vendor = "~x" accepted; key "" accepted
lookup."openrouter/openai/gpt-6-astra" = {tier="giga"}: never applies on any carrier (field lookup is by canonical), silently
```

Proposed change: validate at load that lookup `vendor` matches the segment grammar, `model` passes the model grammar,
field values are strings and keys are non-empty; apply the case policy to lookup identities (or document that the
entry's spelling is authoritative); reject or warn on fields-only keys that are not `vendor/model`-shaped.

Acceptance test: each bad entry raises `RulesError`; under `case = "fold"` a lookup identity is folded.

### F12 (P3) Shape kinds trust the regex to produce a fixed token layout; degenerate regexes are accepted

Location: `fields.py:186-192` (`token[:4]`, `token[4:6]`, `token[6:]`), `fields.py:120-122`; `rules.py:444, 449,
456-461` (compile-only validation of `render.unsafe`, `fields.separators`, `fields.shapes`).

Evidence:

```
override fields.shapes.snapshot8 = ['^\d{6}$'];  parse("openai/gpt-4o-202410") -> ValueError: invalid literal for int() with base 10: ''
override fields.shapes.snapshot4 = ['^\d{3}$'];  parse("openai/gpt-4o-123").snapshot -> '123' (accepted as MMDD)
override fields.separators = "x*" (matches the empty string): family 'g', variant 'p-t-----a-s-t-r-a'
override render.unsafe = "": slug 'unknown__o_p_e_n_a_i_/_g_p_t_-_6_-_a_s_t_r_a' (the slash survives)
override fields.shapes.generation = ['^(a+)+$']: 1.96 s for one 27-character token (exponential backtracking)
```

Proposed change: validate at load that `separators` and `unsafe` do not match the empty string and that each
`snapshot8`, `snapshot4` and `api3` pattern only matches tokens of the documented length (probe the compiled pattern);
state in the README that rules files are trusted input.

Acceptance test: the first four overrides raise `RulesError`; `classify` never raises a bare `ValueError`.

### F13 (P3) Error types, messages and grammar edges

- `identity.py:135-137`: `parse_routed("")` and `parse_routed("openrouter")` raise `UnknownRoute: first segment is
  not a declared route: 'openrouter'`; the segment is a declared route and the model is missing, so `MalformedName`
  (empty model) is the right type.
- `identity.py:159-160`: `parse_routed("openrouter/openrouter/openai/gpt")` reports "slash inside the model name";
  DESIGN.md 360 names this case "repeated route".
- `identity.py:192` with `identity.py:25-33`: `_text` applies the `leading_slash` tolerance to the provider, so
  `parse_pair("/openrouter", "x/y")` and `parse_pair("//openai", "gpt")` succeed against the provider grammar
  (DESIGN.md 367).
- `identity.py:25-33`: the whole input is stripped, so `parse_pair("openrouter", " openai/gpt-6-astra ")` and a name
  ending in U+00A0 succeed, while DESIGN.md 363 and 586 list a "whitespace-padded model" as `MalformedName`. Decide
  (trim or reject) and make the design and the errors list say the same thing.
- `fields.py:134`: `x-ai/grok-4:nitro:free` gives suffix and variant `nitro:free`, outside the variant grammar
  (DESIGN.md 149); either split on every colon or reject a second colon.
- `normalizer.py:46`: a resolver object without a `fingerprint` attribute fails at parse time with `AttributeError`;
  use `getattr(self.resolver, "fingerprint", None)`.
- `resolver.py:33`: `CallableResolver(lambda b: "meta/muse")` yields `AmbiguousModelName: ... m, e, t, a, /, m, u,
  s, e`; reject a `str` result.
- `identity.py:36-38`: "vendor charset: 'öpenai/gpt'" could state the grammar; `UnknownRoute` could list the declared
  routes.

Acceptance test: `parse_routed("openrouter")` raises `MalformedName`; `parse_pair("/openrouter", "x/y")` raises
`MalformedName`.

### F14 (P3) Test gaps, and tests that pass for the wrong reason

- `tests/test_rules.py:91-112`: the schema negatives cover top-level keys and already-validated tables; there is no
  nested `fields.*` typo case, which is exactly the F1 hole.
- No test that `parse_pair` refuses a peelable route as vendor (F3); no two-segment lookup in `parse_routed` (F7;
  `openrouter/auto` is a single-segment remainder).
- `tests/test_identity.py:115-117`: `pytest.raises((MalformedName, UnknownVendor))` accepts either type for ten inputs
  that are all `MalformedName` today; a regression to `UnknownVendor` would pass.
- `tests/test_golden.py:66-76`: three counters of `coverage.json` are compared; `fill` and `errors` are not, so the
  numbers DESIGN.md quotes can drift from the fixture without a failure.
- DESIGN.md 644 promises the fixture run under an override that flips each policy value, and 651 a property test that
  every slug is rejected in every entry point (the slug rows run through `parse` only). Neither exists.
- `fields.py:1-3` (module docstring) and DESIGN.md 382 say a second token of a single-valued kind is kept in `variant`
  and reported; only a second generation is kept (`second-snapshot:0528`, `second-api-version:002` are reported
  only). Pick one behaviour and pin it with a test.

Acceptance test: the negatives listed under F1, F3 and F7; a single-type `pytest.raises(MalformedName)`;
`coverage["fill"]` recomputed from the fixture rows.

### F15 (P3) Code quality

- `model.py:12-22` `DESCRIPTIVE_FIELDS` duplicates `rules.py:28-38` `FIELD_NAMES` (the same nine names); keep one.
- `identity.py:12` imports the private `_SEGMENT_RE` from `rules.py`; export it or move the segment check into rules.
- `identity.py:159-169` and `identity.py:205-214` implement the same "split vendor/model, check the implied vendor,
  finish" twice; one helper removes the duplication and is where the F3 check belongs.
- `rules.py:178-181`: `Rules.default()` re-reads and re-parses the TOML on every call (`Normalizer()` and every
  `Rules.load` do it); 0.6 ms today, but a cached parsed document is one line.
- `rules.py:175`: `document` is a live `dict` on a frozen dataclass; `Rules.default().document["policy"]["case"] =
  "fold"` succeeds and `with_policy()` then reads the mutated copy while `rules.policy.case` still says `preserve`.
  Wrap it in `MappingProxyType` or deep-copy on the way in.
- `LookupEntry.key`, `Suffix.name`, `DateLayout.name` are stored and never read; `__main__.py:39` builds a
  `Normalizer` for the `rules` subcommand, which does not need one.
- `Rules.routes`, `Rules.policy`, `Rules.lookup` expose `Route`, `Policy`, `LookupEntry` (and `FieldRules`,
  `VendorSwitches`, `Suffix`, `DateLayout`) that are not in `__all__`; export them or keep them private behind
  narrower attributes.

## Verification notes

Ran (Python 3.12.13, interpreter of the `tolokaforge` venv, from the library root):

- `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`: all tests pass (every golden row that carries an
  expectation or an error type, 770 of the 928, plus the unit tests).
- `ruff check .`: clean. `ruff format --check .`: 17 files already formatted. `mypy` (strict, `python_version =
  3.10`): no issues in 10 source files.
- `python -m hatchling build -d /tmp/fable-code-review/dist` (hatchling 1.31.0, offline): wheel contains the ten
  modules, `py.typed`, `rules.toml`, `entry_points.txt` with `toloka-model-name-normalizer =
  toloka_model_name_normalizer.__main__:main`; METADATA has `Requires-Python: >=3.10` and `Requires-Dist: tomli>=2.0;
  python_version < '3.11'` and no licence; the sdist contains `src`, `tests` (with the 800 KB `golden.json`),
  `README.md`, `pyproject.toml`, `.gitignore`. The repository stayed clean (`git status --porcelain` empty).
- Probe scripts, kept under `/tmp/fable-code-review/`: `probe_identity.py` (uppercase routes and providers, names
  equal to routes, routes as vendors, `platform_openai` in every position, empty and one-character segments,
  whitespace, NBSP, tab, BOM, LRM, zero-width space, NFD, a 20 000-character name, colon suffixes, slugs, non-string
  input, resolver edge cases, the case policies, round trips), `probe_rules.py` (schema closure, merge types,
  `extends = false`, route case, fingerprint collisions, lookup bypasses, `Rules.load` errors and CLI exit codes,
  ReDoS, degenerate regexes), `probe_fields.py` (compounds, dash minor version, family grammar, suffixes, dates,
  sizes, vocabulary words, strict mode, `generation_key`), `probe_more.py` and `probe_tail.py` (multi-segment lookup,
  fields-only lookup keys, provider leading slash, alias validation, JSON of a NUL variant, `Rules.default()` cost,
  16-thread `default_normalizer()`: one instance, no errors), `golden_inspect.txt` (fixture census).
- Confirmed conforming, beyond the golden rows: equality and hash on `(vendor, model, route)` only; `same_wire`
  versus a shorter chain; the routed round trip for every declared golden row; lookup in `parse`, `parse_canonical`
  and `parse_pair` including under `unknown_provider = "error"`; `case = fold` and `reject` on the identity while
  routes and providers stay case-insensitive; `implied_vendor_mismatch` in both `parse_routed` and `parse_pair`,
  `name_wins` honoured; slugs refused in `parse` and `parse_pair`; `TypeError` for `None`, `int`, `bytes`;
  `leading_slash` both values; `remove_routes` with a list; `implies_vendor` pointing at a removed route still
  resolves; a `vendor_aliases` key that is a route is refused; `why` and `version` excluded from both fingerprints;
  the module-level default is built once, lazily, and `Normalizer()` never reads the environment.
- `tests/data/golden.json` and `tests/data/coverage.json` are byte-identical to the fixtures in the workspace; no
  golden row contains `\u0000`, so F2 does not force a fixture regeneration.
- Not checked: Python 3.10 at runtime (not installed). By reading: `dataclasses.field(kw_only=True)` exists on 3.10,
  `from __future__ import annotations` keeps `str | None` and `re.Pattern[str]` out of runtime evaluation,
  `importlib.resources.files(...).read_text("utf-8")` is available from 3.9, and the mypy `tomli` override makes the
  3.10 branch type-check; the `tomli` import itself has never executed (F9). Not checked: CORPUS.md and INVENTORY.md
  in full, a real PyPI upload, `uv sync` against the lock (no network), behaviour of the console script when
  installed (only `main(argv)` was called).
