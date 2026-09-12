"""Rules: every parsing rule as data. Loaded from TOML, validated against a closed schema (the override
document on its own and the merged result), merged over the packaged default, fingerprinted. Nothing in this
module knows a model name."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from toloka_model_name_normalizer.errors import RulesError
from toloka_model_name_normalizer.model import DESCRIPTIVE_FIELDS

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

RULES_ENV_VAR = "TOLOKA_MODEL_NAME_NORMALIZER_RULES"
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)
SHAPE_KINDS: tuple[str, ...] = ("size", "snapshot8", "snapshot4", "api3", "context", "generation")
DATE_TOKENS: tuple[str, ...] = ("YYYY", "YY", "MM", "DD")

#: The grammar of a vendor, a route and a provider segment.
SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

_POLICY_VALUES: dict[str, tuple[str, ...]] = {
    "unknown_provider": ("vendor", "error"),
    "bare_name": ("resolver", "error", "provider_as_vendor"),
    "case": ("preserve", "fold", "reject"),
    "leading_slash": ("tolerate", "reject"),
    "implied_vendor_mismatch": ("error", "name_wins"),
    "fields": ("best_effort", "strict"),
}
_RENDER_KEYS = ("route_sep", "unsafe", "unsafe_replacement", "route_less")
_VENDOR_SWITCHES = (
    "dash_minor_version",
    "four_digit_snapshot",
    "letter_generations",
    "trailing_v_is_api_version",
    "three_digit_api_version",
)
_TOP_KEYS = (
    "schema_version",
    "version",
    "extends",
    "remove_routes",
    "routes",
    "vendor_aliases",
    "bare_names",
    "lookup",
    "policy",
    "render",
    "fields",
)
_FIELDS_KEYS = ("separators", "suffixes", "shapes", "dates", "vocab", "vendor")
_VOCAB_LISTS = ("tier", "stage", "capability")
_VOCAB_KEYS = (*_VOCAB_LISTS, "compounds")
_VOCAB_TARGETS = ("tier", "stage", "capability", "variant")
# a colon suffix is a variant marker on a priced id, never a release stage
_SUFFIX_TARGETS = ("tier", "capability", "variant")
# every shape kind slices its token by a fixed layout, so a regex for the kind must only match that layout
_SHAPE_PROBES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "snapshot8": (("20240101",), ("0731", "001", "2024010", "202401011", "abcdefgh")),
    "snapshot4": (("0731",), ("20240101", "001", "07311", "abcd")),
    "api3": (("001",), ("0731", "20240101", "0011", "abc")),
}


def check_model_grammar(model: str, route_sep: str) -> str | None:
    """The reason ``model`` breaks the model grammar, or ``None``; shared by the parser and the validator."""
    if not model:
        return "empty model name"
    if "/" in model:
        return "slash inside the model name"
    if model != model.strip():
        return "model name has outer whitespace"
    if route_sep in model:
        return "looks like a slug, which is render-only"
    if any(unicodedata.category(char) in ("Cc", "Cf") for char in model):
        return "control or format character in the model name"
    if model.count(":") > 1:
        return "more than one colon suffix"
    if not _ALNUM_RE.search(model.partition(":")[0]):
        return "no letter or digit before the suffix"
    return None


@dataclass(frozen=True)
class Route:
    name: str
    implies_vendor: str | None
    nestable: bool

    @property
    def peelable(self) -> bool:
        """A route is peeled from a string unless its name is its own implied vendor: ``openai/gpt-6-astra``
        must keep reading as the canonical ``vendor/model``."""
        return self.implies_vendor != self.name


@dataclass(frozen=True)
class Policy:
    unknown_provider: str
    bare_name: str
    case: str
    leading_slash: str
    implied_vendor_mismatch: str
    fields: str


@dataclass(frozen=True)
class Render:
    route_sep: str
    unsafe: re.Pattern[str]
    unsafe_replacement: str
    route_less: str


@dataclass(frozen=True)
class Suffix:
    field: str
    transport: bool


@dataclass(frozen=True)
class DateLayout:
    name: str
    tokens: tuple[str, ...]
    requires_generation: bool


@dataclass(frozen=True)
class VendorSwitches:
    dash_minor_version: bool = False
    four_digit_snapshot: str | None = None
    letter_generations: tuple[str, ...] = ()
    trailing_v_is_api_version: bool = False
    three_digit_api_version: bool = False


@dataclass(frozen=True)
class LookupEntry:
    vendor: str | None
    model: str | None
    fields: Mapping[str, str | None]
    why: str

    @property
    def has_identity(self) -> bool:
        return self.vendor is not None and self.model is not None


@dataclass(frozen=True)
class FieldRules:
    separators: re.Pattern[str]
    suffixes: Mapping[str, Suffix]
    shapes: Mapping[str, tuple[re.Pattern[str], ...]]
    dates: tuple[DateLayout, ...]
    tier: tuple[str, ...]
    stage: tuple[str, ...]
    capability: tuple[str, ...]
    compounds: Mapping[str, str]
    vendor: Mapping[str, VendorSwitches]

    def switches(self, vendor: str) -> VendorSwitches:
        return self.vendor.get(vendor.lower(), VendorSwitches())


@dataclass(frozen=True)
class Rules:
    """A validated, merged rule set. Build one with :meth:`default`, :meth:`load` or :meth:`from_env`."""

    schema_version: int
    version: str
    routes: Mapping[str, Route]
    vendor_aliases: Mapping[str, str]
    bare_names: tuple[tuple[str, str], ...]
    lookup: Mapping[str, LookupEntry]
    policy: Policy
    render: Render
    fields: FieldRules
    identity_fingerprint: str
    fields_fingerprint: str
    document: Mapping[str, Any]

    @classmethod
    def default(cls) -> Rules:
        """The packaged Toloka default (``rules.toml`` inside the wheel)."""
        return cls.from_dict(copy.deepcopy(_default_document()))

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, base: Rules | None = None) -> Rules:
        """One override file merged over ``base`` (the packaged default when omitted). ``extends = false``
        in the file starts from an empty rule set instead."""
        try:
            with open(path, "rb") as handle:
                document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise RulesError(f"{os.fspath(path)}: not valid TOML: {exc}") from exc
        return cls.from_dict(document, base=base if base is not None else cls.default())

    @classmethod
    def from_env(cls, env_var: str = RULES_ENV_VAR) -> Rules:
        """The file named by the environment variable, merged over the default; the default when unset."""
        path = os.environ.get(env_var)
        if not path:
            return cls.default()
        return cls.load(Path(path))

    @classmethod
    def from_dict(cls, document: Mapping[str, Any], *, base: Rules | None = None) -> Rules:
        """Validate ``document`` and build the rule set, merging over ``base`` unless ``extends = false``."""
        _check_shape(document)
        extends = document.get("extends", True)
        if base is not None and extends:
            merged = _merge(base.document, document)
        else:
            merged = _fold_names(_strip_merge_keys(document))
        _validate(merged)
        return cls._build(merged)

    def with_policy(self, **values: str) -> Rules:
        """A copy with policy values replaced (tests, transition overrides)."""
        return Rules.from_dict({"policy": dict(values)}, base=self)

    @classmethod
    def _build(cls, doc: Mapping[str, Any]) -> Rules:
        routes = {
            name: Route(
                name=name,
                implies_vendor=spec.get("implies_vendor"),
                nestable=bool(spec.get("nestable", False)),
            )
            for name, spec in doc.get("routes", {}).items()
        }
        lookup = {
            key: LookupEntry(
                vendor=spec.get("vendor"),
                model=spec.get("model"),
                fields={
                    name: (None if spec[name] == "" else spec[name])
                    for name in DESCRIPTIVE_FIELDS
                    if name in spec
                },
                why=spec["why"],
            )
            for key, spec in doc.get("lookup", {}).items()
        }
        policy_doc = doc["policy"]
        policy = Policy(**{key: policy_doc[key] for key in _POLICY_VALUES})
        render_doc = doc["render"]
        render = Render(
            route_sep=render_doc["route_sep"],
            unsafe=re.compile(render_doc["unsafe"]),
            unsafe_replacement=render_doc["unsafe_replacement"],
            route_less=render_doc["route_less"],
        )
        fields_doc = doc["fields"]
        vocab = fields_doc.get("vocab", {})
        shapes_doc = fields_doc.get("shapes", {})
        fields = FieldRules(
            separators=re.compile(fields_doc["separators"]),
            suffixes={
                name: Suffix(field=spec["field"], transport=bool(spec.get("transport", False)))
                for name, spec in fields_doc.get("suffixes", {}).items()
            },
            shapes={
                kind: tuple(re.compile(pattern) for pattern in shapes_doc.get(kind, []))
                for kind in SHAPE_KINDS
            },
            dates=tuple(
                DateLayout(
                    name=name,
                    tokens=tuple(spec["tokens"]),
                    requires_generation=bool(spec.get("requires_generation", False)),
                )
                for name, spec in fields_doc.get("dates", {}).items()
            ),
            tier=tuple(vocab.get("tier", [])),
            stage=tuple(vocab.get("stage", [])),
            capability=tuple(vocab.get("capability", [])),
            compounds=dict(vocab.get("compounds", {})),
            vendor={
                name: VendorSwitches(
                    dash_minor_version=bool(spec.get("dash_minor_version", False)),
                    four_digit_snapshot=spec.get("four_digit_snapshot"),
                    letter_generations=tuple(spec.get("letter_generations", [])),
                    trailing_v_is_api_version=bool(spec.get("trailing_v_is_api_version", False)),
                    three_digit_api_version=bool(spec.get("three_digit_api_version", False)),
                )
                for name, spec in fields_doc.get("vendor", {}).items()
            },
        )
        identity_fp, fields_fp = fingerprints(doc)
        return cls(
            schema_version=int(doc["schema_version"]),
            version=str(doc["version"]),
            routes=MappingProxyType(routes),
            vendor_aliases=MappingProxyType(dict(doc.get("vendor_aliases", {}))),
            bare_names=tuple((pattern, vendor) for pattern, vendor in doc.get("bare_names", {}).items()),
            lookup=MappingProxyType(lookup),
            policy=policy,
            render=render,
            fields=fields,
            identity_fingerprint=identity_fp,
            fields_fingerprint=fields_fp,
            document=MappingProxyType(copy.deepcopy(dict(doc))),
        )


@functools.lru_cache(maxsize=1)
def _default_document() -> dict[str, Any]:
    text = resources.files("toloka_model_name_normalizer").joinpath("rules.toml").read_text("utf-8")
    try:
        document: dict[str, Any] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - the packaged file is tested
        raise RulesError(f"packaged rules.toml is not valid TOML: {exc}") from exc
    return document


# ---------------------------------------------------------------------------
# shape: known keys and types, on the override alone and on the merged document
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RulesError(message)


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise RulesError(f"unknown key(s) {unknown} in {where or 'the top level'}")


def _table(value: Any, where: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{where} must be a table")
    return value  # type: ignore[no-any-return]


def _str_list(value: Any, where: str) -> list[str]:
    _require(
        isinstance(value, list) and all(isinstance(item, str) for item in value),
        f"{where} must be a list of strings",
    )
    return value  # type: ignore[no-any-return]


def _check_shape(doc: Mapping[str, Any]) -> None:
    """Known keys and value types at every level, without requiring completeness (overrides are partial)."""
    _table(doc, "the rules document")
    _check_keys(doc, _TOP_KEYS, "")
    if "schema_version" in doc:
        _require(
            isinstance(doc["schema_version"], int) and not isinstance(doc["schema_version"], bool),
            "schema_version must be an integer",
        )
    if "version" in doc:
        _require(
            isinstance(doc["version"], str) and bool(doc["version"]), "version must be a non-empty string"
        )
    if "extends" in doc:
        _require(isinstance(doc["extends"], bool), "extends must be a boolean")
    if "remove_routes" in doc:
        _str_list(doc["remove_routes"], "remove_routes")
    for name, spec in _table(doc.get("routes", {}), "routes").items():
        _check_keys(_table(spec, f"routes.{name}"), ("implies_vendor", "nestable"), f"routes.{name}")
        if "implies_vendor" in spec:
            _require(
                isinstance(spec["implies_vendor"], str), f"routes.{name}.implies_vendor must be a string"
            )
        if "nestable" in spec:
            _require(isinstance(spec["nestable"], bool), f"routes.{name}.nestable must be a boolean")
    folded = [name.lower() for name in doc.get("routes", {})]
    _require(len(folded) == len(set(folded)), "routes: two names collide after case folding")
    for table in ("vendor_aliases", "bare_names"):
        for key, value in _table(doc.get(table, {}), table).items():
            _require(isinstance(value, str), f"{table}.{key} must be a string")
    for key, spec in _table(doc.get("lookup", {}), "lookup").items():
        _check_keys(
            _table(spec, f"lookup.{key}"), ("vendor", "model", "why", *DESCRIPTIVE_FIELDS), f"lookup.{key}"
        )
        for name, value in spec.items():
            _require(isinstance(value, str), f"lookup.{key}.{name} must be a string")
    policy = _table(doc.get("policy", {}), "policy")
    _check_keys(policy, tuple(_POLICY_VALUES), "policy")
    for key, values in _POLICY_VALUES.items():
        if key in policy:
            _require(
                policy[key] in values, f"policy.{key} must be one of {list(values)}, got {policy[key]!r}"
            )
    render = _table(doc.get("render", {}), "render")
    _check_keys(render, _RENDER_KEYS, "render")
    for key, value in render.items():
        _require(isinstance(value, str), f"render.{key} must be a string")
    fields = _table(doc.get("fields", {}), "fields")
    _check_keys(fields, _FIELDS_KEYS, "fields")
    if "separators" in fields:
        _require(isinstance(fields["separators"], str), "fields.separators must be a regex string")
    for name, spec in _table(fields.get("suffixes", {}), "fields.suffixes").items():
        _check_keys(
            _table(spec, f"fields.suffixes.{name}"), ("field", "transport"), f"fields.suffixes.{name}"
        )
        _require(
            spec.get("field") in _SUFFIX_TARGETS,
            f"fields.suffixes.{name}.field must be one of {list(_SUFFIX_TARGETS)}",
        )
        if "transport" in spec:
            _require(
                isinstance(spec["transport"], bool), f"fields.suffixes.{name}.transport must be a boolean"
            )
    shapes = _table(fields.get("shapes", {}), "fields.shapes")
    _check_keys(shapes, SHAPE_KINDS, "fields.shapes")
    for kind, patterns in shapes.items():
        _str_list(patterns, f"fields.shapes.{kind}")
    for name, spec in _table(fields.get("dates", {}), "fields.dates").items():
        _check_keys(
            _table(spec, f"fields.dates.{name}"), ("tokens", "requires_generation"), f"fields.dates.{name}"
        )
        tokens = _str_list(spec.get("tokens", []), f"fields.dates.{name}.tokens")
        _require(
            2 <= len(tokens) <= 3 and all(t in DATE_TOKENS for t in tokens),
            f"fields.dates.{name}.tokens must be 2 or 3 of {list(DATE_TOKENS)}",
        )
        _require(len(set(tokens)) == len(tokens), f"fields.dates.{name}.tokens repeats a component")
        if "requires_generation" in spec:
            _require(
                isinstance(spec["requires_generation"], bool),
                f"fields.dates.{name}.requires_generation must be a boolean",
            )
    vocab = _table(fields.get("vocab", {}), "fields.vocab")
    _check_keys(vocab, _VOCAB_KEYS, "fields.vocab")
    for name in _VOCAB_LISTS:
        for word in _str_list(vocab.get(name, []), f"fields.vocab.{name}"):
            _require(
                bool(word) and word == word.lower() and " " not in word,
                f"fields.vocab.{name}: {word!r} must be a lower-case word",
            )
    for compound, target in _table(vocab.get("compounds", {}), "fields.vocab.compounds").items():
        _require(
            "-" in compound and compound == compound.lower(),
            f"fields.vocab.compounds.{compound} must be hyphenated and lower-case",
        )
        _require(
            target in _VOCAB_TARGETS,
            f"fields.vocab.compounds.{compound} must target one of {list(_VOCAB_TARGETS)}",
        )
    for vendor, switches in _table(fields.get("vendor", {}), "fields.vendor").items():
        _check_keys(_table(switches, f"fields.vendor.{vendor}"), _VENDOR_SWITCHES, f"fields.vendor.{vendor}")
        for key in ("dash_minor_version", "trailing_v_is_api_version", "three_digit_api_version"):
            if key in switches:
                _require(isinstance(switches[key], bool), f"fields.vendor.{vendor}.{key} must be a boolean")
        if "four_digit_snapshot" in switches:
            _require(
                switches["four_digit_snapshot"] in ("MMDD", "YYMM"),
                f"fields.vendor.{vendor}.four_digit_snapshot must be MMDD or YYMM",
            )
        if "letter_generations" in switches:
            letters = _str_list(switches["letter_generations"], f"fields.vendor.{vendor}.letter_generations")
            _require(
                all(len(g) == 1 and g.islower() for g in letters),
                f"fields.vendor.{vendor}.letter_generations must be single lower-case letters",
            )


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _strip_merge_keys(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in ("extends", "remove_routes")}


def _fold_names(doc: dict[str, Any]) -> dict[str, Any]:
    """Rule tables are matched case-insensitively, so their keys are stored folded (routes, aliases, bare-name
    patterns, lookup keys, vendor switches) and so is ``implies_vendor``."""
    doc = copy.deepcopy(doc)
    if "routes" in doc:
        doc["routes"] = {
            name.lower(): {
                **spec,
                **({"implies_vendor": spec["implies_vendor"].lower()} if "implies_vendor" in spec else {}),
            }
            for name, spec in doc["routes"].items()
        }
    for table in ("vendor_aliases", "bare_names", "lookup"):
        if table in doc:
            doc[table] = {key.lower(): value for key, value in doc[table].items()}
    if "fields" in doc and "vendor" in doc["fields"]:
        doc["fields"]["vendor"] = {name.lower(): spec for name, spec in doc["fields"]["vendor"].items()}
    return doc


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Later wins. Tables keyed by name replace entry by entry, vocabulary lists extend, scalars overwrite,
    ``remove_routes`` drops default routes. ``override`` has passed :func:`_check_shape`."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    over = _fold_names(_strip_merge_keys(override))
    for key in ("schema_version", "version"):
        if key in over:
            merged[key] = over[key]
    for name in override.get("remove_routes", []):
        merged.get("routes", {}).pop(name.lower(), None)
    for table in ("routes", "vendor_aliases", "bare_names", "lookup", "policy", "render"):
        if table in over:
            merged.setdefault(table, {}).update(over[table])
    if "fields" in over:
        fields = merged.setdefault("fields", {})
        over_fields = over["fields"]
        if "separators" in over_fields:
            fields["separators"] = over_fields["separators"]
        for table in ("suffixes", "shapes", "dates"):
            if table in over_fields:
                fields.setdefault(table, {}).update(over_fields[table])
        if "vocab" in over_fields:
            vocab = fields.setdefault("vocab", {})
            for name in _VOCAB_LISTS:
                if name in over_fields["vocab"]:
                    existing = list(vocab.get(name, []))
                    vocab[name] = existing + [
                        word for word in over_fields["vocab"][name] if word not in existing
                    ]
            if "compounds" in over_fields["vocab"]:
                vocab.setdefault("compounds", {}).update(over_fields["vocab"]["compounds"])
        if "vendor" in over_fields:
            vendors = fields.setdefault("vendor", {})
            for name, switches in over_fields["vendor"].items():
                vendors.setdefault(name, {}).update(switches)
    return merged


# ---------------------------------------------------------------------------
# validation of the merged document: completeness and semantics
# ---------------------------------------------------------------------------


def _compile(pattern: str, where: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise RulesError(f"{where}: invalid regex {pattern!r}: {exc}") from exc


def _validate(doc: Mapping[str, Any]) -> None:
    _check_shape(doc)
    _require("schema_version" in doc, "schema_version is required")
    _require(
        doc["schema_version"] in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported schema_version {doc['schema_version']!r}; "
        f"this library reads {list(SUPPORTED_SCHEMA_VERSIONS)}",
    )
    _require("version" in doc, "version is required")
    for section in ("policy", "render", "fields"):
        _require(isinstance(doc.get(section), Mapping), f"{section} table is required")

    routes = doc.get("routes", {})
    for name, spec in routes.items():
        _require(
            SEGMENT_RE.match(name) is not None and "/" not in name,
            f"route name {name!r} is not a valid segment",
        )
        implied = spec.get("implies_vendor")
        if implied is not None:
            _require(
                SEGMENT_RE.match(implied) is not None,
                f"routes.{name}.implies_vendor must be a vendor segment",
            )
            other = routes.get(implied)
            _require(
                implied == name or other is None or other.get("implies_vendor") == implied,
                f"routes.{name}.implies_vendor names {implied!r}, which is itself a peelable route",
            )
    for alias, vendor in doc.get("vendor_aliases", {}).items():
        _require(
            SEGMENT_RE.match(alias) is not None and SEGMENT_RE.match(vendor) is not None,
            f"vendor_aliases.{alias} must map a vendor segment to a vendor segment",
        )
        _require(alias not in routes, f"vendor_aliases.{alias} is also a route")
    for pattern, vendor in doc.get("bare_names", {}).items():
        _require(
            "*" in pattern or "?" in pattern,
            f"bare_names pattern {pattern!r} has no wildcard; use lookup for an exact name",
        )
        _require(SEGMENT_RE.match(vendor) is not None, f"bare_names.{pattern} must name a vendor segment")

    render = doc["render"]
    for key in _RENDER_KEYS:
        _require(key in render, f"render.{key} is required")
    _require(render["route_less"] in ("unknown", "omit"), "render.route_less must be 'unknown' or 'omit'")
    _require(
        bool(render["route_sep"]) and bool(render["unsafe_replacement"]),
        "render.route_sep and render.unsafe_replacement must be non-empty",
    )
    unsafe = _compile(render["unsafe"], "render.unsafe")
    _require(unsafe.search("") is None, "render.unsafe must not match the empty string")

    for key, spec in doc.get("lookup", {}).items():
        _require(
            bool(key) and key == key.strip() and "/" in key,
            f"lookup key {key!r} must be a non-empty vendor/model or route/name string",
        )
        _require(bool(spec.get("why")), f"lookup.{key} needs a non-empty why")
        _require(("vendor" in spec) == ("model" in spec), f"lookup.{key} must set vendor and model together")
        _require(
            any(name in spec for name in ("vendor", *DESCRIPTIVE_FIELDS)), f"lookup.{key} overrides nothing"
        )
        if "vendor" in spec:
            _require(
                SEGMENT_RE.match(spec["vendor"]) is not None, f"lookup.{key}.vendor must be a vendor segment"
            )
            problem = check_model_grammar(spec["model"], render["route_sep"])
            _require(problem is None, f"lookup.{key}.model: {problem}")
        else:
            _require(
                key.count("/") == 1,
                f"lookup.{key} sets fields only, so its key must be the canonical vendor/model",
            )
        _require(
            spec.get("family", "x") != "", f"lookup.{key}.family must not be empty: family is a static field"
        )

    policy = doc["policy"]
    for key in _POLICY_VALUES:
        _require(key in policy, f"policy.{key} is required")

    fields = doc["fields"]
    _require("separators" in fields, "fields.separators is required")
    separators = _compile(fields["separators"], "fields.separators")
    _require(separators.search("") is None, "fields.separators must not match the empty string")
    for kind, patterns in fields.get("shapes", {}).items():
        for pattern in patterns:
            compiled = _compile(pattern, f"fields.shapes.{kind}")
            if kind in _SHAPE_PROBES:
                positive, negative = _SHAPE_PROBES[kind]
                _require(
                    all(compiled.match(p) for p in positive) and not any(compiled.match(n) for n in negative),
                    f"fields.shapes.{kind}: {pattern!r} must match tokens of that kind's layout only",
                )
    for compound, _target in fields.get("vocab", {}).get("compounds", {}).items():
        _require(
            all(part for part in compound.split("-")), f"fields.vocab.compounds.{compound} has an empty part"
        )


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprints(doc: Mapping[str, Any]) -> tuple[str, str]:
    """``(identity_fingerprint, fields_fingerprint)`` over the canonical JSON of the merged rules.

    The identity side is what decides ``vendor``, ``model``, ``route`` and ``family``: routes, aliases, the
    ordered bare-name patterns (a list, because their order is a rule), the identity half of lookup, the
    policy and the separator set. The fields side is everything else. Names are hashed case-folded, as they
    are matched. ``why``, ``version``, comments and merge keys never participate; ``schema_version`` does.
    """
    lookup = doc.get("lookup", {})
    identity = {
        "schema_version": doc["schema_version"],
        "routes": doc.get("routes", {}),
        "vendor_aliases": doc.get("vendor_aliases", {}),
        "bare_names": [[pattern, vendor] for pattern, vendor in doc.get("bare_names", {}).items()],
        "lookup": {
            key: {k: spec[k] for k in ("vendor", "model") if k in spec}
            for key, spec in lookup.items()
            if "vendor" in spec
        },
        "policy": doc["policy"],
        "separators": doc["fields"]["separators"],
    }
    fields = {
        "schema_version": doc["schema_version"],
        "render": doc["render"],
        "fields": {key: value for key, value in doc["fields"].items() if key != "separators"},
        "lookup": {
            key: {k: spec[k] for k in DESCRIPTIVE_FIELDS if k in spec}
            for key, spec in lookup.items()
            if any(k in spec for k in DESCRIPTIVE_FIELDS)
        },
    }
    return (
        "sha256:" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest(),
        "sha256:" + hashlib.sha256(_canonical_json(fields).encode("utf-8")).hexdigest(),
    )
