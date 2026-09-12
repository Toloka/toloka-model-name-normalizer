"""Rules: every parsing rule as data. Loaded from TOML, validated against a closed schema, merged over the
packaged default, fingerprinted. Nothing in this module knows a model name."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from toloka_model_name_normalizer.errors import RulesError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

RULES_ENV_VAR = "TOLOKA_MODEL_NAME_NORMALIZER_RULES"
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)
SHAPE_KINDS: tuple[str, ...] = ("size", "snapshot8", "snapshot4", "api3", "context", "generation")
DATE_TOKENS: tuple[str, ...] = ("YYYY", "YY", "MM", "DD")
FIELD_NAMES: tuple[str, ...] = (
    "family",
    "generation",
    "tier",
    "variant",
    "snapshot",
    "size",
    "stage",
    "capability",
    "api_version",
)
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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
_VOCAB_KEYS = ("tier", "stage", "capability", "compounds")
_VOCAB_TARGETS = ("tier", "stage", "capability", "variant")
# a colon suffix is a variant marker on a priced id, never a release stage
_SUFFIX_TARGETS = ("tier", "capability", "variant")


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
    name: str
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
    key: str
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
        text = resources.files("toloka_model_name_normalizer").joinpath("rules.toml").read_text("utf-8")
        return cls.from_dict(tomllib.loads(text))

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, base: Rules | None = None) -> Rules:
        """One override file merged over ``base`` (the packaged default when omitted). ``extends = false``
        in the file starts from an empty rule set instead."""
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
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
        _check_keys(document, _TOP_KEYS, "")
        extends = document.get("extends", True)
        if not isinstance(extends, bool):
            raise RulesError("extends must be a boolean")
        merged = (
            _merge(base.document, document) if (base is not None and extends) else _strip_merge_keys(document)
        )
        _validate(merged)
        return cls._build(merged)

    def with_policy(self, **values: str) -> Rules:
        """A copy with policy values replaced (tests, transition overrides)."""
        document = json.loads(json.dumps(self.document))
        document.setdefault("policy", {}).update(values)
        return Rules.from_dict(document)

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
            key.lower(): LookupEntry(
                key=key.lower(),
                vendor=spec.get("vendor"),
                model=spec.get("model"),
                fields={
                    name: (None if spec[name] == "" else spec[name]) for name in FIELD_NAMES if name in spec
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
        vocab = fields_doc["vocab"]
        fields = FieldRules(
            separators=re.compile(fields_doc["separators"]),
            suffixes={
                name: Suffix(name=name, field=spec["field"], transport=bool(spec.get("transport", False)))
                for name, spec in fields_doc.get("suffixes", {}).items()
            },
            shapes={
                kind: tuple(re.compile(pattern) for pattern in fields_doc["shapes"].get(kind, []))
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
                name.lower(): VendorSwitches(
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
            routes=routes,
            vendor_aliases={alias.lower(): vendor for alias, vendor in doc.get("vendor_aliases", {}).items()},
            bare_names=tuple(
                (pattern.lower(), vendor) for pattern, vendor in doc.get("bare_names", {}).items()
            ),
            lookup=lookup,
            policy=policy,
            render=render,
            fields=fields,
            identity_fingerprint=identity_fp,
            fields_fingerprint=fields_fp,
            document=doc,
        )


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _strip_merge_keys(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in ("extends", "remove_routes")}


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Later wins. Tables keyed by name replace entry by entry, vocabulary lists extend, scalars overwrite,
    ``remove_routes`` drops default routes."""
    merged: dict[str, Any] = json.loads(json.dumps(base))
    for key in ("schema_version", "version"):
        if key in override:
            merged[key] = override[key]
    for name in override.get("remove_routes", []):
        merged.get("routes", {}).pop(name, None)
    for table in ("routes", "vendor_aliases", "bare_names", "lookup"):
        if table in override:
            merged.setdefault(table, {}).update(override[table])
    for table in ("policy", "render"):
        if table in override:
            merged.setdefault(table, {}).update(override[table])
    if "fields" in override:
        fields = merged.setdefault("fields", {})
        over = override["fields"]
        if "separators" in over:
            fields["separators"] = over["separators"]
        for table in ("suffixes", "shapes", "dates"):
            if table in over:
                fields.setdefault(table, {}).update(over[table])
        if "vocab" in over:
            vocab = fields.setdefault("vocab", {})
            for name in ("tier", "stage", "capability"):
                if name in over["vocab"]:
                    existing = list(vocab.get(name, []))
                    vocab[name] = existing + [word for word in over["vocab"][name] if word not in existing]
            if "compounds" in over["vocab"]:
                vocab.setdefault("compounds", {}).update(over["vocab"]["compounds"])
        if "vendor" in over:
            vendors = fields.setdefault("vendor", {})
            for name, switches in over["vendor"].items():
                vendors.setdefault(name, {}).update(switches)
    return merged


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise RulesError(f"unknown key(s) {unknown} in {where or 'the top level'}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RulesError(message)


def _validate(doc: Mapping[str, Any]) -> None:
    _check_keys(doc, tuple(key for key in _TOP_KEYS if key not in ("extends", "remove_routes")), "")
    _require("schema_version" in doc, "schema_version is required")
    _require(
        doc["schema_version"] in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported schema_version {doc['schema_version']!r}; "
        f"this library reads {list(SUPPORTED_SCHEMA_VERSIONS)}",
    )
    _require(
        isinstance(doc.get("version"), str) and bool(doc.get("version")), "version must be a non-empty string"
    )
    for section in ("policy", "render", "fields"):
        _require(isinstance(doc.get(section), Mapping), f"{section} table is required")

    routes = doc.get("routes", {})
    _require(isinstance(routes, Mapping), "routes must be a table")
    for name, spec in routes.items():
        _require(
            _SEGMENT_RE.match(name) is not None and "/" not in name,
            f"route name {name!r} is not a valid segment",
        )
        _require(isinstance(spec, Mapping), f"routes.{name} must be a table")
        _check_keys(spec, ("implies_vendor", "nestable"), f"routes.{name}")
        implied = spec.get("implies_vendor")
        if implied is not None:
            _require(
                isinstance(implied, str) and _SEGMENT_RE.match(implied) is not None,
                f"routes.{name}.implies_vendor must be a vendor segment",
            )
            other = routes.get(implied)
            _require(
                implied == name or other is None or other.get("implies_vendor") == implied,
                f"routes.{name}.implies_vendor names {implied!r}, which is itself a peelable route",
            )
        _require(isinstance(spec.get("nestable", False), bool), f"routes.{name}.nestable must be a boolean")

    aliases = doc.get("vendor_aliases", {})
    _require(isinstance(aliases, Mapping), "vendor_aliases must be a table")
    for alias, vendor in aliases.items():
        _require(
            "/" not in alias and isinstance(vendor, str) and "/" not in vendor,
            f"vendor_aliases.{alias} must map a vendor segment to a vendor segment",
        )
        _require(alias.lower() not in {r.lower() for r in routes}, f"vendor_aliases.{alias} is also a route")

    bare = doc.get("bare_names", {})
    _require(isinstance(bare, Mapping), "bare_names must be a table")
    for pattern, vendor in bare.items():
        _require(
            "*" in pattern or "?" in pattern,
            f"bare_names pattern {pattern!r} has no wildcard; use lookup for an exact name",
        )
        _require(
            isinstance(vendor, str) and "/" not in vendor, f"bare_names.{pattern} must name a vendor segment"
        )

    lookup = doc.get("lookup", {})
    _require(isinstance(lookup, Mapping), "lookup must be a table")
    for key, spec in lookup.items():
        _require(isinstance(spec, Mapping), f"lookup.{key} must be a table")
        _check_keys(spec, ("vendor", "model", "why", *FIELD_NAMES), f"lookup.{key}")
        _require(
            isinstance(spec.get("why"), str) and bool(spec.get("why")), f"lookup.{key} needs a non-empty why"
        )
        _require(("vendor" in spec) == ("model" in spec), f"lookup.{key} must set vendor and model together")
        _require(any(name in spec for name in ("vendor", *FIELD_NAMES)), f"lookup.{key} overrides nothing")

    policy = doc["policy"]
    _check_keys(policy, tuple(_POLICY_VALUES), "policy")
    for key, values in _POLICY_VALUES.items():
        _require(key in policy, f"policy.{key} is required")
        _require(policy[key] in values, f"policy.{key} must be one of {list(values)}, got {policy[key]!r}")

    render = doc["render"]
    _check_keys(render, _RENDER_KEYS, "render")
    for key in _RENDER_KEYS:
        _require(isinstance(render.get(key), str), f"render.{key} must be a string")
    _require(render["route_less"] in ("unknown", "omit"), "render.route_less must be 'unknown' or 'omit'")
    _compile(render["unsafe"], "render.unsafe")

    fields = doc["fields"]
    _check_keys(fields, _FIELDS_KEYS, "fields")
    _require(isinstance(fields.get("separators"), str), "fields.separators must be a regex string")
    _compile(fields["separators"], "fields.separators")
    for name, spec in fields.get("suffixes", {}).items():
        _check_keys(spec, ("field", "transport"), f"fields.suffixes.{name}")
        _require(
            spec.get("field") in _SUFFIX_TARGETS,
            f"fields.suffixes.{name}.field must be one of {list(_SUFFIX_TARGETS)}",
        )
    shapes = fields.get("shapes", {})
    _check_keys(shapes, SHAPE_KINDS, "fields.shapes")
    for kind, patterns in shapes.items():
        _require(isinstance(patterns, list), f"fields.shapes.{kind} must be a list of regex strings")
        for pattern in patterns:
            _compile(pattern, f"fields.shapes.{kind}")
    for name, spec in fields.get("dates", {}).items():
        _check_keys(spec, ("tokens", "requires_generation"), f"fields.dates.{name}")
        tokens = spec.get("tokens")
        _require(
            isinstance(tokens, list) and 2 <= len(tokens) <= 3 and all(t in DATE_TOKENS for t in tokens),
            f"fields.dates.{name}.tokens must be 2 or 3 of {list(DATE_TOKENS)}",
        )
        _require(len(set(tokens)) == len(tokens), f"fields.dates.{name}.tokens repeats a component")
    vocab = fields.get("vocab", {})
    _check_keys(vocab, _VOCAB_KEYS, "fields.vocab")
    for name in ("tier", "stage", "capability"):
        words = vocab.get(name, [])
        _require(
            isinstance(words, list) and all(isinstance(w, str) and w for w in words),
            f"fields.vocab.{name} must be a list of words",
        )
    for compound, target in vocab.get("compounds", {}).items():
        _require(
            "-" in compound and target in _VOCAB_TARGETS,
            f"fields.vocab.compounds.{compound} must be hyphenated and target one of {list(_VOCAB_TARGETS)}",
        )
    for vendor, switches in fields.get("vendor", {}).items():
        _require("/" not in vendor, f"fields.vendor.{vendor} is not a vendor segment")
        _check_keys(switches, _VENDOR_SWITCHES, f"fields.vendor.{vendor}")
        if "four_digit_snapshot" in switches:
            _require(
                switches["four_digit_snapshot"] in ("MMDD", "YYMM"),
                f"fields.vendor.{vendor}.four_digit_snapshot must be MMDD or YYMM",
            )
        if "letter_generations" in switches:
            _require(
                all(isinstance(g, str) and len(g) == 1 for g in switches["letter_generations"]),
                f"fields.vendor.{vendor}.letter_generations must be single letters",
            )


def _compile(pattern: str, where: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise RulesError(f"{where}: invalid regex {pattern!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprints(doc: Mapping[str, Any]) -> tuple[str, str]:
    """``(identity_fingerprint, fields_fingerprint)`` over the canonical JSON of the merged rules.

    The identity side is what decides ``vendor``, ``model``, ``route`` and ``family``: routes, aliases,
    bare-name patterns, the identity half of lookup, the policy and the separator set. The fields side is
    everything else. ``why``, ``version``, comments and merge keys never participate; ``schema_version`` does.
    """
    lookup = doc.get("lookup", {})
    identity = {
        "schema_version": doc["schema_version"],
        "routes": doc.get("routes", {}),
        "vendor_aliases": doc.get("vendor_aliases", {}),
        "bare_names": doc.get("bare_names", {}),
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
            key: {k: spec[k] for k in FIELD_NAMES if k in spec}
            for key, spec in lookup.items()
            if any(k in spec for k in FIELD_NAMES)
        },
    }
    return (
        "sha256:" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest(),
        "sha256:" + hashlib.sha256(_canonical_json(fields).encode("utf-8")).hexdigest(),
    )
