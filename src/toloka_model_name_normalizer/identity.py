"""The closed identity grammar: ``vendor``, ``model``, ``route``, one entry point per carrier. Every decision
here is read from :class:`~toloka_model_name_normalizer.rules.Rules`."""

from __future__ import annotations

import fnmatch
import unicodedata
from dataclasses import dataclass

from toloka_model_name_normalizer.errors import AmbiguousModelName, MalformedName, UnknownRoute, UnknownVendor
from toloka_model_name_normalizer.resolver import Resolver, vendor_of_candidate
from toloka_model_name_normalizer.rules import _SEGMENT_RE, Rules


@dataclass(frozen=True)
class Identity:
    vendor: str
    model: str
    route: str | None
    routes: tuple[str, ...]
    lookup: bool
    resolved: bool


def _text(rules: Rules, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"model reference must be str, got {type(value).__name__}")
    text = value.strip()
    if text.startswith("/"):
        if rules.policy.leading_slash == "reject":
            raise MalformedName(f"leading slash: {value!r}")
        text = text.lstrip("/")
    return text


def _check_segment(segment: str, what: str, text: str) -> None:
    if not _SEGMENT_RE.match(segment):
        raise MalformedName(f"{what} charset: {text!r}")


def _check_model(rules: Rules, model: str, text: str) -> None:
    if not model:
        raise MalformedName(f"empty model name: {text!r}")
    if "/" in model:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if model != model.strip():
        raise MalformedName(f"model name has outer whitespace: {text!r}")
    if rules.render.route_sep in model:
        raise MalformedName(f"looks like a slug, which is render-only: {text!r}")
    if any(unicodedata.category(char) in ("Cc", "Cf") for char in model):
        raise MalformedName(f"control or format character in the model name: {text!r}")


def _route(rules: Rules, segment: str) -> str | None:
    """The declared route named by ``segment`` (case-insensitive), or ``None``."""
    key = segment.lower()
    return key if key in rules.routes else None


def _peelable(rules: Rules, route: str) -> bool:
    return rules.routes[route].peelable


def _lookup(rules: Rules, key: str, route: str | None, chain: tuple[str, ...]) -> Identity | None:
    entry = rules.lookup.get(key.lower())
    if entry is None or not entry.has_identity:
        return None
    assert entry.vendor is not None and entry.model is not None
    return Identity(entry.vendor, entry.model, route, chain, lookup=True, resolved=False)


def _finish(
    rules: Rules,
    vendor: str,
    model: str,
    route: str | None,
    chain: tuple[str, ...],
    text: str,
    *,
    resolved: bool = False,
) -> Identity:
    _check_segment(vendor, "vendor", text)
    _check_model(rules, model, text)
    vendor = rules.vendor_aliases.get(vendor.lower(), vendor)
    if rules.policy.case == "reject" and (vendor != vendor.lower() or model != model.lower()):
        raise MalformedName(f"mixed case refused by policy: {text!r}")
    if rules.policy.case == "fold":
        vendor, model = vendor.lower(), model.lower()
    return Identity(vendor, model, route, chain, lookup=False, resolved=resolved)


def _resolve_bare(
    rules: Rules, bare: str, resolver: Resolver | None, route: str | None, chain: tuple[str, ...], text: str
) -> Identity:
    _check_model(rules, bare, text)
    for pattern, vendor in rules.bare_names:
        if fnmatch.fnmatchcase(bare.lower(), pattern):
            return _finish(rules, vendor, bare, route, chain, text)
    policy = rules.policy.bare_name
    if policy == "provider_as_vendor" and route is not None:
        return _finish(rules, route, bare, route, chain, text)
    if policy == "error" or (policy == "provider_as_vendor" and route is None):
        raise UnknownVendor(f"bare name {bare!r}: no route implies a vendor and the policy is {policy!r}")
    if resolver is None:
        raise UnknownVendor(f"bare name {bare!r} and no resolver")
    candidates = tuple(resolver.candidates(bare))
    if not candidates:
        raise UnknownVendor(f"bare name {bare!r}: the resolver has no candidate")
    if len(candidates) > 1:
        raise AmbiguousModelName(bare, candidates)
    return _finish(rules, vendor_of_candidate(candidates[0]), bare, route, chain, text, resolved=True)


def parse_canonical(rules: Rules, text: str) -> Identity:
    """``vendor/model`` with no route: pricing keys, ``model:`` tags, collector ids."""
    value = _text(rules, text)
    hit = _lookup(rules, value, None, ())
    if hit:
        return hit
    parts = value.split("/")
    if len(parts) != 2 or not parts[0]:
        raise MalformedName(f"canonical form needs vendor/model: {text!r}")
    route = _route(rules, parts[0])
    if route is not None and _peelable(rules, route):
        raise MalformedName(f"{parts[0]!r} is a route, not a vendor: {text!r}")
    return _finish(rules, parts[0], parts[1], None, (), text)


def parse_routed(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """The wire form: the first segment must be a declared route, direct-vendor routes included. The lookup
    runs on ``innermost_route/remainder`` after peeling, so a lookup identity keeps the route it travelled on
    (``openrouter/auto`` read here is the wire form of ``(openrouter, auto)``, route ``openrouter``)."""
    value = _text(rules, text)
    parts = value.split("/")
    first = _route(rules, parts[0]) if len(parts) >= 2 else None
    if first is None:
        raise UnknownRoute(f"first segment is not a declared route: {text!r}")
    chain: list[str] = []
    while len(parts) >= 2:
        route = _route(rules, parts[0])
        if route is None:
            break
        if chain and not (rules.routes[chain[-1]].nestable and _peelable(rules, route)):
            break
        if chain and chain[-1] == route:
            raise MalformedName(f"repeated route {route!r}: {text!r}")
        chain.append(route)
        parts = parts[1:]
        implied = rules.routes[route].implies_vendor
        if len(parts) == 1:
            found = _lookup(rules, f"{route}/{parts[0]}", chain[0], tuple(chain))
            if found:
                return found
            if implied:
                return _finish(rules, implied, parts[0], chain[0], tuple(chain), text)
            return _resolve_bare(rules, parts[0], resolver, chain[0], tuple(chain), text)
        if not rules.routes[route].nestable:
            break
    if len(parts) > 2:
        raise MalformedName(f"slash inside the model name: {text!r}")
    inner = _route(rules, parts[0])
    if inner is not None and _peelable(rules, inner):
        raise MalformedName(f"{parts[0]!r} is a route, not a vendor: {text!r}")
    implied = rules.routes[chain[-1]].implies_vendor
    if implied and parts[0].lower() != implied and rules.policy.implied_vendor_mismatch == "error":
        raise MalformedName(
            f"route {chain[-1]!r} implies vendor {implied!r} but the name says {parts[0]!r}: {text!r}"
        )
    return _finish(rules, parts[0], parts[1], chain[0], tuple(chain), text)


def parse_text(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """Generic: lookup, then routed when the first segment is a peelable route, then canonical, then the
    resolver for a single segment."""
    value = _text(rules, text)
    hit = _lookup(rules, value, None, ())
    if hit:
        return hit
    parts = value.split("/")
    first = _route(rules, parts[0]) if len(parts) >= 2 else None
    if first is not None and _peelable(rules, first):
        return parse_routed(rules, text, resolver)
    if len(parts) > 2:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if len(parts) == 2:
        return parse_canonical(rules, text)
    return _resolve_bare(rules, parts[0], resolver, None, (), text)


def parse_pair(rules: Rules, provider: str, name: str, resolver: Resolver | None = None) -> Identity:
    """The ``task.yaml`` form: the provider is the route, the name ``vendor/model`` or bare; never lossy."""
    prov = _text(rules, provider)
    value = _text(rules, name)
    if not prov:
        raise MalformedName("empty provider")
    _check_segment(prov, "provider", provider)
    route_key = prov.lower()
    if value.lower().startswith(route_key + "/"):
        value = value[len(prov) + 1 :]  # the engine's idempotence guard: name already starts with provider/
    hit = _lookup(rules, f"{route_key}/{value}", route_key, (route_key,))
    if hit:
        return hit
    if route_key in rules.routes:
        implied = rules.routes[route_key].implies_vendor
        if "/" in value:
            vendor, model = value.split("/", 1)
            if implied and vendor.lower() != implied and rules.policy.implied_vendor_mismatch == "error":
                raise MalformedName(
                    f"provider {provider!r} implies vendor {implied!r} but the name says {vendor!r}"
                )
            return _finish(rules, vendor, model, route_key, (route_key,), name)
        if implied:
            return _finish(rules, implied, value, route_key, (route_key,), name)
        return _resolve_bare(rules, value, resolver, route_key, (route_key,), name)
    if rules.policy.unknown_provider == "error":
        raise UnknownRoute(f"unknown provider {provider!r}")
    if "/" in value:
        vendor, model = value.split("/", 1)
        return _finish(rules, vendor, model, route_key, (route_key,), name)
    return _finish(rules, prov, value, route_key, (route_key,), name)
