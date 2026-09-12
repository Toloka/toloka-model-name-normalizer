"""The closed identity grammar: ``vendor``, ``model``, ``route``, one entry point per carrier. Every decision
here is read from :class:`~toloka_model_name_normalizer.rules.Rules`."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from toloka_model_name_normalizer.errors import AmbiguousModelName, MalformedName, UnknownRoute, UnknownVendor
from toloka_model_name_normalizer.resolver import Resolver, vendor_of_candidate
from toloka_model_name_normalizer.rules import SEGMENT_RE, Rules, check_model_grammar


@dataclass(frozen=True)
class Identity:
    vendor: str
    model: str
    route: str | None
    routes: tuple[str, ...]
    lookup: bool
    resolved: bool


def _text(rules: Rules, value: object, *, what: str, allow_leading_slash: bool) -> str:
    """The input as a string: never trimmed (outer whitespace is an error); a leading slash is tolerated
    only on a text carrier under ``policy.leading_slash = "tolerate"``."""
    if not isinstance(value, str):
        raise TypeError(f"{what} must be str, got {type(value).__name__}")
    if value != value.strip():
        raise MalformedName(f"{what} has outer whitespace: {value!r}")
    if allow_leading_slash and value.startswith("/"):
        if rules.policy.leading_slash == "reject":
            raise MalformedName(f"leading slash: {value!r}")
        return value.lstrip("/")
    return value


def _check_segment(segment: str, what: str, text: str) -> None:
    if not SEGMENT_RE.match(segment):
        raise MalformedName(
            f"{what} {segment!r} breaks the segment grammar [A-Za-z0-9][A-Za-z0-9._-]*: {text!r}"
        )


def _check_model(rules: Rules, model: str, text: str) -> None:
    problem = check_model_grammar(model, rules.render.route_sep)
    if problem is not None:
        raise MalformedName(f"{problem}: {text!r}")


def _route(rules: Rules, segment: str) -> str | None:
    """The declared route named by ``segment`` (case-insensitive), or ``None``."""
    key = segment.lower()
    return key if key in rules.routes else None


def _peelable(rules: Rules, route: str) -> bool:
    return rules.routes[route].peelable


def _case(rules: Rules, vendor: str, model: str, text: str) -> tuple[str, str]:
    if rules.policy.case == "reject" and (vendor != vendor.lower() or model != model.lower()):
        raise MalformedName(f"mixed case refused by policy: {text!r}")
    if rules.policy.case == "fold":
        return vendor.lower(), model.lower()
    return vendor, model


def _lookup(rules: Rules, key: str, route: str | None, chain: tuple[str, ...]) -> Identity | None:
    """A lookup identity. It bypasses the segment and route checks on purpose (the entry exists to allow what
    the grammar refuses, such as a route name as vendor) but not the case policy."""
    entry = rules.lookup.get(key.lower())
    if entry is None or not entry.has_identity:
        return None
    assert entry.vendor is not None and entry.model is not None
    vendor, model = _case(rules, entry.vendor, entry.model, key)
    return Identity(vendor, model, route, chain, lookup=True, resolved=False)


def _finish(
    rules: Rules,
    vendor: str,
    model: str,
    route: str | None,
    chain: tuple[str, ...],
    text: str,
    *,
    resolved: bool = False,
    route_as_vendor: bool = False,
) -> Identity:
    """Every non-lookup identity ends here: grammar, alias fold, route-as-vendor check, case policy.
    ``route_as_vendor`` is the one exception to that check: ``policy.bare_name = "provider_as_vendor"`` asks
    for the route name as vendor explicitly."""
    _check_segment(vendor, "vendor", text)
    _check_model(rules, model, text)
    vendor = rules.vendor_aliases.get(vendor.lower(), vendor)
    named_route = _route(rules, vendor)
    if named_route is not None and _peelable(rules, named_route) and not route_as_vendor:
        raise MalformedName(f"{vendor!r} is a route, not a vendor: {text!r}")
    vendor, model = _case(rules, vendor, model, text)
    return Identity(vendor, model, route, chain, lookup=False, resolved=resolved)


def _split(
    rules: Rules, value: str, implied: str | None, route: str | None, chain: tuple[str, ...], text: str
) -> Identity:
    """``vendor/model`` under a route that may imply a vendor."""
    vendor, model = value.split("/", 1)
    if "/" in model:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if implied and vendor.lower() != implied and rules.policy.implied_vendor_mismatch == "error":
        raise MalformedName(
            f"route {route!r} implies vendor {implied!r} but the name says {vendor!r}: {text!r}"
        )
    return _finish(rules, vendor, model, route, chain, text)


def _resolve_bare(
    rules: Rules, bare: str, resolver: Resolver | None, route: str | None, chain: tuple[str, ...], text: str
) -> Identity:
    _check_model(rules, bare, text)
    for pattern, vendor in rules.bare_names:
        if fnmatch.fnmatchcase(bare.lower(), pattern):
            return _finish(rules, vendor, bare, route, chain, text)
    policy = rules.policy.bare_name
    if policy == "provider_as_vendor" and route is not None:
        return _finish(rules, route, bare, route, chain, text, route_as_vendor=True)
    if policy == "error" or (policy == "provider_as_vendor" and route is None):
        raise UnknownVendor(
            f"bare name {bare!r}: no route implies a vendor and policy.bare_name is {policy!r}"
        )
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
    value = _text(rules, text, what="model reference", allow_leading_slash=True)
    hit = _lookup(rules, value, None, ())
    if hit:
        return hit
    parts = value.split("/")
    if len(parts) != 2 or not parts[0]:
        raise MalformedName(f"canonical form needs vendor/model: {text!r}")
    return _finish(rules, parts[0], parts[1], None, (), text)


def parse_routed(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """The wire form: the first segment must be a declared route, direct-vendor routes included. The lookup
    runs on ``innermost_route/remainder`` after peeling, so a lookup identity keeps its route."""
    value = _text(rules, text, what="model reference", allow_leading_slash=True)
    parts = value.split("/")
    first = _route(rules, parts[0])
    if first is None:
        raise UnknownRoute(
            f"first segment {parts[0]!r} is not a declared route "
            f"({', '.join(sorted(rules.routes))}): {text!r}"
        )
    if len(parts) == 1:
        raise MalformedName(f"empty model name after route {first!r}: {text!r}")
    chain: list[str] = []
    while len(parts) >= 2:
        route = _route(rules, parts[0])
        if route is None:
            break
        if chain and chain[-1] == route:
            raise MalformedName(f"repeated route {route!r}: {text!r}")
        if chain and not (rules.routes[chain[-1]].nestable and _peelable(rules, route)):
            break
        chain.append(route)
        parts = parts[1:]
        if len(parts) == 1:
            found = _lookup(rules, f"{route}/{parts[0]}", chain[0], tuple(chain))
            if found:
                return found
            implied = rules.routes[route].implies_vendor
            if implied:
                return _finish(rules, implied, parts[0], chain[0], tuple(chain), text)
            return _resolve_bare(rules, parts[0], resolver, chain[0], tuple(chain), text)
        if not rules.routes[route].nestable:
            break
    if len(parts) > 2:
        if _route(rules, parts[0]) == chain[-1]:
            raise MalformedName(f"repeated route {chain[-1]!r}: {text!r}")
        raise MalformedName(f"slash inside the model name: {text!r}")
    found = _lookup(rules, f"{chain[-1]}/{'/'.join(parts)}", chain[0], tuple(chain))
    if found:
        return found
    return _split(
        rules, "/".join(parts), rules.routes[chain[-1]].implies_vendor, chain[0], tuple(chain), text
    )


def parse_text(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """Generic: lookup, then routed when the first segment is a peelable route, then canonical, then the
    resolver for a single segment."""
    value = _text(rules, text, what="model reference", allow_leading_slash=True)
    hit = _lookup(rules, value, None, ())
    if hit:
        return hit
    parts = value.split("/")
    first = _route(rules, parts[0]) if len(parts) >= 2 else None
    if first is not None and _peelable(rules, first):
        return parse_routed(rules, value, resolver)
    if len(parts) > 2:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if len(parts) == 2:
        return parse_canonical(rules, value)
    return _resolve_bare(rules, parts[0], resolver, None, (), text)


def parse_pair(rules: Rules, provider: str, name: str, resolver: Resolver | None = None) -> Identity:
    """The ``task.yaml`` form: the provider is the route, the name ``vendor/model`` or bare; never lossy."""
    prov = _text(rules, provider, what="provider", allow_leading_slash=False)
    value = _text(rules, name, what="model name", allow_leading_slash=False)
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
            return _split(rules, value, implied, route_key, (route_key,), name)
        if implied:
            return _finish(rules, implied, value, route_key, (route_key,), name)
        return _resolve_bare(rules, value, resolver, route_key, (route_key,), name)
    if rules.policy.unknown_provider == "error":
        raise UnknownRoute(f"unknown provider {provider!r} ({', '.join(sorted(rules.routes))} are declared)")
    if "/" in value:
        return _split(rules, value, None, route_key, (route_key,), name)
    return _finish(rules, prov, value, route_key, (route_key,), name)
