"""The closed identity grammar: ``vendor``, ``model``, ``route``, one entry point per carrier. Every decision
here is read from :class:`~toloka_model_name_normalizer.rules.Rules`."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from toloka_model_name_normalizer.errors import AmbiguousModelName, MalformedName, UnknownRoute, UnknownVendor
from toloka_model_name_normalizer.family import family, family_vendor
from toloka_model_name_normalizer.resolver import Resolver, vendor_of_candidate
from toloka_model_name_normalizer.rules import SEGMENT_RE, Route, Rules, check_model_grammar

# the first dash-separated one- or two-digit pair bounded by a separator: claude-opus-4-6, claude-3-5-sonnet;
# never the month and day of a dashed date (claude-opus-4.6-2026-01-15)
_MINOR_RE = re.compile(r"(?<![^-@:])(?<!\d{4}-)(\d{1,2})-(\d{1,2})(?![^-@:])")
_DECIMAL_P_RE = re.compile(r"(?<=\d)p(?=\d)")
_DOTTED_VENDOR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")


@dataclass(frozen=True)
class Identity:
    vendor: str
    model: str
    route: str | None
    routes: tuple[str, ...]
    wire: str
    vendor_source: str
    lookup: bool = False
    resolved: bool = False


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


def _strip(spec: Route, model: str) -> str:
    for pattern in spec.strip:
        model = pattern.sub("", model)
    return model


def _reads_as_route(rules: Rules, route: str, rest: list[str]) -> bool:
    """Whether ``route`` followed by ``rest`` is a route: always, unless the route is also a vendor and
    ``rest`` is one name whose family, read past the route's own ``strip``, is the route's own or unknown
    (``perplexity/sonar-pro`` is vendor perplexity). A name that is only deployment detail is no model of the
    route's vendor."""
    spec = rules.routes[route]
    if not spec.also_vendor or len(rest) != 1:
        return True
    name = _strip(spec, rest[0])
    return not name or family_vendor(rules, name) not in (None, route)


def _case(rules: Rules, vendor: str, model: str, text: str) -> tuple[str, str]:
    if rules.policy.vendor_case == "fold":
        vendor = vendor.lower()
    if rules.policy.case == "reject" and (vendor != vendor.lower() or model != model.lower()):
        raise MalformedName(f"mixed case refused by policy: {text!r}")
    if rules.policy.case == "fold":
        return vendor.lower(), model.lower()
    return vendor, model


def _lookup(
    rules: Rules, key: str, route: str | None, chain: tuple[str, ...], wire: str | None
) -> Identity | None:
    """A lookup identity. It bypasses the segment and route checks on purpose (the entry exists to allow what
    the grammar refuses, such as a route name as vendor) but not the case policy. Without a route there is no
    wire of the lookup's own (a config stem is not sent anywhere): the wire is the canonical form."""
    entry = rules.lookup.get(key.lower())
    if entry is None or not entry.has_identity:
        return None
    assert entry.vendor is not None and entry.model is not None
    vendor, model = _case(rules, entry.vendor, entry.model, key)
    return Identity(vendor, model, route, chain, wire or f"{vendor}/{model}", "lookup", lookup=True)


def _clean(rules: Rules, chain: tuple[str, ...], model: str, text: str) -> str:
    """The spelled model without the deployment detail of the innermost route (``strip``, ``decimal_p``),
    checked against the model grammar before and after."""
    _check_model(rules, model, text)
    route = rules.routes.get(chain[-1]) if chain else None
    if route is not None:
        model = _strip(route, model)
        if route.decimal_p:
            model = _DECIMAL_P_RE.sub(".", model)
        _check_model(rules, model, text)
    return model


def _finish(
    rules: Rules,
    vendor: str,
    model: str,
    route: str | None,
    chain: tuple[str, ...],
    text: str,
    wire: str,
    source: str,
    *,
    resolved: bool = False,
    route_as_vendor: bool = False,
) -> Identity:
    """Every non-lookup identity ends here, with a model already through :func:`_clean`: grammar, alias fold,
    family-scoped alias fold of a spelled vendor, route-as-vendor check, case policy, version dot.
    ``route_as_vendor`` is the one exception to the route check: ``policy.bare_name = "provider_as_vendor"``
    asks for the route name as vendor explicitly."""
    _check_segment(vendor, "vendor", text)
    spelled = source == "spelled"
    folded = rules.vendor_aliases.get(vendor.lower())
    if folded is not None:
        source = source if folded.lower() == vendor.lower() else "alias"
        vendor = folded
    scoped = rules.family_vendor_aliases.get(f"{vendor.lower()}/{family(rules, model)}") if spelled else None
    if scoped is not None and scoped.lower() != vendor.lower():
        vendor, source = scoped, "alias"
    named_route = _route(rules, vendor)
    if (
        named_route is not None
        and _peelable(rules, named_route)
        and not rules.routes[named_route].also_vendor
        and not route_as_vendor
    ):
        raise MalformedName(f"{vendor!r} is a route, not a vendor: {text!r}")
    vendor, model = _case(rules, vendor, model, text)
    if vendor.lower() in rules.minor_dot:
        model = _MINOR_RE.sub(r"\1.\2", model, count=1)
    return Identity(vendor, model, route, chain, wire, source, resolved=resolved)


def _split(
    rules: Rules,
    value: str,
    implied: str | None,
    route: str | None,
    chain: tuple[str, ...],
    text: str,
    wire: str,
) -> Identity:
    """``vendor/model`` under a route that may imply a vendor."""
    vendor, model = value.split("/", 1)
    if "/" in model:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if implied and vendor.lower() != implied and rules.policy.implied_vendor_mismatch == "error":
        raise MalformedName(
            f"route {route!r} implies vendor {implied!r} but the name says {vendor!r}: {text!r}"
        )
    return _finish(rules, vendor, _clean(rules, chain, model, text), route, chain, text, wire, "spelled")


def _resolve_bare(
    rules: Rules,
    bare: str,
    resolver: Resolver | None,
    route: str | None,
    chain: tuple[str, ...],
    text: str,
    wire: str,
) -> Identity:
    """lookup has run; then the ``bare_names`` patterns, the policy, the caller's resolver (an empty answer
    falls through), the families table."""
    model = _clean(rules, chain, bare, text)
    for pattern, vendor in rules.bare_names:
        if fnmatch.fnmatchcase(bare.lower(), pattern):
            return _finish(rules, vendor, model, route, chain, text, wire, "pattern")
    policy = rules.policy.bare_name
    if policy == "provider_as_vendor" and route is not None:
        return _finish(rules, route, model, route, chain, text, wire, "route", route_as_vendor=True)
    if policy == "error" or (policy == "provider_as_vendor" and route is None):
        raise UnknownVendor(
            f"bare name {bare!r}: no route implies a vendor and policy.bare_name is {policy!r}"
        )
    if resolver is not None:
        candidates = tuple(resolver.candidates(bare))
        if len(candidates) > 1:
            raise AmbiguousModelName(bare, candidates)
        if candidates:
            vendor = vendor_of_candidate(candidates[0])
            return _finish(rules, vendor, model, route, chain, text, wire, "resolver", resolved=True)
    if rules.policy.family_vendor == "infer":
        inferred = family_vendor(rules, model)
        if inferred is not None:
            return _finish(rules, inferred, model, route, chain, text, wire, "family")
    if resolver is None:
        raise UnknownVendor(f"bare name {bare!r} and no resolver")
    raise UnknownVendor(f"bare name {bare!r}: the resolver has no candidate")


def _dotted(route: Route, name: str) -> tuple[str, str] | None:
    """``[region.]vendor.model`` (Bedrock, OCI): the spelled vendor and the model, or ``None``. The model must
    start with a letter, so a dotted version (``llama3.1-70b``) never reads as a vendor."""
    head, dot, rest = name.partition(".")
    if dot and head.lower() in route.regions:
        vendor, dot, model = rest.partition(".")
        if dot and _DOTTED_VENDOR_RE.match(vendor) and model[:1].isalpha():
            return vendor, model
    if dot and _DOTTED_VENDOR_RE.match(head) and rest[:1].isalpha():
        return head, rest
    return None


def _skip(route: Route, parts: list[str]) -> list[str]:
    """``parts`` without the route's leading deployment segments; the model segment always stays."""
    moved = True
    while moved and len(parts) > 1:
        moved = False
        for path in route.skip_paths:
            if len(parts) > len(path) and all(
                fnmatch.fnmatchcase(segment.lower(), pattern)
                for segment, pattern in zip(parts[: len(path)], path, strict=True)
            ):
                parts, moved = parts[len(path) :], True
                break
        if not moved and any(
            fnmatch.fnmatchcase(parts[0].lower(), pattern) for pattern in route.skip_segments
        ):
            parts, moved = parts[1:], True
    return parts


def _bare_under(
    rules: Rules,
    route: str,
    name: str,
    resolver: Resolver | None,
    chain: tuple[str, ...],
    text: str,
    wire: str,
) -> Identity:
    """One name after the innermost route: a dotted vendor id, a route's implied vendor, or a bare name."""
    spec = rules.routes[route]
    dotted = _dotted(spec, name) if spec.dotted_vendor else None
    if dotted is not None:
        vendor, model = dotted
        return _finish(
            rules, vendor, _clean(rules, chain, model, text), chain[0], chain, text, wire, "spelled"
        )
    if spec.implies_vendor:
        model = _clean(rules, chain, name, text)
        return _finish(rules, spec.implies_vendor, model, chain[0], chain, text, wire, "route")
    return _resolve_bare(rules, name, resolver, chain[0], chain, text, wire)


def _routed(rules: Rules, value: str, resolver: Resolver | None, text: str) -> Identity:
    """The wire form with the leading slash already handled. A nestable route continues the chain when the
    next segment reads as a peelable route; the lookup runs on ``innermost_route/remainder`` after peeling, so
    a lookup identity keeps its route. The wire is the remainder after the chain, deployment segments
    included."""
    if not value:
        raise MalformedName(f"empty model reference: {text!r}")
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
    wire = value
    while len(parts) >= 2:
        route = _route(rules, parts[0])
        if route is None:
            break
        if chain and chain[-1] == route:
            raise MalformedName(f"repeated route {route!r}: {text!r}")
        if chain and not (
            rules.routes[chain[-1]].nestable
            and _peelable(rules, route)
            and _reads_as_route(rules, route, parts[1:])
        ):
            break
        chain.append(route)
        wire = "/".join(parts[1:])
        parts = _skip(rules.routes[route], parts[1:])
        if len(parts) == 1:
            found = _lookup(rules, f"{route}/{parts[0]}", chain[0], tuple(chain), wire)
            if found:
                return found
            return _bare_under(rules, route, parts[0], resolver, tuple(chain), text, wire)
        if not rules.routes[route].nestable:
            break
    if len(parts) > 2:
        if _route(rules, parts[0]) == chain[-1]:
            raise MalformedName(f"repeated route {chain[-1]!r}: {text!r}")
        raise MalformedName(f"slash inside the model name: {text!r}")
    found = _lookup(rules, f"{chain[-1]}/{'/'.join(parts)}", chain[0], tuple(chain), wire)
    if found:
        return found
    implied = rules.routes[chain[-1]].implies_vendor
    return _split(rules, "/".join(parts), implied, chain[0], tuple(chain), text, wire)


def parse_canonical(rules: Rules, text: str) -> Identity:
    """``vendor/model`` with no route: pricing keys, ``model:`` tags, collector ids. A vendor that is also a
    route (``databricks``) drops that route's deployment detail from its own model, as the routed form
    does."""
    value = _text(rules, text, what="model reference", allow_leading_slash=True)
    hit = _lookup(rules, value, None, (), None)
    if hit:
        return hit
    parts = value.split("/")
    if len(parts) != 2 or not parts[0]:
        raise MalformedName(f"canonical form needs vendor/model: {text!r}")
    route = _route(rules, parts[0])
    own = (route,) if route is not None and rules.routes[route].also_vendor else ()
    return _finish(rules, parts[0], _clean(rules, own, parts[1], text), None, (), text, value, "spelled")


def parse_routed(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """The wire form: the first segment must be a declared route, direct-vendor routes included."""
    return _routed(
        rules, _text(rules, text, what="model reference", allow_leading_slash=True), resolver, text
    )


def parse_text(rules: Rules, text: str, resolver: Resolver | None = None) -> Identity:
    """Generic: lookup, then routed when the first segment is a peelable route that reads as a route, then
    canonical, then the bare-name resolution for a single segment."""
    value = _text(rules, text, what="model reference", allow_leading_slash=True)
    hit = _lookup(rules, value, None, (), None)
    if hit:
        return hit
    parts = value.split("/")
    first = _route(rules, parts[0]) if len(parts) >= 2 else None
    if first is not None and _peelable(rules, first) and _reads_as_route(rules, first, parts[1:]):
        return _routed(rules, value, resolver, text)
    if len(parts) > 2:
        raise MalformedName(f"slash inside the model name: {text!r}")
    if len(parts) == 2:
        return parse_canonical(rules, value)
    return _resolve_bare(rules, parts[0], resolver, None, (), text, value)


def parse_pair(rules: Rules, provider: str, name: str, resolver: Resolver | None = None) -> Identity:
    """The ``task.yaml`` form: the provider is the route, the name ``vendor/model`` or bare; never lossy. A
    declared provider reads exactly as its routed string ``provider/name``, so a proxy provider continues the
    chain into a route inside the name."""
    prov = _text(rules, provider, what="provider", allow_leading_slash=False)
    value = _text(rules, name, what="model name", allow_leading_slash=False)
    if not prov:
        raise MalformedName("empty provider")
    _check_segment(prov, "provider", provider)
    route_key = prov.lower()
    if value.lower().startswith(route_key + "/"):
        value = value[len(prov) + 1 :]  # the engine's idempotence guard: name already starts with provider/
    hit = _lookup(rules, f"{route_key}/{value}", route_key, (route_key,), value)
    if hit:
        return hit
    if route_key in rules.routes:
        return _routed(rules, f"{route_key}/{value}", resolver, name)
    if rules.policy.unknown_provider == "error":
        raise UnknownRoute(f"unknown provider {provider!r} ({', '.join(sorted(rules.routes))} are declared)")
    chain = (route_key,)
    if "/" in value:
        return _split(rules, value, None, route_key, chain, name, value)
    return _finish(rules, prov, _clean(rules, chain, value, name), route_key, chain, name, value, "route")
