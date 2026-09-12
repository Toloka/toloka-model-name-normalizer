"""Renderings of an identity: the wire form and the filesystem slug. Both are one-way; nothing here is parsed
back (a slug is not injective, see the design)."""

from __future__ import annotations

from toloka_model_name_normalizer.rules import Rules


def implied_vendor(rules: Rules, route: str) -> str:
    """What the wire name of ``route`` leaves out: the declared ``implies_vendor``, or the route itself for an
    undeclared provider (a direct route to itself)."""
    declared = rules.routes.get(route)
    if declared is None:
        return route
    return declared.implies_vendor or ""


def wire_name(rules: Rules, vendor: str, model: str, route: str | None) -> str:
    """The name as it travels on ``route``: bare when the route implies the vendor or is the vendor."""
    if route is not None and (implied_vendor(rules, route) == vendor.lower() or route == vendor.lower()):
        return model
    return f"{vendor}/{model}"


def routed(rules: Rules, vendor: str, model: str, routes: tuple[str, ...]) -> str:
    """The whole route chain plus the wire name; the canonical form when there is no route."""
    if not routes:
        return f"{vendor}/{model}"
    return "/".join(routes) + "/" + wire_name(rules, vendor, model, routes[-1])


def slug(rules: Rules, vendor: str, model: str, route: str | None) -> str:
    """The identity as one filesystem-safe token: ``<vendor>_<model>`` with unsafe characters collapsed. The
    route is a property of the value, not part of the slug; ``render.route_in_slug`` switches to the engine's
    ``model_id_slug`` shape ``<route>__<wire name>`` when there is a route. Never a placeholder route."""

    def component(text: str) -> str:
        return rules.render.unsafe.sub(rules.render.unsafe_replacement, text).strip(
            rules.render.unsafe_replacement
        )

    if route is not None and rules.render.route_in_slug:
        name = component(wire_name(rules, vendor, model, route)) or "unnamed"
        return f"{component(route)}{rules.render.route_sep}{name}"
    return component(f"{vendor}/{model}") or "unnamed"
