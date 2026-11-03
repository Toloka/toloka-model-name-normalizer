"""Renderings of a value: the wire form and the filesystem slug. Both are one-way; nothing here is parsed back
(a slug is not injective, see the design)."""

from __future__ import annotations

from toloka_model_name_normalizer.rules import Rules


def routed(routes: tuple[str, ...], wire: str) -> str:
    """The whole route chain plus the wire name as it was sent; the wire alone when there is no route."""
    return "/".join((*routes, wire)) if routes else wire


def slug(rules: Rules, vendor: str, model: str, routes: tuple[str, ...], wire: str) -> str:
    """The identity as one filesystem-safe token: ``<vendor>_<model>`` with unsafe characters collapsed. The
    route is a property of the value, not part of the slug; ``render.route_in_slug`` switches to the engine's
    ``model_id_slug`` shape ``<provider>__<name as sent>`` when there is a route. Never a placeholder
    route."""

    def component(text: str) -> str:
        return rules.render.unsafe.sub(rules.render.unsafe_replacement, text).strip(
            rules.render.unsafe_replacement
        )

    if routes and rules.render.route_in_slug:
        name = component("/".join((*routes[1:], wire))) or "unnamed"
        return f"{component(routes[0])}{rules.render.route_sep}{name}"
    return component(f"{vendor}/{model}") or "unnamed"
