"""The value object. Identity is ``vendor``, ``model``, ``route``; everything else is descriptive and
excluded from equality and hashing. Design: README.md, section "What it does"."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_GENERATION_RE = re.compile(r"^([a-z]?)(\d+(?:\.\d+)*)?([a-z]?)$")

DESCRIPTIVE_FIELDS: tuple[str, ...] = (
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


def generation_key(generation: str | None) -> tuple[int, tuple[int, ...], str, str, str]:
    """Sort key for ``generation``: has digits, numeric tuple, prefix letter, suffix letter, raw token.

    ``4.20`` sorts after ``4.5`` (version order), ``4`` before ``4o``, ``01`` equals ``1`` on the tuple and
    differs on the raw token, letter-only generations sort before numeric ones and alphabetically among
    themselves (which is not chronology: ``command-r`` predates ``command-a``). ``None`` sorts first.
    """
    if generation is None:
        return (-1, (), "", "", "")
    match = _GENERATION_RE.match(generation)
    if not match:
        return (0, (), "", "", generation)
    digits = match.group(2)
    numbers = tuple(int(part) for part in digits.split(".")) if digits else ()
    return (1 if numbers else 0, numbers, match.group(1), match.group(3), generation)


@dataclass(frozen=True)
class Provenance:
    """Which rules, resolver and entry point produced a value."""

    input_kind: str
    rules_version: str
    identity_fingerprint: str
    fields_fingerprint: str
    resolver_fingerprint: str | None
    lookup: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind,
            "rules_version": self.rules_version,
            "identity_fingerprint": self.identity_fingerprint,
            "fields_fingerprint": self.fields_fingerprint,
            "resolver_fingerprint": self.resolver_fingerprint,
            "lookup": self.lookup,
        }


@dataclass(frozen=True)
class ModelName:
    """One model reference: identity plus rule-derived descriptive fields.

    Equality and hashing use ``vendor``, ``model`` and ``route`` only, with ``None`` a regular value.
    ``routes`` is the whole route chain as peeled, outermost first; ``route`` is its first element.
    ``routed`` and ``slug`` are renderings computed by the normalizer from the rules that parsed the value.
    """

    vendor: str
    model: str
    route: str | None
    routes: tuple[str, ...] = field(compare=False, kw_only=True)
    family: str = field(compare=False, kw_only=True)
    generation: str | None = field(compare=False, kw_only=True, default=None)
    tier: str | None = field(compare=False, kw_only=True, default=None)
    variant: str | None = field(compare=False, kw_only=True, default=None)
    snapshot: str | None = field(compare=False, kw_only=True, default=None)
    size: str | None = field(compare=False, kw_only=True, default=None)
    stage: str | None = field(compare=False, kw_only=True, default=None)
    capability: str | None = field(compare=False, kw_only=True, default=None)
    api_version: str | None = field(compare=False, kw_only=True, default=None)
    suffix: str | None = field(compare=False, kw_only=True, default=None)
    transport_suffix: bool = field(compare=False, kw_only=True, default=False)
    unresolved: tuple[str, ...] = field(compare=False, kw_only=True, default=())
    routed: str = field(compare=False, kw_only=True)
    slug: str = field(compare=False, kw_only=True)
    provenance: Provenance = field(compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if not self.vendor or not self.model:
            raise ValueError("vendor and model must be non-empty")
        if self.routes and self.routes[0] != self.route:
            raise ValueError("route must be the first element of routes")
        if not self.routes and self.route is not None:
            raise ValueError("route without a route chain")

    @property
    def canonical(self) -> str:
        """``vendor/model``, the route-blind identity string: pricing keys, tags, collector values."""
        return f"{self.vendor}/{self.model}"

    def same_model(self, other: ModelName) -> bool:
        """Same vendor and model, routes ignored."""
        return self.vendor == other.vendor and self.model == other.model

    def same_wire(self, other: ModelName) -> bool:
        """Equal and the same route chain, so the same ``routed`` rendering."""
        return self == other and self.routes == other.routes

    def generation_key(self) -> tuple[int, tuple[int, ...], str, str, str]:
        return generation_key(self.generation)

    def tags(self) -> list[str]:
        """The identity-side Langfuse tags: the three facets that are frozen with the identity rules."""
        return [
            f"model:{self.canonical}",
            f"model_vendor:{self.vendor}",
            f"model_family:{self.family}",
        ]

    def fields(self) -> dict[str, str | None]:
        """Every static and dynamic descriptive field with an explicit ``None``, for metadata."""
        return {
            "vendor": self.vendor,
            "family": self.family,
            "generation": self.generation,
            "tier": self.tier,
            "variant": self.variant,
            "snapshot": self.snapshot,
            "size": self.size,
            "stage": self.stage,
            "capability": self.capability,
            "api_version": self.api_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "model": self.model,
            "route": self.route,
            "routes": list(self.routes),
            "canonical": self.canonical,
            "routed": self.routed,
            "slug": self.slug,
            "fields": self.fields(),
            "suffix": self.suffix,
            "transport_suffix": self.transport_suffix,
            "unresolved": list(self.unresolved),
            "provenance": self.provenance.to_dict(),
        }
