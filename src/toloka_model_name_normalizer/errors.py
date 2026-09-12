"""Errors raised by the normalizer. Every parse error is a ``ValueError`` so callers that treat a bad
model reference like any other bad input need no special case; the subclasses carry the reason."""

from __future__ import annotations

from collections.abc import Sequence


class ModelNameError(ValueError):
    """Base of every parse error."""


class MalformedName(ModelNameError):
    """The text or pair does not fit the identity grammar (empty model, slash inside the model, bad
    vendor charset, a slug offered as input, a repeated route, an implied-vendor mismatch, ...)."""


class UnknownVendor(ModelNameError):
    """A bare name that neither a route, nor the ``bare_names`` table, nor the resolver can place."""


class AmbiguousModelName(ModelNameError):
    """The resolver returned more than one candidate for a bare name."""

    def __init__(self, bare_name: str, candidates: Sequence[str]) -> None:
        self.bare_name = bare_name
        self.candidates = tuple(candidates)
        super().__init__(f"bare name {bare_name!r} is ambiguous: {', '.join(self.candidates)}")


class UnknownRoute(ModelNameError):
    """``parse_routed`` received a first segment that is not a declared route, or the pair form named an
    undeclared provider under ``policy.unknown_provider = "error"``."""


class UnresolvedFields(ModelNameError):
    """Under ``policy.fields = "strict"``: the field parser could not place every token."""

    def __init__(self, canonical: str, unresolved: Sequence[str]) -> None:
        self.canonical = canonical
        self.unresolved = tuple(unresolved)
        super().__init__(f"{canonical}: unresolved tokens {', '.join(self.unresolved)}")


class RulesError(ValueError):
    """The rules document violates the schema, or asks for a schema version this library cannot read."""
