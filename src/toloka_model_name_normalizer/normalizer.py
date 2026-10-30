"""The :class:`Normalizer` binds a rule set and a resolver and turns any carrier into a :class:`ModelName`.
Module-level functions use a lazily built default from the environment."""

from __future__ import annotations

import threading

from toloka_model_name_normalizer import identity as _identity
from toloka_model_name_normalizer.errors import UnresolvedFields
from toloka_model_name_normalizer.fields import classify
from toloka_model_name_normalizer.model import ModelName, Provenance
from toloka_model_name_normalizer.render import routed as _routed
from toloka_model_name_normalizer.render import slug as _slug
from toloka_model_name_normalizer.resolver import Resolver
from toloka_model_name_normalizer.rules import Rules


class Normalizer:
    """Parses model references under one rule set, with an optional resolver for bare names."""

    def __init__(self, rules: Rules | None = None, resolver: Resolver | None = None) -> None:
        self.rules = rules if rules is not None else Rules.default()
        self.resolver = resolver

    def parse_pair(self, provider: str, name: str) -> ModelName:
        """The ``task.yaml`` form (``provider``, ``name``); never lossy."""
        return self._build(_identity.parse_pair(self.rules, provider, name, self.resolver), "pair")

    def parse_canonical(self, text: str) -> ModelName:
        """``vendor/model`` with no route."""
        return self._build(_identity.parse_canonical(self.rules, text), "canonical")

    def parse_routed(self, text: str) -> ModelName:
        """The wire form: ``route/.../vendor/model`` or ``route/model`` for a vendor-implying route."""
        return self._build(_identity.parse_routed(self.rules, text, self.resolver), "routed")

    def parse(self, text: str) -> ModelName:
        """Generic: lookup, routed when the first segment is a peelable route, canonical, or the resolver."""
        return self._build(_identity.parse_text(self.rules, text, self.resolver), "text")

    def _build(self, ident: _identity.Identity, input_kind: str) -> ModelName:
        fields = classify(self.rules, ident.vendor, ident.model)
        if self.rules.policy.fields == "strict" and fields.unresolved:
            raise UnresolvedFields(f"{ident.vendor}/{ident.model}", fields.unresolved)
        resolver_fingerprint = getattr(self.resolver, "fingerprint", None) if ident.resolved else None
        return ModelName(
            ident.vendor,
            ident.model,
            ident.route,
            routes=ident.routes,
            family=fields.family,
            generation=fields.generation,
            tier=fields.tier,
            variant=fields.variant,
            snapshot=fields.snapshot,
            size=fields.size,
            stage=fields.stage,
            capability=fields.capability,
            api_version=fields.api_version,
            suffix=fields.suffix,
            transport_suffix=fields.transport_suffix,
            unresolved=fields.unresolved,
            wire=ident.wire,
            routed=_routed(ident.routes, ident.wire),
            slug=_slug(self.rules, ident.vendor, ident.model, ident.routes, ident.wire),
            provenance=Provenance(
                input_kind=input_kind,
                rules_version=self.rules.version,
                identity_fingerprint=self.rules.identity_fingerprint,
                fields_fingerprint=self.rules.fields_fingerprint,
                resolver_fingerprint=resolver_fingerprint,
                lookup=ident.lookup or fields.lookup,
                vendor_source=ident.vendor_source,
            ),
        )


_default_lock = threading.Lock()
_default: Normalizer | None = None


def default_normalizer() -> Normalizer:
    """The module-level normalizer: built on first use from :meth:`Rules.from_env`, never earlier."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Normalizer(Rules.from_env())
        return _default


def reset_default() -> None:
    """Forget the module-level normalizer so the next call re-reads the environment (tests)."""
    global _default
    with _default_lock:
        _default = None


def parse(text: str) -> ModelName:
    return default_normalizer().parse(text)


def parse_pair(provider: str, name: str) -> ModelName:
    return default_normalizer().parse_pair(provider, name)


def parse_canonical(text: str) -> ModelName:
    return default_normalizer().parse_canonical(text)


def parse_routed(text: str) -> ModelName:
    return default_normalizer().parse_routed(text)
