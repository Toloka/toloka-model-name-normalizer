"""toloka-model-name-normalizer: one identity and rule-derived descriptive fields for every LLM model
reference. Every parsing rule is configuration (``rules.toml``); the wheel ships the Toloka default and any
deployment overrides it from outside. This is the curated public surface of the library."""

from toloka_model_name_normalizer.errors import (
    AmbiguousModelName,
    MalformedName,
    ModelNameError,
    RulesError,
    UnknownRoute,
    UnknownVendor,
    UnresolvedFields,
)
from toloka_model_name_normalizer.model import ModelName, Provenance, generation_key
from toloka_model_name_normalizer.normalizer import (
    Normalizer,
    default_normalizer,
    parse,
    parse_canonical,
    parse_pair,
    parse_routed,
    reset_default,
)
from toloka_model_name_normalizer.resolver import CallableResolver, Resolver
from toloka_model_name_normalizer.rules import (
    RULES_ENV_VAR,
    DateLayout,
    FieldRules,
    LookupEntry,
    Policy,
    Render,
    Route,
    Rules,
    Suffix,
    VendorSwitches,
)

__version__ = "0.1.0"

__all__ = [
    "RULES_ENV_VAR",
    "AmbiguousModelName",
    "CallableResolver",
    "DateLayout",
    "FieldRules",
    "LookupEntry",
    "MalformedName",
    "ModelName",
    "ModelNameError",
    "Normalizer",
    "Policy",
    "Provenance",
    "Render",
    "Resolver",
    "Route",
    "Rules",
    "RulesError",
    "Suffix",
    "UnknownRoute",
    "UnknownVendor",
    "UnresolvedFields",
    "VendorSwitches",
    "__version__",
    "default_normalizer",
    "generation_key",
    "parse",
    "parse_canonical",
    "parse_pair",
    "parse_routed",
    "reset_default",
]
