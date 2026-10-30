"""The family of a model name: the first token after the separators and the glued-letter split, past a leading
vendor or platform token. Identity reads it to infer a bare name's vendor and fields report it, so the two can
never disagree."""

from __future__ import annotations

import re

from toloka_model_name_normalizer.rules import Rules

_GLUED_RE = re.compile(r"^([a-z]+)(\d.*)$")


def _glued(token: str) -> list[str]:
    match = _GLUED_RE.match(token)
    return [match.group(1), match.group(2)] if match else [token]


def _is_family(rules: Rules, token: str) -> bool:
    return token in rules.families or _glued(token)[0] in rules.families


def tokens(rules: Rules, model: str) -> list[str]:
    """The lower-cased tokens of ``model`` before its ``:`` suffix. A first token that names no family is a
    vendor or platform token when the second one does (``meta-llama-3.1``, ``cohere-command-a``,
    ``AI21-Jamba-1.5``), and is dropped; the first token left is split into its letters and its version
    (``qwen3`` is ``qwen``, ``3``)."""
    base = model.lower().partition(":")[0]
    raw = [token for token in rules.fields.separators.split(base) if token]
    if len(raw) >= 2 and not _is_family(rules, raw[0]) and _is_family(rules, raw[1]):
        raw = raw[1:]
    return _glued(raw[0]) + raw[1:] if raw else []


def family(rules: Rules, model: str) -> str:
    found = tokens(rules, model)
    return found[0] if found else ""


def family_vendor(rules: Rules, model: str) -> str | None:
    """The vendor the families table gives the model's family, or ``None``."""
    return rules.families.get(family(rules, model))
