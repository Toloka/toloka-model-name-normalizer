"""The token classifier: descriptive fields from the model name, fully driven by ``rules.fields``. Walks the
tokens left to right; a token no rule claims lands in ``variant``; a second generation-shaped token is kept in
``variant`` and reported in ``unresolved``; a second snapshot or API version is only reported."""

from __future__ import annotations

import re
from dataclasses import dataclass

from toloka_model_name_normalizer.rules import DateLayout, Rules, VendorSwitches

_GLUED_RE = re.compile(r"^([a-z]+)(\d.*)$")
_LETTER_SIZE_RE = re.compile(r"^([a-z])(\d+(?:\.\d+)?[bt])$")
_DIGITS_RE = re.compile(r"^\d+$")
_TWO_DIGITS_RE = re.compile(r"^\d{2}$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_V_INT_RE = re.compile(r"^v\d+$")


@dataclass(frozen=True)
class FieldSet:
    family: str
    generation: str | None
    tier: str | None
    variant: str | None
    snapshot: str | None
    size: str | None
    stage: str | None
    capability: str | None
    api_version: str | None
    suffix: str | None
    transport_suffix: bool
    unresolved: tuple[str, ...]
    lookup: bool


class _Draft:
    """Mutable accumulator while the token walk runs."""

    def __init__(self) -> None:
        self.family = ""
        self.generation: str | None = None
        self.snapshot: str | None = None
        self.api_version: str | None = None
        self.suffix: str | None = None
        self.transport_suffix = False
        self.lists: dict[str, list[str]] = {
            "tier": [],
            "variant": [],
            "size": [],
            "stage": [],
            "capability": [],
        }
        self.unresolved: list[str] = []

    def add(self, field: str, value: str) -> None:
        self.lists[field].append(value)

    def set_snapshot(self, value: str) -> None:
        if self.snapshot is None:
            self.snapshot = value
        else:
            self.unresolved.append(f"second-snapshot:{value}")

    def set_api_version(self, value: str) -> None:
        if self.api_version is None:
            self.api_version = value
        else:
            self.unresolved.append(f"second-api-version:{value}")


def _valid_date(year: int | None, month: int | None, day: int | None) -> bool:
    return (
        (year is None or 1900 <= year <= 2099)
        and (month is None or 1 <= month <= 12)
        and (day is None or 1 <= day <= 31)
    )


def _match_date(layout: DateLayout, tokens: list[str], index: int, draft: _Draft) -> int:
    """Tokens consumed by ``layout`` at ``index`` (0 when it does not match). A matching but impossible
    date is consumed and reported in ``unresolved``, never stored as a snapshot."""
    if layout.requires_generation and draft.generation is None:
        return 0
    if index + len(layout.tokens) > len(tokens):
        return 0
    values: dict[str, str] = {}
    for offset, component in enumerate(layout.tokens):
        token = tokens[index + offset]
        if component == "YYYY":
            if not _YEAR_RE.match(token):
                return 0
        elif not _TWO_DIGITS_RE.match(token):
            return 0
        values[component] = token
    year = int(values["YYYY"]) if "YYYY" in values else (2000 + int(values["YY"]) if "YY" in values else None)
    month = int(values["MM"]) if "MM" in values else None
    day = int(values["DD"]) if "DD" in values else None
    rendered = "-".join(values[component] for component in ("YYYY", "YY", "MM", "DD") if component in values)
    if _valid_date(year, month, day):
        draft.set_snapshot(rendered)
    else:
        draft.unresolved.append(f"bad-date:{rendered}")
    return len(layout.tokens)


def _shape(rules: Rules, token: str) -> str | None:
    for kind, patterns in rules.fields.shapes.items():
        if any(pattern.match(token) for pattern in patterns):
            return kind
    return None


def _is_snapshot_shaped(rules: Rules, token: str) -> bool:
    return _shape(rules, token) in ("snapshot4", "snapshot8")


def _four_digit(token: str, switches: VendorSwitches, draft: _Draft) -> str | None:
    """A 4-digit token: a snapshot under the vendor's convention (validated), else a generation kind."""
    mode = switches.four_digit_snapshot
    if mode is None:
        return "generation" if draft.generation is None else "unresolved-4digit"
    if mode == "MMDD":
        valid = _valid_date(None, int(token[:2]), int(token[2:]))
    else:
        valid = _valid_date(2000 + int(token[:2]), int(token[2:]), None)
    if not valid:
        draft.unresolved.append(f"bad-date:{token}")
    else:
        draft.set_snapshot(token)
    return None


def _extract_compounds(rules: Rules, rest: list[str], draft: _Draft) -> list[str]:
    """Remove every configured compound that appears as a run of whole tokens; longer compounds first."""
    for compound, target in sorted(rules.fields.compounds.items(), key=lambda item: -len(item[0])):
        parts = compound.split("-")
        index = 0
        while index + len(parts) <= len(rest):
            if rest[index : index + len(parts)] == parts:
                draft.add(target, compound)
                rest = rest[:index] + rest[index + len(parts) :]
            else:
                index += 1
    return rest


def classify(rules: Rules, vendor: str, model: str) -> FieldSet:
    """Decompose ``model`` (of ``vendor``) into descriptive fields under ``rules``."""
    switches = rules.fields.switches(vendor)
    lower = model.lower()
    base, _, suffix = lower.partition(":")
    raw = [token for token in rules.fields.separators.split(base) if token]
    draft = _Draft()
    if not raw:  # unreachable through the parsers, whose grammar requires a letter or digit before the suffix
        draft.unresolved.append("empty")
        return _freeze(draft, rules, vendor, model)
    glued = _GLUED_RE.match(raw[0])
    tokens = ([glued.group(1), glued.group(2)] + raw[1:]) if glued else raw
    draft.family = tokens[0]
    rest = _extract_compounds(rules, tokens[1:], draft)
    index = 0
    while index < len(rest):
        token = rest[index]
        consumed = 0
        for layout in rules.fields.dates:
            consumed = _match_date(layout, rest, index, draft)
            if consumed:
                break
        if consumed:
            index += consumed
            continue
        if (
            switches.dash_minor_version
            and _DIGITS_RE.match(token)
            and index + 1 < len(rest)
            and _DIGITS_RE.match(rest[index + 1])
            and draft.generation is None
            and not _is_snapshot_shaped(rules, token)
            and not _is_snapshot_shaped(rules, rest[index + 1])
        ):
            draft.generation = f"{token}.{rest[index + 1]}"
            index += 2
            continue
        letter_size = _LETTER_SIZE_RE.match(token)
        if letter_size and letter_size.group(1) in switches.letter_generations and draft.generation is None:
            draft.generation = letter_size.group(1)
            draft.add("size", letter_size.group(2))
            index += 1
            continue
        kind = _shape(rules, token)
        if kind == "snapshot4" and not (len(token) == 4 and token.isdigit()):
            kind = "unresolved-shape"
        if kind == "snapshot8" and not (len(token) == 8 and token.isdigit()):
            kind = "unresolved-shape"
        if kind == "api3" and not (len(token) == 3 and token.isdigit()):
            kind = "unresolved-shape"
        if kind == "snapshot4":
            kind = _four_digit(token, switches, draft)
            if kind is None:
                index += 1
                continue
        if kind == "snapshot8":
            if _valid_date(int(token[:4]), int(token[4:6]), int(token[6:])):
                draft.set_snapshot(token)
            else:
                draft.unresolved.append(f"bad-date:{token}")
            index += 1
            continue
        if kind == "api3":
            kind = "api_version" if switches.three_digit_api_version else "generation"
        if kind == "generation":
            if token in switches.letter_generations and draft.generation is None:
                draft.generation = token
            elif _V_INT_RE.match(token) and switches.trailing_v_is_api_version and index == len(rest) - 1:
                draft.set_api_version(token)
            elif draft.generation is None:
                draft.generation = token
            else:
                draft.add("variant", token)
                draft.unresolved.append(f"second-version:{token}")
            index += 1
            continue
        if kind == "api_version":
            draft.set_api_version(token)
        elif kind == "size":
            draft.add("size", token)
        elif kind == "context":
            draft.add("variant", token)
        elif kind is not None and kind.startswith("unresolved"):
            draft.unresolved.append(f"{kind}:{token}")
        elif token in switches.letter_generations and draft.generation is None:
            draft.generation = token
        elif token in rules.fields.tier:
            draft.add("tier", token)
        elif token in rules.fields.stage:
            draft.add("stage", token)
        elif token in rules.fields.capability:
            draft.add("capability", token)
        else:
            draft.add("variant", token)
        index += 1
    if suffix:
        draft.suffix = suffix
        rule = rules.fields.suffixes.get(suffix)
        if rule is not None:
            draft.add(rule.field, suffix)
            draft.transport_suffix = rule.transport
        else:
            draft.add("variant", suffix)
            draft.unresolved.append(f"unknown-suffix:{suffix}")
    return _freeze(draft, rules, vendor, model)


def _freeze(draft: _Draft, rules: Rules, vendor: str, model: str) -> FieldSet:
    def joined(field: str) -> str | None:
        return "-".join(draft.lists[field]) if draft.lists[field] else None

    values: dict[str, str | None] = {
        "family": draft.family or None,
        "generation": draft.generation,
        "tier": joined("tier"),
        "variant": joined("variant"),
        "snapshot": draft.snapshot,
        "size": joined("size"),
        "stage": joined("stage"),
        "capability": joined("capability"),
        "api_version": draft.api_version,
    }
    entry = rules.lookup.get(f"{vendor}/{model}".lower())
    lookup = False
    if entry is not None and entry.fields:
        values.update(entry.fields)
        lookup = True
    return FieldSet(
        family=values["family"] or "",
        generation=values["generation"],
        tier=values["tier"],
        variant=values["variant"],
        snapshot=values["snapshot"],
        size=values["size"],
        stage=values["stage"],
        capability=values["capability"],
        api_version=values["api_version"],
        suffix=draft.suffix,
        transport_suffix=draft.transport_suffix,
        unresolved=tuple(draft.unresolved),
        lookup=lookup,
    )
