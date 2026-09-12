"""``python -m toloka_model_name_normalizer``: parse a reference and print the value as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from toloka_model_name_normalizer.errors import ModelNameError, RulesError
from toloka_model_name_normalizer.normalizer import Normalizer
from toloka_model_name_normalizer.rules import Rules


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toloka-model-name-normalizer", description=__doc__)
    parser.add_argument("--rules", help="override rules file (TOML), merged over the packaged default")
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse", help="generic entry point")
    parse.add_argument("text")
    pair = sub.add_parser("pair", help="the (provider, name) form of task.yaml")
    pair.add_argument("provider")
    pair.add_argument("name")
    canonical = sub.add_parser("canonical", help="vendor/model with no route")
    canonical.add_argument("text")
    routed = sub.add_parser("routed", help="the wire form, first segment a declared route")
    routed.add_argument("text")
    sub.add_parser("rules", help="print the rule set's version and fingerprints")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rules = Rules.load(args.rules) if args.rules else Rules.from_env()
    except (RulesError, OSError) as exc:
        print(f"rules error: {exc}", file=sys.stderr)
        return 2
    if args.command == "rules":
        summary = {
            "version": rules.version,
            "schema_version": rules.schema_version,
            "identity_fingerprint": rules.identity_fingerprint,
            "fields_fingerprint": rules.fields_fingerprint,
        }
        print(json.dumps(summary, indent=2))
        return 0
    normalizer = Normalizer(rules)
    try:
        if args.command == "parse":
            name = normalizer.parse(args.text)
        elif args.command == "pair":
            name = normalizer.parse_pair(args.provider, args.name)
        elif args.command == "canonical":
            name = normalizer.parse_canonical(args.text)
        else:
            name = normalizer.parse_routed(args.text)
    except ModelNameError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(name.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
