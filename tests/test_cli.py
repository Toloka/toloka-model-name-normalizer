"""The ``python -m`` entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toloka_model_name_normalizer.__main__ import main


def test_parse_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "openrouter/openai/gpt-6-astra"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["canonical"] == "openai/gpt-6-astra" and out["route"] == "openrouter"
    assert out["fields"]["variant"] == "astra"


def test_pair_and_modes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pair", "meta", "muse-spark-1.2"]) == 0
    assert json.loads(capsys.readouterr().out)["canonical"] == "meta/muse-spark-1.2"
    assert main(["canonical", "openai/gpt-6-astra"]) == 0
    assert json.loads(capsys.readouterr().out)["route"] is None
    assert main(["routed", "platform_openai/gpt-6-astra"]) == 0
    assert json.loads(capsys.readouterr().out)["route"] == "platform_openai"


def test_parse_error_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "muse-spark-1.1"]) == 1
    assert "UnknownVendor" in capsys.readouterr().err


def test_rules_subcommand_and_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "rules.toml"
    path.write_text('version = "cli-test"\n', "utf-8")
    assert main(["--rules", str(path), "rules"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "cli-test" and out["identity_fingerprint"].startswith("sha256:")


def test_bad_rules_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "rules.toml"
    path.write_text("catalogue = 1\n", "utf-8")
    assert main(["--rules", str(path), "parse", "openai/gpt-6-astra"]) == 2
    assert "rules error" in capsys.readouterr().err
