"""Packaged demo preference data must retain the corrected example pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.utils.data_lint import run_lint
from soup_cli.utils.demo_bundles import _bundle_source_path, get_bundle
from tests.conftest import make_cli_runner

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "src" / "soup_cli" / "data" / "_fixtures"
EXAMPLES = ROOT / "examples" / "data"
TWINS = sorted(path.name for path in FIXTURES.glob("*.jsonl") if (EXAMPLES / path.name).is_file())
PREFERENCE_BUNDLES = ("dpo_demo", "sharegpt_demo")


def test_package_example_twins_were_found() -> None:
    assert {"alpaca_tiny.jsonl", "reasoning_math.jsonl", "dpo_sample.jsonl",
            "chat_preferences.jsonl"} <= set(TWINS)


@pytest.mark.parametrize("name", TWINS)
def test_package_fixture_matches_its_example_twin(name: str) -> None:
    assert (FIXTURES / name).read_bytes() == (EXAMPLES / name).read_bytes(), name


@pytest.mark.parametrize("name", PREFERENCE_BUNDLES)
def test_resolved_preference_demo_lints_clean(name: str) -> None:
    source = Path(_bundle_source_path(get_bundle(name)))
    assert source.parent == FIXTURES
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    report = run_lint(rows, "auto")
    assert report.overall == "OK", [(check.name, check.verdict) for check in report.checks]


@pytest.mark.parametrize("name", PREFERENCE_BUNDLES)
def test_cli_copies_a_preference_demo_that_lints_clean(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = make_cli_runner()
    copied = runner.invoke(app, ["data", "demo", name, "--output", "demo.jsonl"])
    assert copied.exit_code == 0, (copied.output, repr(copied.exception))
    assert (tmp_path / "demo.jsonl").read_bytes() == (
        EXAMPLES / get_bundle(name).fixture
    ).read_bytes()
    linted = runner.invoke(app, ["data", "lint", "demo.jsonl"])
    assert linted.exit_code == 0, (linted.output, repr(linted.exception))
