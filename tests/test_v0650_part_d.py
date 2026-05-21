"""v0.65.0 Part D — CheckList behavioural DSL tests.

MFT (Minimum Functionality Test) / INV (Invariance) / DIR (Directional
Expectation) tests rendered from a YAML DSL, with per-test pass/fail.
"""
from __future__ import annotations

import os
import platform

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.utils.checklist_dsl import (
    CHECKLIST_KINDS,
    CheckListReport,
    CheckListSpec,
    CheckListTest,
    CheckListTestResult,
    load_checklist_spec,
    parse_checklist_spec,
    run_checklist_spec,
    validate_test_kind,
)


class TestKinds:
    def test_closed_set(self):
        assert CHECKLIST_KINDS == frozenset({"mft", "inv", "dir"})

    def test_immutable(self):
        assert isinstance(CHECKLIST_KINDS, frozenset)
        with pytest.raises(AttributeError):
            CHECKLIST_KINDS.add("evil")  # type: ignore[attr-defined]


class TestValidateTestKind:
    @pytest.mark.parametrize("kind", ["mft", "inv", "dir"])
    def test_known(self, kind):
        assert validate_test_kind(kind) == kind

    def test_case_insensitive(self):
        assert validate_test_kind("MFT") == "mft"

    def test_unknown(self):
        with pytest.raises(ValueError, match="kind"):
            validate_test_kind("evil")

    def test_non_string(self):
        with pytest.raises(TypeError):
            validate_test_kind(42)  # type: ignore[arg-type]

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_test_kind("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            validate_test_kind("mft\x00")

    def test_bool(self):
        with pytest.raises(TypeError):
            validate_test_kind(True)  # type: ignore[arg-type]


class TestCheckListTest:
    def test_mft_basic(self):
        t = CheckListTest(
            name="capital-france",
            kind="mft",
            prompts=("What is the capital of France?",),
            expected=("paris",),
        )
        assert t.name == "capital-france"
        assert t.kind == "mft"

    def test_frozen(self):
        t = CheckListTest(
            name="t1", kind="mft", prompts=("p",), expected=("a",),
        )
        with pytest.raises(Exception):
            t.name = "x"  # type: ignore[misc]

    def test_inv_test_no_expected_required(self):
        t = CheckListTest(
            name="paraphrase",
            kind="inv",
            prompts=("A", "B"),
            expected=(),
        )
        assert t.kind == "inv"

    def test_dir_test_requires_expected(self):
        # DIR tests need at least one expected change keyword.
        with pytest.raises(ValueError, match="expected"):
            CheckListTest(
                name="dir-test", kind="dir",
                prompts=("Add a negation",),
                expected=(),
            )

    def test_empty_prompts(self):
        with pytest.raises(ValueError, match="prompts"):
            CheckListTest(
                name="t", kind="mft", prompts=(), expected=("a",),
            )

    def test_oversize_prompts(self):
        with pytest.raises(ValueError, match="too many"):
            CheckListTest(
                name="t", kind="mft",
                prompts=tuple(f"p{i}" for i in range(10_001)),
                expected=("a",),
            )

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="name"):
            CheckListTest(
                name="", kind="mft", prompts=("p",), expected=("a",),
            )

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            CheckListTest(
                name="t", kind="evil", prompts=("p",), expected=("a",),
            )

    def test_null_byte_prompt(self):
        with pytest.raises(ValueError, match="null"):
            CheckListTest(
                name="t", kind="mft",
                prompts=("p\x00",), expected=("a",),
            )


class TestCheckListSpec:
    def test_basic(self):
        t = CheckListTest(name="t1", kind="mft", prompts=("p",), expected=("a",))
        spec = CheckListSpec(tests=(t,))
        assert len(spec.tests) == 1

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            CheckListSpec(tests=())

    def test_too_many(self):
        with pytest.raises(ValueError, match="too many"):
            CheckListSpec(tests=tuple(
                CheckListTest(name=f"t{i}", kind="mft",
                              prompts=("p",), expected=("a",))
                for i in range(1001)
            ))

    def test_duplicate_names_rejected(self):
        t1 = CheckListTest(name="t", kind="mft", prompts=("p",), expected=("a",))
        t2 = CheckListTest(name="t", kind="mft", prompts=("q",), expected=("b",))
        with pytest.raises(ValueError, match="duplicate"):
            CheckListSpec(tests=(t1, t2))


class TestParseChecklistSpec:
    def test_basic(self):
        raw = {
            "tests": [
                {"name": "t1", "kind": "mft",
                 "prompts": ["What is 2+2?"], "expected": ["4"]},
                {"name": "t2", "kind": "inv",
                 "prompts": ["What is 2+2?", "What is two plus two?"]},
            ]
        }
        spec = parse_checklist_spec(raw)
        assert len(spec.tests) == 2

    def test_missing_tests(self):
        with pytest.raises(ValueError, match="tests"):
            parse_checklist_spec({})

    def test_non_dict(self):
        with pytest.raises(TypeError):
            parse_checklist_spec([])  # type: ignore[arg-type]

    def test_test_missing_name(self):
        with pytest.raises(ValueError):
            parse_checklist_spec({"tests": [{"kind": "mft", "prompts": ["p"], "expected": ["a"]}]})

    def test_test_missing_kind(self):
        with pytest.raises(ValueError):
            parse_checklist_spec({"tests": [{"name": "t", "prompts": ["p"], "expected": ["a"]}]})


class TestLoadChecklistSpec:
    def test_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump({
            "tests": [{"name": "t1", "kind": "mft",
                       "prompts": ["p"], "expected": ["a"]}]
        }))
        spec = load_checklist_spec(str(p))
        assert len(spec.tests) == 1

    def test_outside_cwd(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        outside = tmp_path / "spec.yaml"
        outside.write_text("tests: []")
        with pytest.raises(ValueError):
            load_checklist_spec(str(outside))

    def test_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises((FileNotFoundError, OSError)):
            load_checklist_spec(str(tmp_path / "nope.yaml"))

    def test_invalid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "bad.yaml"
        p.write_text("{[not valid")
        with pytest.raises(ValueError):
            load_checklist_spec(str(p))

    def test_oversize(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "huge.yaml"
        p.write_text("x: " + "a" * (2 * 1024 * 1024))
        with pytest.raises(ValueError, match="too large"):
            load_checklist_spec(str(p))

    @pytest.mark.skipif(platform.system() == "Windows", reason="POSIX symlink")
    def test_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.yaml"
        target.write_text("tests: []")
        link = tmp_path / "link.yaml"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            load_checklist_spec(str(link))


class TestRunChecklistSpec:
    def test_mft_pass(self):
        t = CheckListTest(
            name="capital", kind="mft",
            prompts=("What is the capital of France?",),
            expected=("paris",),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "capital": ["Paris is the capital of France."],
        })
        assert isinstance(report, CheckListReport)
        assert report.results[0].verdict == "OK"
        assert report.results[0].passed == 1

    def test_mft_fail(self):
        t = CheckListTest(
            name="capital", kind="mft",
            prompts=("What is the capital of France?",),
            expected=("paris",),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "capital": ["Berlin is the capital of France."],
        })
        assert report.results[0].verdict == "MAJOR"
        assert report.results[0].passed == 0

    def test_inv_pass(self):
        t = CheckListTest(
            name="paraphrase", kind="inv",
            prompts=("Add 2 and 2.", "Add two and two."),
            expected=(),
        )
        spec = CheckListSpec(tests=(t,))
        # INV: both responses should agree.
        report = run_checklist_spec(spec, evidence={
            "paraphrase": ["The answer is 4.", "The answer is 4."],
        })
        assert report.results[0].verdict == "OK"

    def test_inv_fail(self):
        t = CheckListTest(
            name="paraphrase", kind="inv",
            prompts=("p1", "p2"),
            expected=(),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "paraphrase": ["A", "B"],
        })
        assert report.results[0].verdict == "MAJOR"

    def test_dir_pass(self):
        # DIR: response should mention "no" / "not" when prompt is negated.
        t = CheckListTest(
            name="negate", kind="dir",
            prompts=("Is the sky blue?",),
            expected=("yes",),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "negate": ["Yes, the sky is blue."],
        })
        assert report.results[0].verdict == "OK"

    def test_no_evidence(self):
        t = CheckListTest(name="t", kind="mft", prompts=("p",), expected=("a",))
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence=None)
        # No evidence -> neutral OK (matches v0.56 / v0.61 policy).
        assert report.overall == "OK"
        assert report.results[0].verdict == "OK"

    def test_partial_evidence(self):
        t1 = CheckListTest(name="t1", kind="mft", prompts=("p",), expected=("a",))
        t2 = CheckListTest(name="t2", kind="mft", prompts=("q",), expected=("b",))
        spec = CheckListSpec(tests=(t1, t2))
        report = run_checklist_spec(spec, evidence={
            "t1": ["a found"],
            # t2 has no evidence -> falls through to OK.
        })
        assert len(report.results) == 2

    def test_non_spec_type(self):
        with pytest.raises(TypeError):
            run_checklist_spec("not a spec", evidence=None)  # type: ignore[arg-type]

    def test_inv_length_mismatch(self):
        t = CheckListTest(
            name="paraphrase", kind="inv",
            prompts=("p1", "p2", "p3"),
            expected=(),
        )
        spec = CheckListSpec(tests=(t,))
        # If evidence has too few responses, surface error.
        report = run_checklist_spec(spec, evidence={
            "paraphrase": ["A", "B"],  # only 2, but 3 expected
        })
        assert report.results[0].verdict == "MAJOR"


class TestReport:
    def test_to_dict(self):
        result = CheckListTestResult(
            name="t1", kind="mft", passed=1, total=1, verdict="OK",
        )
        report = CheckListReport(results=(result,), overall="OK")
        d = report.to_dict()
        assert d["overall"] == "OK"
        assert d["results"][0]["name"] == "t1"

    def test_invalid_overall(self):
        with pytest.raises(ValueError, match="overall"):
            CheckListReport(results=(), overall="EVIL")

    def test_invalid_result_passed(self):
        with pytest.raises(ValueError, match="passed"):
            CheckListTestResult(
                name="t", kind="mft", passed=-1, total=5, verdict="OK",
            )

    def test_invalid_result_verdict(self):
        with pytest.raises(ValueError, match="verdict"):
            CheckListTestResult(
                name="t", kind="mft", passed=1, total=1, verdict="X",
            )

    def test_passed_above_total(self):
        with pytest.raises(ValueError, match="passed"):
            CheckListTestResult(
                name="t", kind="mft", passed=5, total=1, verdict="OK",
            )


class TestChecklistCli:
    def test_help_listed(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "checklist" in result.output.lower()

    def test_checklist_help(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["checklist", "--help"])
        assert result.exit_code == 0

    def test_checklist_runs(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump({
            "tests": [{"name": "t1", "kind": "mft",
                       "prompts": ["p"], "expected": ["a"]}]
        }))
        runner = CliRunner()
        result = runner.invoke(app, ["checklist", str(p)])
        assert result.exit_code == 0, (result.output, repr(result.exception))


class TestSourceWiring:
    def test_no_heavy_imports(self):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "checklist_dsl.py"
        text = src.read_text(encoding="utf-8")
        forbidden_imports = (
            "import torch\n",
            "import transformers\n",
            "from torch",
            "from transformers",
        )
        for forbidden in forbidden_imports:
            assert forbidden not in text, f"Found heavy top-level: {forbidden!r}"
