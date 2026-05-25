"""v0.69.0 Part A — `soup build` dbt-for-SFT DAG parser + topo sort + plan."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import build_dag


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Allowlist + immutability
# -----------------------------------------------------------------------------


class TestSupportedKinds:
    def test_supported_kinds_exact(self) -> None:
        assert build_dag.SUPPORTED_MODEL_KINDS == frozenset(
            {"incremental", "table", "view"}
        )

    def test_supported_kinds_is_frozenset(self) -> None:
        assert isinstance(build_dag.SUPPORTED_MODEL_KINDS, frozenset)

    def test_supported_kinds_immutable(self) -> None:
        with pytest.raises(AttributeError):
            build_dag.SUPPORTED_MODEL_KINDS.add("evil")  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# validate_model_kind
# -----------------------------------------------------------------------------


class TestValidateModelKind:
    def test_happy_path(self) -> None:
        assert build_dag.validate_model_kind("incremental") == "incremental"
        assert build_dag.validate_model_kind("table") == "table"
        assert build_dag.validate_model_kind("view") == "view"

    def test_case_insensitive(self) -> None:
        assert build_dag.validate_model_kind("Incremental") == "incremental"
        assert build_dag.validate_model_kind("VIEW") == "view"

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown model kind"):
            build_dag.validate_model_kind("ephemeral")

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            build_dag.validate_model_kind(123)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            build_dag.validate_model_kind(True)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_kind("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            build_dag.validate_model_kind("incremental\x00x")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_kind("x" * 100)


# -----------------------------------------------------------------------------
# validate_model_name
# -----------------------------------------------------------------------------


class TestValidateModelName:
    def test_happy(self) -> None:
        assert build_dag.validate_model_name("raw_chat") == "raw_chat"
        assert build_dag.validate_model_name("step-1") == "step-1"
        assert build_dag.validate_model_name("a.b") == "a.b"

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            build_dag.validate_model_name(42)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            build_dag.validate_model_name(True)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_name("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            build_dag.validate_model_name("a\x00b")

    def test_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_name("../etc")

    def test_path_separator(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_name("a/b")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            build_dag.validate_model_name("a" * 200)


# -----------------------------------------------------------------------------
# BuildModel frozen + validation
# -----------------------------------------------------------------------------


class TestBuildModel:
    def test_happy(self) -> None:
        model = build_dag.BuildModel(
            name="raw",
            kind="incremental",
            transform="identity",
            refs=(),
            source="data/raw.jsonl",
            config={},
        )
        assert model.name == "raw"
        assert model.kind == "incremental"
        assert model.refs == ()

    def test_frozen(self) -> None:
        model = build_dag.BuildModel(
            name="raw",
            kind="incremental",
            transform="identity",
            refs=(),
            source="data/raw.jsonl",
            config={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.name = "evil"  # type: ignore[misc]

    def test_refs_must_be_tuple(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            build_dag.BuildModel(
                name="x",
                kind="incremental",
                transform="identity",
                refs=["y"],  # type: ignore[arg-type]
                source=None,
                config={},
            )

    def test_invalid_kind_propagates(self) -> None:
        with pytest.raises(ValueError):
            build_dag.BuildModel(
                name="raw",
                kind="ephemeral",
                transform="identity",
                refs=(),
                source=None,
                config={},
            )

    def test_invalid_name_propagates(self) -> None:
        with pytest.raises(ValueError):
            build_dag.BuildModel(
                name="../etc",
                kind="incremental",
                transform="identity",
                refs=(),
                source=None,
                config={},
            )


# -----------------------------------------------------------------------------
# Plan parsing + topo sort
# -----------------------------------------------------------------------------


class TestParseBuildPlan:
    def test_happy_linear(self) -> None:
        raw = {
            "models": [
                {
                    "name": "raw",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "identity",
                },
                {
                    "name": "filtered",
                    "kind": "incremental",
                    "refs": ["raw"],
                    "transform": "filter_low_quality",
                },
                {
                    "name": "tokenized",
                    "kind": "incremental",
                    "refs": ["filtered"],
                    "transform": "tokenize",
                },
            ]
        }
        plan = build_dag.parse_build_plan(raw)
        assert plan.topo_order == ("raw", "filtered", "tokenized")
        assert len(plan.models) == 3

    def test_happy_diamond(self) -> None:
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "source": "data/seed.jsonl",
                    "transform": "x",
                },
                {"name": "b", "kind": "incremental", "refs": ["a"], "transform": "x"},
                {"name": "c", "kind": "incremental", "refs": ["a"], "transform": "x"},
                {
                    "name": "d",
                    "kind": "incremental",
                    "refs": ["b", "c"],
                    "transform": "x",
                },
            ]
        }
        plan = build_dag.parse_build_plan(raw)
        # a must come first, d must come last
        assert plan.topo_order[0] == "a"
        assert plan.topo_order[-1] == "d"

    def test_non_dict(self) -> None:
        with pytest.raises(TypeError):
            build_dag.parse_build_plan(["models"])  # type: ignore[arg-type]

    def test_empty_models_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_dag.parse_build_plan({"models": []})

    def test_missing_models_key(self) -> None:
        with pytest.raises(ValueError):
            build_dag.parse_build_plan({})

    def test_models_not_list(self) -> None:
        with pytest.raises(ValueError):
            build_dag.parse_build_plan({"models": "raw_chat"})

    def test_cycle_rejected(self) -> None:
        # Both have refs → no source; cycle rejection fires inside topo sort.
        raw = {
            "models": [
                {"name": "a", "kind": "incremental", "refs": ["b"], "transform": "x"},
                {"name": "b", "kind": "incremental", "refs": ["a"], "transform": "x"},
            ]
        }
        with pytest.raises(ValueError, match="cycle"):
            build_dag.parse_build_plan(raw)

    def test_self_loop_rejected(self) -> None:
        # refs=["a"] but self-ref also implies refs (not seed); cross-validator
        # accepts (refs set + source None), self-loop fires in second pass.
        raw = {
            "models": [
                {"name": "a", "kind": "incremental", "refs": ["a"], "transform": "x"},
            ]
        }
        with pytest.raises(ValueError, match="self"):
            build_dag.parse_build_plan(raw)

    def test_dangling_ref_rejected(self) -> None:
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "refs": ["missing"],
                    "transform": "x",
                },
            ]
        }
        with pytest.raises(ValueError, match="missing|unknown|not found"):
            build_dag.parse_build_plan(raw)

    def test_duplicate_name_rejected(self) -> None:
        # Duplicate-name check fires BEFORE BuildModel construction, so the
        # seed/derived cross-validator is not reached. Use minimal shape.
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                },
                {
                    "name": "a",
                    "kind": "table",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                },
            ]
        }
        with pytest.raises(ValueError, match="duplicate"):
            build_dag.parse_build_plan(raw)

    def test_oversize_models(self) -> None:
        # Many-model cap fires BEFORE BuildModel construction.
        raw = {
            "models": [
                {
                    "name": f"m{i}",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                }
                for i in range(build_dag._MAX_MODELS + 1)
            ]
        }
        with pytest.raises(ValueError, match="exceeds"):
            build_dag.parse_build_plan(raw)

    def test_duplicate_ref_in_same_model_rejected(self) -> None:
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                },
                {
                    "name": "b",
                    "kind": "incremental",
                    "refs": ["a", "a"],
                    "transform": "x",
                },
            ]
        }
        with pytest.raises(ValueError, match="duplicate"):
            build_dag.parse_build_plan(raw)

    def test_seed_without_source_rejected(self) -> None:
        # New cross-validator: refs=() + source=None is degenerate.
        raw = {
            "models": [
                {"name": "a", "kind": "incremental", "transform": "x"},
            ]
        }
        with pytest.raises(ValueError, match="source"):
            build_dag.parse_build_plan(raw)

    def test_refs_and_source_mutually_exclusive(self) -> None:
        # New cross-validator: refs + source together rejected.
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                },
                {
                    "name": "b",
                    "kind": "incremental",
                    "refs": ["a"],
                    "source": "data/other.jsonl",
                    "transform": "x",
                },
            ]
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_dag.parse_build_plan(raw)


# -----------------------------------------------------------------------------
# parse_build_yaml + load_build_yaml
# -----------------------------------------------------------------------------


class TestParseBuildYaml:
    def test_happy(self) -> None:
        text = (
            "models:\n"
            "  - name: raw\n"
            "    kind: incremental\n"
            "    source: data/raw.jsonl\n"
            "    transform: identity\n"
            "  - name: out\n"
            "    kind: incremental\n"
            "    refs: [raw]\n"
            "    transform: identity\n"
        )
        plan = build_dag.parse_build_yaml(text)
        assert plan.topo_order == ("raw", "out")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(ValueError, match="invalid YAML"):
            build_dag.parse_build_yaml("models: [unclosed")

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            build_dag.parse_build_yaml(42)  # type: ignore[arg-type]

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            build_dag.parse_build_yaml("models:\x00")

    def test_oversize(self) -> None:
        text = "models:\n" + ("  - name: x\n    kind: incremental\n    transform: y\n" * 100000)
        with pytest.raises(ValueError, match="exceeds"):
            build_dag.parse_build_yaml(text)


class TestLoadBuildYaml:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        text = (
            "models:\n"
            "  - name: raw\n"
            "    kind: incremental\n"
            "    source: data/raw.jsonl\n"
            "    transform: identity\n"
        )
        path = _write(tmp_path / "build.yaml", text)
        plan = build_dag.load_build_yaml(str(path))
        assert plan.topo_order == ("raw",)

    def test_outside_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "build.yaml", "models: []")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match="cwd"):
            build_dag.load_build_yaml(str(outside / "build.yaml"))

    def test_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_dag.load_build_yaml(str(tmp_path / "nope.yaml"))

    def test_null_byte(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null"):
            build_dag.load_build_yaml("a\x00b.yaml")

    def test_empty_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            build_dag.load_build_yaml("")

    def test_non_string(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            build_dag.load_build_yaml(42)  # type: ignore[arg-type]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = _write(
            tmp_path / "real.yaml",
            "models:\n  - name: a\n    kind: incremental\n    transform: x\n",
        )
        link = tmp_path / "link.yaml"
        os.symlink(str(target), str(link))
        with pytest.raises(ValueError, match="symlink"):
            build_dag.load_build_yaml(str(link))


# -----------------------------------------------------------------------------
# Incremental diff (re-tokenize only changed rows)
# -----------------------------------------------------------------------------


class TestRowHash:
    def test_deterministic(self) -> None:
        row = {"id": "1", "text": "hello"}
        assert build_dag.compute_row_hash(row) == build_dag.compute_row_hash(row)

    def test_key_order_independent(self) -> None:
        a = {"id": "1", "text": "hello"}
        b = {"text": "hello", "id": "1"}
        assert build_dag.compute_row_hash(a) == build_dag.compute_row_hash(b)

    def test_content_sensitive(self) -> None:
        a = {"id": "1", "text": "hello"}
        b = {"id": "1", "text": "world"}
        assert build_dag.compute_row_hash(a) != build_dag.compute_row_hash(b)

    def test_non_dict(self) -> None:
        with pytest.raises(TypeError):
            build_dag.compute_row_hash([1, 2])  # type: ignore[arg-type]


class TestIncrementalDiff:
    def test_all_new(self) -> None:
        new = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
        report = build_dag.incremental_diff([], new)
        assert report.added == 2
        assert report.changed == 0
        assert report.removed == 0
        assert report.unchanged == 0

    def test_all_unchanged(self) -> None:
        rows = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
        report = build_dag.incremental_diff(rows, rows)
        assert report.unchanged == 2
        assert report.added == 0
        assert report.changed == 0
        assert report.removed == 0

    def test_changed_row(self) -> None:
        prev = [{"id": "1", "text": "a"}]
        new = [{"id": "1", "text": "b"}]
        report = build_dag.incremental_diff(prev, new)
        assert report.changed == 1
        assert report.added == 0

    def test_removed_row(self) -> None:
        prev = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
        new = [{"id": "1", "text": "a"}]
        report = build_dag.incremental_diff(prev, new)
        assert report.removed == 1
        assert report.unchanged == 1

    def test_missing_id_field(self) -> None:
        with pytest.raises(ValueError, match="id"):
            build_dag.incremental_diff([{"text": "a"}], [])

    def test_frozen_report(self) -> None:
        report = build_dag.incremental_diff([], [])
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.added = 999  # type: ignore[misc]


class TestBuildPlanFrozen:
    """TDD review #12 — BuildPlan dataclass must be immutable."""

    def test_frozen(self) -> None:
        raw = {
            "models": [
                {
                    "name": "a",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "x",
                }
            ]
        }
        plan = build_dag.parse_build_plan(raw)
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.topo_order = ("evil",)  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Plan rendering (dry-run output)
# -----------------------------------------------------------------------------


class TestRenderPlanTable:
    def test_happy(self) -> None:
        raw = {
            "models": [
                {
                    "name": "raw",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "identity",
                },
                {
                    "name": "filtered",
                    "kind": "incremental",
                    "refs": ["raw"],
                    "transform": "filter",
                },
            ]
        }
        plan = build_dag.parse_build_plan(raw)
        rendered = build_dag.render_plan_table(plan)
        assert "raw" in rendered
        assert "filtered" in rendered
        # topo order preserved
        assert rendered.index("raw") < rendered.index("filtered")

    def test_non_plan_type(self) -> None:
        with pytest.raises(TypeError):
            build_dag.render_plan_table({"models": []})  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# run_build (deferred-live)
# -----------------------------------------------------------------------------


class TestRunBuild:
    def test_deferred(self) -> None:
        raw = {
            "models": [
                {
                    "name": "raw",
                    "kind": "incremental",
                    "source": "data/raw.jsonl",
                    "transform": "identity",
                },
            ]
        }
        plan = build_dag.parse_build_plan(raw)
        with pytest.raises(NotImplementedError, match="v0.69.1"):
            build_dag.run_build(plan)

    def test_run_build_validates_plan_type(self) -> None:
        with pytest.raises(TypeError):
            build_dag.run_build({"models": []})  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# CLI: `soup build`
# -----------------------------------------------------------------------------


class TestSoupBuildCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0, result.output
        assert "build" in result.output.lower()

    def test_dry_run_happy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(
            tmp_path / "build.yaml",
            "models:\n"
            "  - name: raw\n"
            "    kind: incremental\n"
            "    source: data/raw.jsonl\n"
            "    transform: identity\n"
            "  - name: filtered\n"
            "    kind: incremental\n"
            "    refs: [raw]\n"
            "    transform: filter\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["build", str(path), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "raw" in result.output
        assert "filtered" in result.output

    def test_missing_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["build", "nope.yaml", "--dry-run"])
        assert result.exit_code != 0

    def test_live_deferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(
            tmp_path / "build.yaml",
            "models:\n  - name: raw\n    kind: incremental\n"
            "    source: data/raw.jsonl\n    transform: identity\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["build", str(path)])
        # deferred runner exits non-zero with v0.69.1 marker
        assert result.exit_code != 0
        assert "0.69.1" in result.output or "deferred" in result.output.lower()

    def test_outside_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(
            outside / "build.yaml",
            "models:\n  - name: a\n    kind: incremental\n    transform: x\n",
        )
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        runner = CliRunner()
        result = runner.invoke(app, ["build", str(outside / "build.yaml"), "--dry-run"])
        assert result.exit_code != 0


# -----------------------------------------------------------------------------
# Source-grep wiring
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_cli_registers_build(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cli = (root / "soup_cli" / "cli.py").read_text(encoding="utf-8")
        assert (
            'name="build"' in cli
            or "build_cmd" in cli
            or "from soup_cli.commands import build" in cli
        )

    def test_no_heavy_top_level_imports(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "build_dag.py").read_text(encoding="utf-8")
        # yaml is also lazy-imported inside parse_build_yaml (TDD review H2).
        for forbidden in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport peft",
            "\nimport yaml",
        ):
            assert forbidden not in src, f"top-level import found: {forbidden!r}"

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        major_minor = tuple(int(x) for x in __version__.split(".")[:2])
        assert major_minor >= (0, 69)
