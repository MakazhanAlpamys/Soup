"""Review-fix follow-up tests for v0.55.0.

Covers gaps surfaced by python-review / security-review / code-review /
tdd-guide:

- ``MappingProxyType`` immutability on ``_RECOMMENDED_SCORERS`` and
  ``_METRIC_DIRECTION``
- ``SCORER_TYPES`` is a frozenset (O(1) membership)
- Frozen invariants on every dataclass (``FrozenInstanceError``)
- Symlink-target rejection on every atomic-write surface (POSIX)
- TOCTOU unconditional ``lstat`` on every read surface
- Boundary tests on ``n_samples`` / ``ci_level`` / ``per_cluster``
- ``shlex.quote`` in the rendered pre-push hook (no hand-rolled escape)
- ``overwrite=str`` / non-bool rejection on ``write_pre_push_hook``
- Run-id + suite-path oversize rejection
- Coverage table title is markup-escaped
- Source-grep: no top-level torch/transformers/peft import in any
  v0.55.0 module
- ``GateThresholds`` validates finite + bool-as-int at construction
"""

from __future__ import annotations

import dataclasses
import os
import types as _types
from pathlib import Path

import pytest

from soup_cli.utils.canary_discovery import (
    CanarySet,
    discover_canaries,
    load_canary_set,
    write_canary_set,
)
from soup_cli.utils.eval_design import (
    SCORER_TYPES,
    design_evals_from_data,
    load_eval_design,
    write_eval_design,
)
from soup_cli.utils.eval_gate_hook import (
    GateThresholds,
    RegressionVerdict,
    decide_regression,
    paired_bootstrap_ci,
    render_pre_push_hook,
    write_pre_push_hook,
)
from soup_cli.utils.eval_lock_coverage import (
    compute_coverage,
    lock_suite,
)

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink test")


# ---------------------------------------------------------------------------
# MappingProxyType / frozenset immutability
# ---------------------------------------------------------------------------

class TestImmutableRegistries:
    def test_recommended_scorers_is_mappingproxy(self):
        from soup_cli.utils.eval_lock_coverage import _RECOMMENDED_SCORERS
        assert isinstance(_RECOMMENDED_SCORERS, _types.MappingProxyType)
        with pytest.raises(TypeError):
            _RECOMMENDED_SCORERS["evil"] = ()  # type: ignore[index]

    def test_metric_direction_is_mappingproxy(self):
        from soup_cli.utils.eval_gate_hook import _METRIC_DIRECTION
        assert isinstance(_METRIC_DIRECTION, _types.MappingProxyType)
        with pytest.raises(TypeError):
            _METRIC_DIRECTION["evil"] = 0  # type: ignore[index]

    def test_scorer_types_is_frozenset(self):
        assert isinstance(SCORER_TYPES, frozenset)
        # Membership uses set semantics, not tuple-position equality.
        assert "judge" in SCORER_TYPES
        assert "rlvr" in SCORER_TYPES


# ---------------------------------------------------------------------------
# Frozen invariants on every public dataclass
# ---------------------------------------------------------------------------

class TestFrozenDataclasses:
    def test_canary_set_frozen(self):
        c = CanarySet(
            held_out=(), adjacent_skills=(), memorization_probes=(),
            cluster_count=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.cluster_count = 5  # type: ignore[misc]

    def test_locked_suite_frozen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="x")
        locked = lock_suite(design, "evals/locked.json")
        with pytest.raises(dataclasses.FrozenInstanceError):
            locked.checksum = "deadbeef"  # type: ignore[misc]

    def test_coverage_report_frozen(self):
        design = design_evals_from_data([{"output": "x"}], goal="x")
        report = compute_coverage(design, task_category="summarization")
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.task_category = "evil"  # type: ignore[misc]

    def test_gate_thresholds_frozen(self):
        thr = GateThresholds()
        with pytest.raises(dataclasses.FrozenInstanceError):
            thr.task_accuracy = -1.0  # type: ignore[misc]

    def test_regression_verdict_frozen(self):
        verdict = decide_regression(
            "task_accuracy", [0.5] * 50, [0.5] * 50, GateThresholds(),
            n_samples=200, seed=0,
        )
        assert isinstance(verdict, RegressionVerdict)
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.regressed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GateThresholds validation at construction
# ---------------------------------------------------------------------------

class TestGateThresholdsValidation:
    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            GateThresholds(task_accuracy=float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            GateThresholds(p95_latency_ms=float("inf"))

    def test_bool_rejected_on_every_field(self):
        with pytest.raises(TypeError, match="bool"):
            GateThresholds(task_accuracy=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="bool"):
            GateThresholds(p95_latency_ms=False)  # type: ignore[arg-type]

    def test_string_rejected(self):
        with pytest.raises(TypeError, match="number"):
            GateThresholds(refusal_rate="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Atomic-write symlink rejection on every surface (POSIX)
# ---------------------------------------------------------------------------

@POSIX_ONLY
class TestSymlinkAtomicWriteRejection:
    def test_write_eval_design_rejects_symlink_target(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real-target"
        real.write_text("")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        design = design_evals_from_data([{"output": "x"}], goal="x")
        with pytest.raises(ValueError, match="symlink"):
            write_eval_design(design, "link.json")

    def test_write_canary_set_rejects_symlink_target(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real-target"
        real.write_text("")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        c = discover_canaries([{"prompt": "x"}], num_clusters=1)
        with pytest.raises(ValueError, match="symlink"):
            write_canary_set(c, "link.json")

    def test_lock_suite_rejects_symlink_target(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real-target"
        real.write_text("")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        design = design_evals_from_data([{"output": "x"}], goal="x")
        with pytest.raises(ValueError, match="symlink"):
            lock_suite(design, "link.json")

    def test_load_canary_set_rejects_symlink(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text("{}")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            load_canary_set("link.json")


# ---------------------------------------------------------------------------
# Read-side unconditional lstat (TOCTOU defence)
# ---------------------------------------------------------------------------

class TestReadSideTOCTOU:
    def test_load_eval_design_missing_file_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_eval_design("missing.json")

    def test_load_canary_set_missing_file_friendly_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_canary_set("missing.json")


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestBootstrapBoundaries:
    def test_n_samples_exact_lo_accepted(self):
        # _MIN_BOOTSTRAP_SAMPLES = 100 — must accept.
        result = paired_bootstrap_ci(
            [0.5] * 5, [0.5] * 5, n_samples=100, seed=0,
        )
        assert result is not None

    def test_n_samples_exact_hi_accepted(self):
        # _MAX_BOOTSTRAP_SAMPLES = 100_000.
        result = paired_bootstrap_ci(
            [0.5] * 5, [0.5] * 5, n_samples=100_000, seed=0,
        )
        assert result is not None

    def test_n_samples_lo_minus_one_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.5], [0.5], n_samples=99, seed=0)

    def test_ci_level_strict_open_interval(self):
        # 0.001 and 0.999 must both be accepted; 0.0 / 1.0 rejected.
        paired_bootstrap_ci(
            [0.5] * 5, [0.5] * 5, n_samples=100, ci_level=0.001, seed=0,
        )
        paired_bootstrap_ci(
            [0.5] * 5, [0.5] * 5, n_samples=100, ci_level=0.999, seed=0,
        )
        with pytest.raises(ValueError):
            paired_bootstrap_ci(
                [0.5], [0.5], n_samples=100, ci_level=0.0, seed=0,
            )

    def test_decide_regression_well_above_tolerance_not_regressed(self):
        # Delta of -0.01 with tolerance of -0.02 → CI upper ≈ -0.01,
        # which is > -0.02, so NOT regressed.
        baseline = [0.5] * 50
        thr = GateThresholds()  # task_accuracy = -0.02
        candidate = [0.49] * 50  # delta = -0.01, comfortably better than tol
        verdict = decide_regression(
            "task_accuracy", baseline, candidate, thr,
            n_samples=200, seed=0,
        )
        assert verdict.regressed is False

    def test_decide_regression_well_below_tolerance_regressed(self):
        # Delta of -0.10 with tolerance of -0.02 → CI upper ≈ -0.10,
        # which is < -0.02, so regressed.
        baseline = [0.5] * 50
        thr = GateThresholds()  # task_accuracy = -0.02
        candidate = [0.40] * 50  # delta = -0.10, worse than tol
        verdict = decide_regression(
            "task_accuracy", baseline, candidate, thr,
            n_samples=200, seed=0,
        )
        assert verdict.regressed is True


class TestPerClusterBoundaries:
    def test_per_cluster_zero_rejected(self):
        with pytest.raises(ValueError):
            discover_canaries([{"prompt": "x"}], per_cluster=0)

    def test_per_cluster_negative_rejected(self):
        with pytest.raises(ValueError):
            discover_canaries([{"prompt": "x"}], per_cluster=-1)

    def test_per_cluster_bool_rejected(self):
        with pytest.raises(TypeError):
            discover_canaries([{"prompt": "x"}], per_cluster=True)  # type: ignore[arg-type]

    def test_per_cluster_oversize_rejected(self):
        with pytest.raises(ValueError):
            discover_canaries([{"prompt": "x"}], per_cluster=999)


# ---------------------------------------------------------------------------
# shlex.quote substitution + injection resistance
# ---------------------------------------------------------------------------

class TestShellEscape:
    def test_render_uses_shlex_quote(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # A path containing both single quotes and shell metacharacters.
        (tmp_path / "weird's & name.json").write_text("{}")
        body = render_pre_push_hook(
            baseline_run_id="abc",
            suite_path="weird's & name.json",
        )
        # The dangerous `&` must NOT appear unescaped.
        # shlex.quote on POSIX wraps the whole thing in single quotes
        # and replaces inner single quotes with `'"'"'`.
        # On Windows, shlex.quote returns the input as-is when it's
        # already safe — but with `'` and `&` inside, it'll still wrap
        # in single quotes.
        assert "weird" in body
        # The `&` must never appear bare (unquoted) in the body —
        # there must be at least one ' surrounding it.
        # We accept either POSIX-style single-quote wrapping OR
        # double-quote wrapping depending on platform.
        idx = body.index("&")
        assert (
            body[idx - 1] in ("'", '"')
            or "'\"'\"'" in body  # POSIX shlex.quote idiom
        )

    def test_render_no_handrolled_escape_symbol_remains(self):
        from soup_cli.utils import eval_gate_hook as mod
        # Ensure the new helper name exists and the old function is gone.
        assert hasattr(mod, "_safe_shell_quote")
        assert not hasattr(mod, "_shell_quote")


# ---------------------------------------------------------------------------
# write_pre_push_hook — overwrite must be strict bool
# ---------------------------------------------------------------------------

class TestOverwriteValidation:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")

    def test_overwrite_str_rejected(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with pytest.raises(TypeError, match="bool"):
            write_pre_push_hook(
                baseline_run_id="r1",
                suite_path="evals/locked.json",
                hook_path="hooks/pre-push",
                overwrite="yes",  # type: ignore[arg-type]
            )

    def test_overwrite_int_one_rejected(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with pytest.raises(TypeError, match="bool"):
            write_pre_push_hook(
                baseline_run_id="r1",
                suite_path="evals/locked.json",
                hook_path="hooks/pre-push",
                overwrite=1,  # type: ignore[arg-type]
            )

    def test_overwrite_zero_rejected(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        with pytest.raises(TypeError, match="bool"):
            write_pre_push_hook(
                baseline_run_id="r1",
                suite_path="evals/locked.json",
                hook_path="hooks/pre-push",
                overwrite=0,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Oversize rejection on hook fields
# ---------------------------------------------------------------------------

class TestHookOversize:
    def test_suite_path_oversize_rejected(self):
        with pytest.raises(ValueError, match="4096"):
            render_pre_push_hook(
                baseline_run_id="r1",
                suite_path="evals/" + "a" * 5000 + ".json",
            )

    def test_run_id_oversize_rejected(self):
        with pytest.raises(ValueError):
            render_pre_push_hook(
                baseline_run_id="a" * 200,
                suite_path="evals/locked.json",
            )


# ---------------------------------------------------------------------------
# Source-grep: top-level imports do not include heavy deps
# ---------------------------------------------------------------------------

class TestNoHeavyImports:
    REPO_ROOT = Path(__file__).resolve().parent.parent

    @pytest.mark.parametrize(
        "module",
        [
            "soup_cli/utils/eval_design.py",
            "soup_cli/utils/canary_discovery.py",
            "soup_cli/utils/eval_lock_coverage.py",
            "soup_cli/utils/eval_gate_hook.py",
            "soup_cli/utils/_eval_text.py",
            "soup_cli/commands/_eval_v0550.py",
        ],
    )
    def test_no_top_level_torch_or_transformers(self, module):
        text = (self.REPO_ROOT / module).read_text(encoding="utf-8")
        # Look only at module-level (zero-indented) import lines.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if line.startswith((" ", "\t")):
                continue  # nested inside a function
            assert "import torch" not in stripped, (
                f"{module} has top-level torch import"
            )
            assert "from transformers" not in stripped, (
                f"{module} has top-level transformers import"
            )
            assert "from peft" not in stripped, (
                f"{module} has top-level peft import"
            )


# ---------------------------------------------------------------------------
# Cross-module helpers extracted (no more private import)
# ---------------------------------------------------------------------------

class TestSharedTextUtils:
    def test_shared_module_exposes_row_text_and_tokenize(self):
        from soup_cli.utils import _eval_text
        assert hasattr(_eval_text, "row_text")
        assert hasattr(_eval_text, "tokenize")

    def test_canary_no_longer_imports_from_eval_design(self):
        src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli" / "utils" / "canary_discovery.py"
        ).read_text(encoding="utf-8")
        assert "from soup_cli.utils.eval_design import _row_text" not in src
        assert "from soup_cli.utils._eval_text import" in src


# ---------------------------------------------------------------------------
# Coverage scorer_mix consistency
# ---------------------------------------------------------------------------

class TestScorerMixCompleteness:
    def test_every_scorer_type_present_with_count(self):
        # compute_coverage must emit a count for every SCORER_TYPES entry,
        # even when zero — guard against the silent "missing key passes
        # because absent equals zero" hazard.
        design = design_evals_from_data([{"output": "x"}], goal="x")
        report = compute_coverage(design, task_category="summarization")
        for scorer in SCORER_TYPES:
            assert scorer in report.scorer_mix, f"{scorer} missing"


# ---------------------------------------------------------------------------
# `soup eval against` — run-vs-run paired-bootstrap CI
# ---------------------------------------------------------------------------

class TestEvalAgainst:
    def _make_tracker(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker
        return ExperimentTracker(db_path=Path(tmp_path) / "t.db")

    def test_get_metric_series_happy(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        run_id = tracker.start_run(
            config_dict={"base": "test-model", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={"memory_total": ""},
        )
        for step, loss in enumerate([0.5, 0.4, 0.3]):
            tracker.log_metrics(run_id=run_id, step=step, loss=loss)
        series = tracker.get_metric_series(run_id, "loss")
        assert series == [0.5, 0.4, 0.3]

    def test_get_metric_series_unknown_metric_returns_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        run_id = tracker.start_run(
            config_dict={"base": "m", "task": "sft"},
            device="cpu",
            device_name="cpu",
            gpu_info={"memory_total": ""},
        )
        tracker.log_metrics(run_id=run_id, step=0, loss=0.5)
        # `task_accuracy` isn't a column → empty series.
        assert tracker.get_metric_series(run_id, "task_accuracy") == []

    def test_get_metric_series_rejects_empty_args(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        with pytest.raises(ValueError):
            tracker.get_metric_series("", "loss")
        with pytest.raises(ValueError):
            tracker.get_metric_series("run-1", "")

    def test_against_cli_help_lists_flag(self):
        import re as _re

        from typer.testing import CliRunner

        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["against", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Strip Rich ANSI escapes — macOS CI uses a narrower terminal that
        # wraps option names, breaking bare substring searches (precedent:
        # v0.53.5/v0.53.6/v0.53.8/v0.53.9 CI hardening commits).
        out = _re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--candidate" in out
        assert "--metric" in out
        assert "--json-only" in out


# ---------------------------------------------------------------------------
# Coverage table title is markup-escaped
# ---------------------------------------------------------------------------

class TestCoverageMarkupEscape:
    def test_v0550_module_escapes_task_category(self):
        src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli" / "commands" / "_eval_v0550.py"
        ).read_text(encoding="utf-8")
        # The coverage table title must wrap report.task_category in escape().
        assert "escape(report.task_category)" in src
